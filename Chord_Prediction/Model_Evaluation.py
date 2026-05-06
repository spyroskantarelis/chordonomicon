# ------------------ Imports ------------------
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
 
from Helpers import (
    RandomCutDataset,
    collate_fn,
    ChordModel,
    stratified_group_split,
)
 
 
# ------------------ Args ------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate chord prediction model")
    parser.add_argument("--model_type",          choices=["rnn", "gru", "lstm"], default="rnn")
    parser.add_argument("--representation",       choices=["chord", "triad", "tetrad"], required=True)
    parser.add_argument("--model_path",           type=str, required=True,  help="Path to model checkpoint (.pth)")
    parser.add_argument("--model_name",           type=str, required=True,  help="Human-readable model name")
    parser.add_argument("--dataset_path",         type=str, default="./data/segmented_4tokens.pkl")
    parser.add_argument("--second_dataset_path",  type=str, default="./data/labeled_4tokens.pkl", help="Held-out dataset for --full_dataset")
    parser.add_argument("--vocabs_path",          type=str, default="./data/vocabs/vocabs.pkl")
    parser.add_argument("--save_dir",             type=str, default="./results")
    parser.add_argument("--batch_size",           type=int, default=4096)
    parser.add_argument("--top_n",                type=int, default=10, help="Top N mismatches to display")
    parser.add_argument("--device",               type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--full_dataset",         action="store_true", help="Evaluate on second dataset instead")
    return parser.parse_args()
 
 
args = parse_args()
 
MODE             = args.representation
MODEL_NAME       = args.model_name
N                = args.top_n
SAVE_DIR         = Path(args.save_dir)
DEVICE           = torch.device(args.device)
VOCABS_PATH      = Path(args.vocabs_path)
DATA_PATH        = Path(args.dataset_path)
SECOND_DATA_PATH = Path(args.second_dataset_path)
DATASET_NAME     = DATA_PATH.stem
FULL_DATASET     = args.full_dataset
 
checkpoint   = torch.load(args.model_path, map_location=DEVICE)
SEED         = checkpoint.get("seed", 0)
SAMPLE_SIZE  = checkpoint.get("sample_size") or 200000  # fallback if missing
BATCH_SIZE   = args.batch_size
 
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
 
PART_NAMES = {
    "triad":  ["root", "qual", "bass"],
    "tetrad": ["root", "qual", "exts", "bass"],
}


# ------------------ Evaluation ------------------
def topk_correct(topk_preds, true_labels, k):
    return (topk_preds[:, :k] == true_labels.unsqueeze(1)).any(dim=1).sum().item()


def evaluate(model, dataloader, device, bass_to_idx=None):
    model.eval()
    ce_loss = nn.CrossEntropyLoss(reduction='none')  # per-sample loss

    total_loss, total   = 0, 0
    correct_full        = 0
    correct_chord       = 0
    correct_part        = [0]*len(model.part_sizes)
    correct_bass_exists = 0
    total_bass_exists   = 0 

    correct_part_top1 = [0] * len(model.part_sizes)
    correct_part_top2 = [0] * len(model.part_sizes)
    correct_part_top3 = [0] * len(model.part_sizes)
    
    length_stats = defaultdict(lambda: {"count": 0, "correct": 0, "loss_sum": 0.0})
    mismatches = {name: defaultdict(int) for name in PART_NAMES.get(model.mode, [])}
    mismatches["chord"] = defaultdict(lambda: {"loss_true_sum": 0.0, "loss_pred_sum": 0.0, "count": 0})

    with torch.no_grad():
        for batch in dataloader:
            X, lengths = batch[0].to(device), batch[-1].to(device)
            outputs = model(X, lengths)

            if model.mode == "chord":
                Y_chord = batch[1].to(device)
                per_sample_loss = ce_loss(outputs, Y_chord)
                preds = outputs.argmax(dim=1)
                per_sample_loss_pred = ce_loss(outputs, preds)

                total_loss += per_sample_loss.sum().item()
                
                correct_chord += (preds == Y_chord).sum().item()
                chord_topk = outputs.topk(3, dim=1).indices

            else:
                Y_chord, Y_parts = batch[1].to(device), batch[2].to(device)
                num_parts = Y_parts.size(1)
                *part_logits, chord_logits = outputs

                per_sample_loss = ce_loss(chord_logits, Y_chord)
                chord_pred = chord_logits.argmax(dim=1)
                per_sample_loss_pred = ce_loss(chord_logits, chord_pred)

                total_loss += per_sample_loss.sum().item()
                correct_chord += (chord_pred == Y_chord).sum().item()

                part_preds = [logits.argmax(dim=1) for logits in part_logits]
                full_match = torch.ones_like(part_preds[0], dtype=torch.bool)

                for i, pred in enumerate(part_preds):
                    correct_part[i] += (pred == Y_parts[:, i]).sum().item()
                    full_match      &= (pred == Y_parts[:, i])
                
                correct_full += full_match.sum().item()

                # Bass accuracy (excluding N bass)
                bass_true = Y_parts[:, -1]
                bass_pred = part_preds[-1]
                bass_exists_mask = (bass_true != bass_to_idx["N"] - 1) & (bass_pred != bass_to_idx["N"] - 1)
                correct_bass_exists += ((bass_pred == bass_true) & bass_exists_mask).sum().item()
                total_bass_exists += bass_exists_mask.sum().item()

                # Top-k per part (and chord)
                part_topk = [logits.topk(3, dim=1).indices for logits in part_logits]
                chord_topk = chord_logits.topk(3, dim=1).indices

                for i in range(num_parts):
                    correct_part_top1[i] += topk_correct(part_topk[i], Y_parts[:, i], k=1)
                    correct_part_top2[i] += topk_correct(part_topk[i], Y_parts[:, i], k=2)
                    correct_part_top3[i] += topk_correct(part_topk[i], Y_parts[:, i], k=3)

            total += X.size(0)

            # Per-sample mismatch and length tracking
            preds_chord = chord_pred if model.mode != "chord" else preds
            for i in range(Y_chord.size(0)):
                true_len = int(lengths[i].cpu().item())
                true_chord = int(Y_chord[i].cpu())
                pred_chord = int(preds_chord[i].item())
                
                if model.mode != "chord":
                    for p in range(num_parts):
                        true_val = int(Y_parts[i, p].cpu())
                        pred_val = int(part_preds[p][i].cpu())
                        if true_val != pred_val:
                            mismatches[PART_NAMES[model.mode][p]][(true_val, pred_val)] += 1

                if pred_chord != true_chord:
                    key = (true_chord, pred_chord)
                    mismatches["chord"][key]["loss_true_sum"] += per_sample_loss[i].item()
                    mismatches["chord"][key]["loss_pred_sum"] += per_sample_loss_pred[i].item()
                    mismatches["chord"][key]["count"] += 1

                length_stats[true_len]["count"] += 1
                length_stats[true_len]["loss_sum"] += per_sample_loss[i].item()
                if pred_chord == true_chord:
                    length_stats[true_len]["correct"] += 1

    if total_bass_exists == 0:
        total_bass_exists = 1

    part_names = PART_NAMES.get(model.mode, [])

    for i, name in enumerate(part_names):
        print(
            f"{name.capitalize()} | "
            f"Top-1: {correct_part_top1[i]/total:.4f} | "
            f"Top-2: {correct_part_top2[i]/total:.4f} | "
            f"Top-3: {correct_part_top3[i]/total:.4f}"
        )

    metrics = {
        "loss": total_loss / total,
        "chord_acc": correct_chord / total,
        "full_acc": correct_full / total if model.mode != "chord" else None,
        "bass_ex_acc": correct_bass_exists / total_bass_exists if model.mode != "chord" else None,
    }

    for i, name in enumerate(part_names):
        metrics[f"{name}_acc"] = correct_part[i] / total

    # Add top-k per part
    """
    for i, name in enumerate(part_names):
        metrics[f"{name}_top1"] = correct_part_top1[i] / total
        metrics[f"{name}_top2"] = correct_part_top2[i] / total
        metrics[f"{name}_top3"] = correct_part_top3[i] / total
    """
    
    length_acc = {
        l: {
            "accuracy":  v["correct"] / v["count"],
            "mean_loss": v["loss_sum"] / v["count"],
            "count":     v["count"],
        }
        for l, v in sorted(length_stats.items())
        if v["count"] > 0
    }

    return metrics, mismatches, length_acc


# ------------------ Results ------------------
def decode_chord_str(parts_ids, idx_to_root, idx_to_qual, idx_to_extensions, idx_to_bass, mode):
    """Build a readable chord string from a parts_ids tuple."""
    r = idx_to_root[parts_ids[0] - 1]
    q = idx_to_qual[parts_ids[1] - 1]
    if mode == "tetrad":
        e = idx_to_extensions[parts_ids[2] - 1]
        b = idx_to_bass[parts_ids[3] - 1]
        return f"{r}:{q}{e}" + (f"/{b}" if b != "N" else "")
    else:  # triad
        b = idx_to_bass[parts_ids[2] - 1]
        return f"{r}:{q}" + (f"/{b}" if b != "N" else "")
    

def build_results_json(model_name, representation, dataset_name, sample_size, metrics, length_acc, decoded_mismatches):
    length_list = [
        {"length": int(l), "accuracy": float(v["accuracy"]),
         "mean_loss": float(v["mean_loss"]), "count": int(v["count"])}
        for l, v in sorted(length_acc.items())
    ]
 
    top_mismatches = {}
    for part, entries in decoded_mismatches.items():
        part_entries = []
        for entry in entries:
            true_label, pred_label = entry[0], entry[1]
            count = entry[2]
            e = {"true": str(true_label), "pred": str(pred_label), "count": int(count)}
            if len(entry) > 3:
                e.update({
                    "loss_true_mean": float(entry[3]),
                    "loss_pred_mean": float(entry[4]),
                    "loss_diff_mean": float(entry[5]),
                })
            part_entries.append(e)
        top_mismatches[part] = part_entries
 
    return {
        "meta": {
            "model_name":     model_name,
            "representation": representation,
            "dataset":        dataset_name,
            "sample_size":    int(sample_size),
        },
        "metrics":              {k: (float(v) if v is not None else None) for k, v in metrics.items()},
        "accuracy_per_length":  length_list,
        "top_mismatches":       top_mismatches,
    }


def save_results_json(result_dict, save_dir: Path, full_dataset: bool, second_data_path: Path):
    save_dir.mkdir(parents=True, exist_ok=True)
    meta   = result_dict["meta"]
    model  = str(meta["model_name"]).replace(" ", "_")
    reprn  = str(meta["representation"]).replace(" ", "_")
    samples = str(meta["sample_size"])
 
    suffix = ""
    if full_dataset:
        suffix = f"_{second_data_path.stem.split('_')[0]}_full"
 
    filepath = save_dir / f"{model}_{reprn}_{samples}{suffix}.json"
    filepath.write_text(json.dumps(result_dict, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ Saved results to {filepath}")


# ------------------ Main ------------------
if __name__ == "__main__":
    print("Loading dataset...")
    df = pd.read_pickle(DATA_PATH)
    
    print("Loading vocabs...")
    vocabs = pd.read_pickle(VOCABS_PATH)

    CHORD_VOCAB = len(vocabs["chord_to_idx"])

    idx_to_root       = {v-1: k for k, v in vocabs["root_to_idx"].items()}
    idx_to_quality    = {v-1: k for k, v in vocabs["quality_to_idx"].items()}
    idx_to_qualex     = {v-1: k for k, v in vocabs["qualex_to_idx"].items()}
    idx_to_extensions = {v-1: k for k, v in vocabs["extensions_to_idx"].items()}
    idx_to_bass       = {v-1: k for k, v in vocabs["bass_to_idx"].items()}
    bass_to_idx = vocabs["bass_to_idx"]

    if MODE == "chord":
        parts_map  = None
        part_sizes = [CHORD_VOCAB]
    elif MODE == "triad":
        parts_map  = vocabs["part_ids_to_chord_id_3"]
        part_sizes = [len(vocabs["root_to_idx"]), len(vocabs["qualex_to_idx"]), len(vocabs["bass_to_idx"])]
    else:  # tetrad
        parts_map  = vocabs["part_ids_to_chord_id_4"]
        part_sizes = [len(vocabs["root_to_idx"]), len(vocabs["quality_to_idx"]), len(vocabs["extensions_to_idx"]), len(vocabs["bass_to_idx"])]    

    # Recreate the exact same splits as training
    train_ids, val_ids, test_ids = stratified_group_split(df, group_col='id', label_col='label', val_size=0.1, test_size=0.1, seed=SEED)
    used_ids = set(train_ids) | set(val_ids)
    test_df_full = df[df['id'].isin(test_ids)].reset_index(drop=True)
    
    if FULL_DATASET:
        df2      = pd.read_pickle(SECOND_DATA_PATH)
        test_df  = df2[~df2["id"].isin(used_ids)].reset_index(drop=True)
        print(f"Using second dataset: {len(test_df)} rows (after removing train/val ids)")
    else:
        test_df  = test_df_full
        print(f"Using test split: {len(test_df)} rows")

    test_seqs = [np.array(ch, dtype=np.int64) for ch in test_df['chords']]
    test_dataset = RandomCutDataset(test_seqs, MODE, parts_map=parts_map, seq_len_min=5, max_len=50)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda b: collate_fn(b, MODE, CHORD_VOCAB))
        
    # Load model
    model = ChordModel(mode=MODE, model_type=args.model_type,
                      part_sizes=part_sizes,
                      chord_vocab_size=CHORD_VOCAB,
                      embed_dim=checkpoint["embed_dim"],
                      hidden_dim=checkpoint["hidden_dim"],
                      num_layers=checkpoint["num_layers"],
                      pad_token=CHORD_VOCAB).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Evaluate
    metrics, mismatches, length_acc = evaluate(model, test_loader, DEVICE, bass_to_idx=bass_to_idx)

    print("\n📊 Test Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {f'{v:.4f}' if v is not None else 'N/A'}")

    print("\n📈 Accuracy per sequence length:")
    for l, stats in length_acc.items():
        print(f"  Length {l}: acc={stats['accuracy']:.4f}, loss={stats['mean_loss']:.4f}, count={stats['count']}")

    # Decode mismatches for readability
    decoded_mismatches = {}

    if MODE != "chord":
        qual_idx_map = idx_to_qualex if MODE == "triad" else idx_to_quality
        
        chord_id_to_parts_ids = vocabs[f"chord_id_to_part_ids_{'4' if MODE == 'tetrad' else '3'}"]

        for name, idx_map in [("root", idx_to_root), ("qual", qual_idx_map)]:
            decoded_mismatches[name] = sorted(
                [(idx_map[t], idx_map[p], c) for (t, p), c in mismatches[name].items()],
                key=lambda x: x[2], reverse=True,
            )

        if MODE == "tetrad":
            decoded_mismatches["exts"] = sorted([
                (idx_to_extensions[t], idx_to_extensions[p], count)
                for (t, p), count in mismatches["exts"].items()
            ], key=lambda x: x[2], reverse=True)

        decoded_mismatches["bass"] = sorted([
            (idx_to_bass[t], idx_to_bass[p], count)
            for (t, p), count in mismatches["bass"].items()
        ], key=lambda x: x[2], reverse=True)

        decoded_mismatches["bass_exists"] = sorted([
            (idx_to_bass[t], idx_to_bass[p], count)
            for (t, p), count in mismatches["bass"].items()
            if idx_to_bass[t] != "N" and idx_to_bass[p] != "N"
        ], key=lambda x: x[2], reverse=True)

        decoded_mismatches["full_chord"] = sorted([
                    (
                        decode_chord_str(chord_id_to_parts_ids[t], idx_to_root, qual_idx_map, idx_to_extensions, idx_to_bass, MODE),
                        decode_chord_str(chord_id_to_parts_ids[p], idx_to_root, qual_idx_map, idx_to_extensions, idx_to_bass, MODE),
                        stats["count"],
                        stats["loss_true_sum"] / stats["count"],
                        stats["loss_pred_sum"] / stats["count"],
                        (stats["loss_pred_sum"] - stats["loss_true_sum"]) / stats["count"],
                    )
                    for (t, p), stats in mismatches["chord"].items()
                ], key=lambda x: x[2], reverse=True)

    else:
        idx_to_chord = {v: k for k, v in vocabs["chord_to_idx"].items()}
        decoded_mismatches["full_chord"] = sorted([
            (
                idx_to_chord[t], idx_to_chord[p],
                stats["count"],
                stats["loss_true_sum"] / stats["count"],
                stats["loss_pred_sum"] / stats["count"],
                (stats["loss_pred_sum"] - stats["loss_true_sum"]) / stats["count"]
            )
            for (t, p), stats in mismatches["chord"].items()
        ], key=lambda x: x[2], reverse=True)

    # Print top N mismatches
    print(f"\n❌ Top {N} mismatches:")
    for part, mism in decoded_mismatches.items():
        print(f"\n{part.upper()}:")
        for entry in mism[:N]:
            true, pred, count = entry[0], entry[1], entry[2]
            if len(entry) > 3:
                print(f"  True: {true:<10} | Pred: {pred:<10} | Count: {count} "
                      f"| Loss_true: {entry[3]:.4f} | Loss_pred: {entry[4]:.4f} | Loss_diff: {entry[5]:.4f}")
            else:
                print(f"  True: {true:<10} | Pred: {pred:<10} | Count: {count}")

    part_keys = {"triad": ["root", "qual", "bass"], "tetrad": ["root", "qual", "exts", "bass"]}
    json_mismatches = {}
    for key in part_keys.get(MODE, []) + (["bass_exists", "full_chord"] if MODE != "chord" else ["full_chord"]):
        out_key = "chord" if key == "full_chord" else key
        json_mismatches[out_key] = decoded_mismatches.get(key, [])

    # Build result dict
    result_dict = build_results_json(
        model_name=MODEL_NAME,
        representation=MODE,
        dataset_name=DATASET_NAME,
        sample_size=SAMPLE_SIZE,
        metrics=metrics,
        length_acc=length_acc,
        decoded_mismatches=json_mismatches
    )

    # Save JSON
    save_results_json(result_dict, save_dir=SAVE_DIR, full_dataset=FULL_DATASET, second_data_path=SECOND_DATA_PATH)
