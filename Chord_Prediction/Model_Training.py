# ------------------ Imports ------------------
import argparse
import json
import random
import time
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
    stratified_samples,
)
 
 
# ------------------ Args ------------------
def get_args():
    parser = argparse.ArgumentParser(description="Train multi-head chord predictor")
    parser.add_argument("--representation", choices=["chord", "triad", "tetrad"], default="chord")
    parser.add_argument("--model_type",     choices=["rnn", "gru", "lstm"],       default="rnn")
    parser.add_argument("--data_path",      type=str, default="./data/segmented_4tokens.pkl")
    parser.add_argument("--vocabs_path",    type=str, default="./data/vocabs/vocabs.pkl")
    parser.add_argument("--embed_dim",      type=int, default=16)
    parser.add_argument("--hidden_dim",     type=int, default=256)
    parser.add_argument("--num_layers",     type=int, default=1)
    parser.add_argument("--batch_size",     type=int, default=4096)
    parser.add_argument("--epochs",         type=int, default=200)
    parser.add_argument("--lr",             type=float, default=5e-3)
    parser.add_argument("--sample_size",    type=int, default=200000)
    parser.add_argument("--num",            type=int, default=0)
    parser.add_argument("--seed",           type=int, default=int(time.time()) % (2**32))
    parser.add_argument("--device",         type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


args = get_args()
 
NUMBER      = args.num
EMBED_DIM   = args.embed_dim
HIDDEN_DIM  = args.hidden_dim
NUM_LAYERS  = args.num_layers
BATCH_SIZE  = args.batch_size
EPOCHS      = args.epochs
LR          = args.lr
DEVICE      = torch.device(args.device)
SEED        = args.seed
SAMPLE_SIZE = args.sample_size
DATA_PATH   = Path(args.data_path)
VOCABS_PATH = Path(args.vocabs_path)
MODE        = args.representation
 
MODELS_DIR        = Path("./models")
BEST_MODEL_PATH   = MODELS_DIR / f"best_{args.model_type.upper()}_model.pth"
BEST_METRICS_PATH = MODELS_DIR / f"best_{args.model_type.upper()}_model_metrics.json"
 
PART_NAMES = {"triad": ["Root", "Qual", "Bass"], "tetrad": ["Root", "Qual", "Exts", "Bass"]}
 
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# ------------------ Training ------------------
def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        X, lengths = batch[0].to(device), batch[-1].to(device)
        optimizer.zero_grad()
        outputs = model(X, lengths)

        if model.mode == "chord":
            loss = loss_fn(outputs, batch[1].to(device))
        else:
            Y_chord, Y_parts = batch[1].to(device), batch[2].to(device)
            *part_logits, chord_logits = outputs
            loss = loss_fn(chord_logits, Y_chord) + sum(loss_fn(part_logits[i], Y_parts[:, i]) for i in range(Y_parts.size(1))) # Chord loss and Part loss

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)

    return total_loss / len(dataloader.dataset)


def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss, total = 0, 0
    correct_full      = 0
    correct_chord     = 0
    correct_part      = [0]*len(model.part_sizes)

    with torch.no_grad():
        for batch in dataloader:
            X, lengths = batch[0].to(device), batch[-1].to(device)
            outputs = model(X, lengths)

            if model.mode == "chord":
                Y_chord = batch[1].to(device)
                total_loss += loss_fn(outputs, Y_chord).item() * X.size(0)
                correct_chord += (outputs.argmax(dim=1) == Y_chord).sum().item()
            else:
                Y_chord, Y_parts   = batch[1].to(device), batch[2].to(device)
                *part_logits, chord_logits = outputs

                total_loss += loss_fn(chord_logits, Y_chord).item() * X.size(0)

                part_preds = [logits.argmax(dim=1) for logits in part_logits]
                full_match = torch.ones_like(part_preds[0], dtype=torch.bool)

                for i, pred in enumerate(part_preds):
                    correct_part[i] += (pred == Y_parts[:, i]).sum().item()
                    full_match      &= (pred == Y_parts[:, i])
            
                correct_chord += (chord_logits.argmax(dim=1) == Y_chord).sum().item()
                correct_full += full_match.sum().item()

            total += X.size(0)

    if model.mode == "chord":
        return total_loss / total, correct_chord / total
    else:
        return (
            total_loss / total,
            [c / total for c in correct_part],  # dynamic list
            correct_full / total,
            correct_chord / total
        )


# ------------------ Pipeline ------------------
def run_train_val_test(dataset_df, parts_map, chord_vocab_size, rep_config, epochs=EPOCHS, batch_size=BATCH_SIZE, device=DEVICE, sample_size=SAMPLE_SIZE):

    # --- Splits ---
    train_ids, val_ids, test_ids = stratified_group_split(
        dataset_df, group_col='id', label_col='label',
        val_size=0.1, test_size=0.1, seed=SEED
    )
    train_ids, val_ids, test_ids = set(train_ids), set(val_ids), set(test_ids)

    assert not (train_ids & val_ids),  "Overlap between train and val IDs"
    assert not (train_ids & test_ids), "Overlap between train and test IDs"
    assert not (val_ids  & test_ids),  "Overlap between val and test IDs"
    print(f"✅ No ID leakage  |  Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

    train_df = dataset_df[dataset_df['id'].isin(train_ids)].reset_index(drop=True)
    val_df   = dataset_df[dataset_df['id'].isin(val_ids)].reset_index(drop=True)
    test_df  = dataset_df[dataset_df['id'].isin(test_ids)].reset_index(drop=True)

    for name, split in [("Full", dataset_df), ("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            print(f"--- {name} ({len(split)} rows) ---")
            print(split["label"].value_counts(normalize=True).sort_index())

    # --- Subsample ---
    train_df = stratified_samples(train_df, label_col='label', sample_size=int(sample_size*0.8), random_state=SEED)
    val_df   = stratified_samples(val_df,   label_col='label', sample_size=int(sample_size*0.1), random_state=SEED)
    test_df  = stratified_samples(test_df,  label_col='label', sample_size=int(sample_size*0.1), random_state=SEED)

    to_seqs = lambda df: [np.array(ch, dtype=np.int64) for ch in df["chords"]]
    train_seqs, val_seqs, test_seqs = to_seqs(train_df), to_seqs(val_df), to_seqs(test_df)
    print(f"Samples  |  Train: {len(train_seqs)}, Val: {len(val_seqs)}, Test: {len(test_seqs)}")

    # --- Loaders ---
    make_dataset = lambda seqs: RandomCutDataset(seqs, MODE, parts_map=parts_map, seq_len_min=5, max_len=50)
    make_loader  = lambda ds, shuffle: DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        collate_fn=lambda b: collate_fn(b, MODE, pad_token=chord_vocab_size),
    )

    train_loader = make_loader(make_dataset(train_seqs), shuffle=True)
    val_loader   = make_loader(make_dataset(val_seqs),   shuffle=False)
    test_loader  = make_loader(make_dataset(test_seqs),  shuffle=False)

    # --- Model ---
    model = ChordModel(mode=MODE, model_type=args.model_type, part_sizes=rep_config["part_sizes"], chord_vocab_size=chord_vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, pad_token=chord_vocab_size).to(device)
    loss_fn   = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6,
    )

    # --- Training loop ---
    best_val_loss   = float("inf")
    best_model_state = None
    best_epoch      = 0
    best_val_acc    = 0
    epochs_no_improve = 0
    early_stop_patience = 10

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_out = evaluate(model, val_loader, loss_fn, device)
        current_lr = optimizer.param_groups[0]['lr']

        if MODE == "chord":
            val_loss, chord_acc = val_out
            print(
                f"Epoch {epoch}/{epochs} | Train: {train_loss:.4f} | "
                f"Val: {val_loss:.4f} | Chord Acc: {chord_acc:.4f} | "
                f"LR: {current_lr:.2e} | {time.time()-start_time:.1f}s"
            )
        else:
            val_loss, part_accs, full_acc, chord_acc = val_out
            parts_str = " | ".join(
                f"{n} Acc: {a:.4f}" for n, a in zip(PART_NAMES[MODE], part_accs)
            )
            print(
                f"Epoch {epoch}/{epochs} | Train: {train_loss:.4f} | "
                f"Val: {val_loss:.4f} | {parts_str} | "
                f"Full Acc: {full_acc:.4f} | Chord Acc: {chord_acc:.4f} | "
                f"LR: {current_lr:.2e} | {time.time()-start_time:.1f}s"
            )


        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch       = epoch
            best_val_acc     = chord_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        scheduler.step(val_loss)

        if epochs_no_improve >= early_stop_patience:
            print(f"⚠️ Early stopping at epoch {epoch} (no improvement in {early_stop_patience} epochs)")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"✅ Best model: epoch {best_epoch} | Val Loss: {best_val_loss:.4f} | Val Acc: {best_val_acc:.4f}")

    # --- Test ---
    test_out = evaluate(model, test_loader, loss_fn, device)

    if MODE == "chord":
        test_loss, chord_acc = test_out
        full_acc = None
        print(f"Test Loss: {test_loss:.4f} | Chord Acc: {chord_acc:.4f}")
    else:
        test_loss, part_accs, full_acc, chord_acc = test_out
        parts_str = " | ".join(
            f"{n} Acc: {a:.4f}" for n, a in zip(PART_NAMES[MODE], part_accs)
        )
        print(f"Test Loss: {test_loss:.4f} | {parts_str} | Full Acc: {full_acc:.4f} | Chord Acc: {chord_acc:.4f}")
 
    return model, best_val_acc, best_val_loss, full_acc, chord_acc, test_loss


# ------------------ Path helpers ------------------
def get_model_path(config: dict) -> Path:
    d = MODELS_DIR / config["model_type"].upper()
    d.mkdir(parents=True, exist_ok=True)
    return d / (
        f"{config['model_type']}_model"
        f"_embed{config['embed_dim']}"
        f"_hidden{config['hidden_dim']}"
        f"_layers{config['num_layers']}"
        f"_lr{config['lr']}"
        f"_sample{config['sample_size']}"
        f"___{config['num']}.pth"
    )

 
def get_metrics_path(config: dict) -> Path:
    d = MODELS_DIR / config["model_type"].upper() / "json"
    d.mkdir(parents=True, exist_ok=True)
    return d / (
        f"{config['model_type']}"
        f"_embed{config['embed_dim']}"
        f"_hidden{config['hidden_dim']}"
        f"_layers{config['num_layers']}"
        f"_lr{config['lr']}"
        f"_sample{config['sample_size']}"
        f"___{config['num']}.json"
    )


# ------------------ Main ------------------
if __name__ == "__main__":
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    df = pd.read_pickle(DATA_PATH)
    print(f"Loaded {len(df)} rows")

    print("Loading vocabs...")
    vocabs = pd.read_pickle(VOCABS_PATH)
    CHORD_VOCAB = len(vocabs["chord_to_idx"])

    if MODE == "chord":
        parts_map  = None
        part_sizes = [CHORD_VOCAB]
    elif MODE == "triad":
        parts_map  = vocabs["part_ids_to_chord_id_3"]
        part_sizes = [len(vocabs["root_to_idx"]), len(vocabs["qualex_to_idx"]), len(vocabs["bass_to_idx"])]
    else:  # tetrad
        parts_map  = vocabs["part_ids_to_chord_id_4"]
        part_sizes = [len(vocabs["root_to_idx"]), len(vocabs["quality_to_idx"]), len(vocabs["extensions_to_idx"]), len(vocabs["bass_to_idx"])]

    model, val_acc, val_loss, full_acc, chord_acc, test_loss = run_train_val_test(
        df, parts_map=parts_map, chord_vocab_size=CHORD_VOCAB,
        rep_config={"part_sizes": part_sizes}, sample_size=SAMPLE_SIZE,
    )

    # Build metrics dict
    config = {
        "model_type": args.model_type,
        "embed_dim": EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_layers": NUM_LAYERS,
        "lr": LR,
        "sample_size": SAMPLE_SIZE,
        "num": NUMBER
    }

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "seed":        SEED,
        "embed_dim":   EMBED_DIM,
        "hidden_dim":  HIDDEN_DIM,
        "num_layers":  NUM_LAYERS,
        "lr":          LR,
        "sample_size": SAMPLE_SIZE,
    }
    
    new_metrics = {
        "val_acc": val_acc,
        "val_loss": val_loss,
        "full_acc": full_acc,
        "chord_acc": chord_acc,
        "test_loss": test_loss,
        "seed": SEED,
    }

    # Save this run
    model_path   = get_model_path(config)
    metrics_path = get_metrics_path(config)
    torch.save(checkpoint, model_path)
    metrics_path.write_text(json.dumps(new_metrics, indent=2))

    full_str = f"full_acc={full_acc:.4f} | " if full_acc is not None else ""
    print(
        f"✅ Run saved at {model_path} — "
        f"val_acc={val_acc:.4f}, val_loss={val_loss:.4f}, "
        f"{full_str}chord_acc={chord_acc:.4f}, test_loss={test_loss:.4f}"
    )

    # Update best model if this run is better
    best_metrics = {}
    if BEST_METRICS_PATH.exists():
        try:
            best_metrics = json.loads(BEST_METRICS_PATH.read_text())
        except json.JSONDecodeError:
            print("⚠️ Existing best metrics file is invalid JSON. Resetting.")
 
    if new_metrics["val_loss"] < best_metrics.get("val_loss", float("inf")):
        torch.save(checkpoint, BEST_MODEL_PATH)
        BEST_METRICS_PATH.write_text(json.dumps(new_metrics, indent=2))
        print(
            f"✅ New best model saved at {BEST_MODEL_PATH} "
            f"(val_loss={val_loss:.4f}, chord_acc={chord_acc:.4f}, SEED={SEED})"
        )
    else:
        print(
            f"⚠️ Not a new best (previous is still better with "
            f"val_loss={best_metrics.get('val_loss', float('inf')):.4f}, "
            f"chord_acc={best_metrics.get('chord_acc', 0):.4f})"
        )
