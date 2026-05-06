# Chord Prediction

An RNN-based chord prediction system trained on the Chordonomicon dataset. Given a sequence of chords, the model predicts the next chord. Three chord representations are supported — flat token, triad (root + qualex + bass), and tetrad (root + quality + extensions + bass) — and three recurrent architectures — RNN, GRU, LSTM.

---

## Project structure

```
.
├── Chord_Embeddings.py   # Data preprocessing and vocabulary building
├── Helpers.py            # Dataset, model, and split utilities
├── Model_Training.py     # Training script
├── Model_Evaluation.py   # Evaluation script
├── Run.py                # Entry point — configure and launch train/eval
├── data/
│   ├── chordonomicon_v2.csv
│   ├── filtered_1token.pkl      # all songs, 1-token encoding
│   ├── filtered_3tokens.pkl     # all songs, 3-token encoding
│   ├── filtered_4tokens.pkl     # all songs, 4-token encoding
│   ├── labeled_1token.pkl       # labeled songs (labels stripped), 1-token
│   ├── labeled_3tokens.pkl
│   ├── labeled_4tokens.pkl
│   ├── segmented_1token.pkl     # labeled songs split into sections, 1-token
│   ├── segmented_3tokens.pkl
│   ├── segmented_4tokens.pkl
│   └── vocabs/
│       ├── vocabs.pkl           # all vocabulary and mapping dicts
│       ├── vocab_chords.csv
│       ├── vocab_roots.csv
│       ├── vocab_qualities.csv
│       ├── vocab_extensions.csv
│       ├── vocab_basses.csv
│       └── vocab_qualexes.csv
├── models/
│   ├── best_LSTM_model.pth
│   ├── best_LSTM_model_metrics.json
│   ├── LSTM/
│   │   └── json/
│   ├── GRU/
│   │   └── json/
│   └── RNN/
│       └── json/
└── results/
```

---

## Step 1 — Preprocess the dataset

Run `Chord_Embeddings.py` once to build all vocabularies and encoded sequence files.

```bash
python Chord_Embeddings.py
```

This reads `data/chordonomicon_v2.csv` and produces all `data/*.pkl` files and `data/vocabs/` outputs. The `data/` and `data/vocabs/` directories are created automatically if they don't exist.

---

## Step 2 — Train a model

```bash
python Model_Training.py \
    --representation triad \
    --model_type     lstm \
    --data_path      ./data/segmented_3tokens.pkl \
    --vocabs_path    ./data/vocabs/vocabs.pkl \
    --embed_dim      16 \
    --hidden_dim     256 \
    --num_layers     1 \
    --batch_size     4096 \
    --epochs         200 \
    --lr             5e-3 \
    --sample_size    200000 \
    --num            0
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--representation` | `chord` | Chord encoding: `chord`, `triad`, or `tetrad` |
| `--model_type` | `rnn` | Recurrent architecture: `rnn`, `gru`, or `lstm` |
| `--data_path` | `segmented_4tokens.pkl` | Path to encoded sequence dataset |
| `--vocabs_path` | `data/vocabs/vocabs.pkl` | Path to vocabs pickle |
| `--embed_dim` | `16` | Embedding dimension |
| `--hidden_dim` | `256` | RNN hidden dimension |
| `--num_layers` | `1` | Number of RNN layers |
| `--batch_size` | `4096` | Batch size |
| `--epochs` | `200` | Maximum training epochs |
| `--lr` | `5e-3` | Learning rate |
| `--sample_size` | `200000` | Total samples drawn (80% train, 10% val, 10% test) |
| `--num` | `0` | Run id, appended to saved filenames |
| `--seed` | random | Random seed (saved in checkpoint for reproducibility) |
| `--device` | auto | `cuda` or `cpu` |

### Data split

Songs are split by id — no song appears in more than one split. The split is stratified by section label. Early stopping triggers after 10 epochs of no improvement on validation loss. The learning rate is halved after 3 epochs of no improvement (`ReduceLROnPlateau`).

### Outputs

Each run saves to `models/<MODEL_TYPE>/` using a filename that encodes all hyperparameters. If the run achieves a new best validation loss, the checkpoint is also copied to `models/best_<MODEL_TYPE>_model.pth`.

---

## Step 3 — Evaluate a model

```bash
python Model_Evaluation.py \
    --representation  triad \
    --model_type      lstm \
    --model_path      ./models/best_LSTM_model.pth \
    --model_name      LSTM_triad_baseline \
    --dataset_path    ./data/segmented_3tokens.pkl \
    --vocabs_path     ./data/vocabs/vocabs.pkl \
    --batch_size      4096 \
    --top_n           10
```

To evaluate on a held-out second dataset instead of the test split:

```bash
python Model_Evaluation.py \
    ... \
    --second_dataset_path ./data/labeled_3tokens.pkl \
    --full_dataset
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--representation` | — | Required. Must match the trained model |
| `--model_type` | `rnn` | Must match the trained model |
| `--model_path` | — | Required. Path to `.pth` checkpoint |
| `--model_name` | — | Required. Human-readable name, used in output filename |
| `--dataset_path` | `segmented_4tokens.pkl` | Primary dataset (used to recreate splits) |
| `--second_dataset_path` | `labeled_4tokens.pkl` | Held-out dataset for `--full_dataset` |
| `--vocabs_path` | `data/vocabs/vocabs.pkl` | Path to vocabs pickle |
| `--save_dir` | `./results` | Directory for JSON results |
| `--batch_size` | `4096` | Batch size |
| `--top_n` | `10` | Number of top mismatches to print |
| `--full_dataset` | off | Evaluate on `second_dataset_path` instead of test split |

### Outputs

Results are saved to `results/<model_name>_<representation>_<sample_size>.json` containing metrics, per-length accuracy, and decoded mismatch tables. Evaluation is deterministic — the seed is restored from the checkpoint, giving identical results across runs.

---

## Representations

| Name | Encoding | Dataset files |
|---|---|---|
| `chord` | Single chord id (0-indexed) | `*_1token.pkl` |
| `triad` | `[root_id, qualex_id, bass_id]` | `*_3tokens.pkl` |
| `tetrad` | `[root_id, quality_id, extension_id, bass_id]` | `*_4tokens.pkl` |

In `triad` mode, quality and extensions are combined into a single `qualex` token. In `tetrad` mode they are split. All part ids are 1-indexed in the vocabulary; the dataset shifts them to 0-indexed for the model.

---

## Vocabularies

`data/vocabs/vocabs.pkl` contains 20 tables:

- **Encoding** — `chord_to_idx`, `root_to_idx`, `quality_to_idx`, `extensions_to_idx`, `bass_to_idx`, `qualex_to_idx`
- **Decoding** — `idx_to_chord`, `idx_to_root`, `idx_to_quality`, `idx_to_extensions`, `idx_to_bass`, `idx_to_qualex`
- **Text lookups** — `chord_to_parts_3/4`, `parts_to_chord_3/4`
- **Integer lookups** — `part_ids_to_chord_id_3/4`, `chord_id_to_part_ids_3/4`

Human-readable CSV versions are saved alongside in `data/vocabs/`.

---

## Using Run.py

`Run.py` is a convenience script — edit the `cmd` and `cmd2` lists to configure training and evaluation, then run:

```bash
python Run.py
```
