# ------------------ Imports ------------------
import pandas as pd
import time
from tqdm import tqdm
from pathlib import Path

tqdm.pandas()


start_time = time.time()


# ------------------ Config ------------------
DATA_DIR = Path("./data")
VOCABS_DIR = DATA_DIR / "vocabs"
DATASET  = DATA_DIR / "chordonomicon_v2.csv"

# Encoded sequence outputs  (<pipeline>_<encoding>.pkl)
#   Pipelines : all_songs | sections | labeled
#   Encodings : 1token (chord id) | 3tokens (root+qualex+bass) | 4tokens (root+quality+ext+bass)
OUTPUT_PATHS = {
    ("all_songs", "1token"):   DATA_DIR / "filtered_1token.pkl",
    ("all_songs", "3tokens"):  DATA_DIR / "filtered_3tokens.pkl",
    ("all_songs", "4tokens"):  DATA_DIR / "filtered_4tokens.pkl",

    ("labeled",   "1token"):   DATA_DIR / "labeled_1token.pkl",
    ("labeled",   "3tokens"):  DATA_DIR / "labeled_3tokens.pkl",
    ("labeled",   "4tokens"):  DATA_DIR / "labeled_4tokens.pkl",
 
    ("sections",  "1token"):   DATA_DIR / "segmented_1token.pkl",
    ("sections",  "3tokens"):  DATA_DIR / "segmented_3tokens.pkl",
    ("sections",  "4tokens"):  DATA_DIR / "segmented_4tokens.pkl",
 
}

VOCABS_PATH = VOCABS_DIR / "vocabs.pkl"

CSV_PATHS = {
    "chords":     VOCABS_DIR / "vocab_chords.csv",
    "roots":      VOCABS_DIR / "vocab_roots.csv",
    "qualities":  VOCABS_DIR / "vocab_qualities.csv",
    "extensions": VOCABS_DIR / "vocab_extensions.csv",
    "basses":     VOCABS_DIR / "vocab_basses.csv",
    "qualexes":   VOCABS_DIR / "vocab_qualexes.csv",
}

# ------------------ Globals ------------------
QUALITY_PREFIXES = ["dim", "aug", "min", "no3d"]
QUALITY_SUFFIXES = ["sus2", "sus4"]

# ------------------ Helpers ------------------
def scrap(chords: str) -> bool:
    """Keep only songs without weirdly-spaced slash chords."""
    return "/ " not in chords


def collect_chords(chords: str) -> set:
    """Return unique chord symbols from a song (skip section labels)."""
    return {chord for chord in chords.split() if not is_label(chord)}


def is_label(chord: str) -> bool:
    return "<" in chord and ">" in chord


def format_label(token: str) -> str:
    """'<Verse_1>' -> 'Verse',  '<Chorus>' -> 'Chorus'. Non-labels pass through."""
    if not isinstance(token, str) or "<" not in token:
        return token
    return token.strip("<>").split("_")[0]


def split_quality_extensions(qual_extents: str) -> tuple[str, str]:
    if qual_extents == "maj":
        return "maj", "N"
 
    for q in QUALITY_PREFIXES:
        if qual_extents.startswith(q):
            ext = qual_extents[len(q):]
            return q, ext or "N"
 
    for q in QUALITY_SUFFIXES:
        if qual_extents.endswith(q):
            ext = qual_extents[: -len(q)]
            return q, ext or "N"
 
    return "maj", qual_extents  # unknown quality -> treat as extension

def parse_chord(chord: str):
    """Split chord into (root, quality+extensions, bass) or (root, quality, extensions, bass)."""
    parts = chord.split("/")
    main = parts[0]
    bass = parts[1] if len(parts) > 1 else "N"

    if len(main) >= 3 and (main[1] in ["b", "s"]) and (main[2] != "u"):
        root = main[:2]
        qualex = main[2:]
    elif len(main) >= 3:
        root = main[0]
        qualex = main[1:]
    elif len(main) == 2 and (main[1] not in ["b", "s"]):
        root = main[0]
        qualex = main[1]
    else:
        root = main
        qualex = "maj"

    quality, extensions = split_quality_extensions(qualex)
    
    return root, quality, extensions, bass, qualex


# ------------------ Sequence helpers ------------------
def has_label(chords: list[str]) -> bool:
    return any(is_label(c) for c in chords)


def remove_labels(chords: list[str]) -> list[str]:
    return [c for c in chords if not is_label(c)]


def extract_parts(chords: list[str]) -> list[list[str]]:
    """Split a chord list on section labels -> list of [label, chord, chord, ...]."""
    sections, last = [], 0
    for i, chord in enumerate(chords):
        if is_label(chord) and i > 0:
            sections.append(chords[last:i])
            last = i
    sections.append(chords[last:])
    return sections


# ------------------ Encoders ------------------
def encode_1token(chords: list[str]) -> list[int]:
    return [chord_to_idx[c] for c in chords]
 
 
def encode_3tokens(chords: list[str]) -> list[list[int]]:
    return [
        [root_to_idx[r], qualex_to_idx[qe], bass_to_idx[b]]
        for c in chords
        for r, qe, b in (chord_to_parts_3[c],)
    ]


def encode_4tokens(chords: list[str]) -> list[list[int]]:
    return [
        [root_to_idx[r], quality_to_idx[q], extensions_to_idx[e], bass_to_idx[b]]
        for c in chords
        for r, q, e, b in (chord_to_parts_4[c],)
    ]

 
def save_encoded(df: pd.DataFrame, encode_fn, path: Path) -> None:
    out = df.copy()
    out["chords"] = out["chords"].progress_apply(encode_fn)
    cols = ["id", "chords"] + (["label"] if "label" in out.columns else [])
    out[cols].to_pickle(path)
 
 
def save_vocab_csv(token_to_idx: dict, token_col: str, path: Path) -> None:
    """Write a two-column CSV: token | id, sorted by id."""
    pd.DataFrame(
        sorted(token_to_idx.items(), key=lambda x: x[1]),
        columns=[token_col, "id"],
    ).to_csv(path, index=False)


# ------------------ Main ------------------
 
# Ensure output directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
VOCABS_DIR.mkdir(parents=True, exist_ok=True)

print("Loading dataset...")
df = pd.read_csv(DATASET, usecols=["id", "chords", "decade", "main_genre"], low_memory=False)
df.rename(columns={"main_genre": "genre"}, inplace=True)
df = df[
    (df["decade"].notna() | df["genre"].notna())
    & ((df["decade"] != "") | (df["genre"] != ""))
    & (df["decade"] != df["genre"])
]
 
print("Removing songs with malformed chord progressions...")
df = df[df["chords"].progress_apply(scrap)]
 
print("Collecting unique chords...")
chords_set = set().union(*df["chords"].progress_apply(collect_chords))


# ------ Build vocabularies ------
print("Parsing chords into components...")
roots, qualities, extensions, basses, qualex = set(), set(), set(), set(), set()
chord_to_parts_3 = {}
parts_to_chord_3 = {}
chord_to_parts_4 = {}
parts_to_chord_4 = {}

for chord in sorted(chords_set):
    r, q, e, b, qe = parse_chord(chord)
    chord_to_parts_3[chord] = (r, qe, b)
    parts_to_chord_3[(r, qe, b)] = chord  # store reverse mapping
    chord_to_parts_4[chord] = (r, q, e, b)
    parts_to_chord_4[(r, q, e, b)] = chord
    roots.add(r); qualities.add(q); extensions.add(e); basses.add(b); qualex.add(qe)

# Token -> index  (1-indexed for component vocabs, 0-indexed for chords)
chord_to_idx      = {c: i   for i, c in enumerate(sorted(chords_set))}
root_to_idx       = {r: i+1 for i, r in enumerate(sorted(roots))}
quality_to_idx    = {q: i+1 for i, q in enumerate(sorted(qualities))}
extensions_to_idx = {e: i+1 for i, e in enumerate(sorted(extensions))}
bass_to_idx       = {b: i+1 for i, b in enumerate(sorted(basses))}
qualex_to_idx     = {qe: i+1 for i, qe in enumerate(sorted(qualex))}

# Integer-key cross-lookups  (chord_id <-> part_ids)
part_ids_to_chord_id_3 = {
    (root_to_idx[r], qualex_to_idx[qe], bass_to_idx[b]): chord_to_idx[chord]
    for (r, qe, b), chord in parts_to_chord_3.items()
}
part_ids_to_chord_id_4 = {
    (root_to_idx[r], quality_to_idx[q], extensions_to_idx[e], bass_to_idx[b]): chord_to_idx[chord]
    for (r, q, e, b), chord in parts_to_chord_4.items()
}
chord_id_to_part_ids_3 = {v: k for k, v in part_ids_to_chord_id_3.items()}
chord_id_to_part_ids_4 = {v: k for k, v in part_ids_to_chord_id_4.items()}


# ------ Save vocab CSVs ------
print("Saving vocab CSVs...")
save_vocab_csv(chord_to_idx,      "chord",     CSV_PATHS["chords"])
save_vocab_csv(root_to_idx,       "root",      CSV_PATHS["roots"])
save_vocab_csv(quality_to_idx,    "quality",   CSV_PATHS["qualities"])
save_vocab_csv(extensions_to_idx, "extension", CSV_PATHS["extensions"])
save_vocab_csv(bass_to_idx,       "bass",      CSV_PATHS["basses"])
save_vocab_csv(qualex_to_idx,     "qualex",    CSV_PATHS["qualexes"])
 
for name, path in CSV_PATHS.items():
    print(f"  -> {path}")


# ------ Save vocab pickle ------
print("Saving vocabs and mappings...")
vocabs = {
    # Token -> index  (for encoding)
    "chord_to_idx":      chord_to_idx,
    "root_to_idx":       root_to_idx,
    "quality_to_idx":    quality_to_idx,
    "extensions_to_idx": extensions_to_idx,
    "bass_to_idx":       bass_to_idx,
    "qualex_to_idx":     qualex_to_idx,
 
    # Index -> token  (for decoding)
    "idx_to_chord":      {v: k for k, v in chord_to_idx.items()},
    "idx_to_root":       {v: k for k, v in root_to_idx.items()},
    "idx_to_quality":    {v: k for k, v in quality_to_idx.items()},
    "idx_to_extensions": {v: k for k, v in extensions_to_idx.items()},
    "idx_to_bass":       {v: k for k, v in bass_to_idx.items()},
    "idx_to_qualex":     {v: k for k, v in qualex_to_idx.items()},
 
    # Text-based  chord <-> parts
    "chord_to_parts_3":  chord_to_parts_3,   # chord -> (root, qualex, bass)
    "chord_to_parts_4":  chord_to_parts_4,   # chord -> (root, quality, ext, bass)
    "parts_to_chord_3":  parts_to_chord_3,   # (root, qualex, bass)         -> chord
    "parts_to_chord_4":  parts_to_chord_4,   # (root, quality, ext, bass)   -> chord
 
    # Integer-based  chord_id <-> part_ids
    "part_ids_to_chord_id_3": part_ids_to_chord_id_3,  # (r_id, qe_id, b_id)        -> chord_id
    "part_ids_to_chord_id_4": part_ids_to_chord_id_4,  # (r_id, q_id, e_id, b_id)   -> chord_id
    "chord_id_to_part_ids_3": chord_id_to_part_ids_3,  # chord_id -> (r_id, qe_id, b_id)
    "chord_id_to_part_ids_4": chord_id_to_part_ids_4,  # chord_id -> (r_id, q_id, e_id, b_id)
}
 
pd.to_pickle(vocabs, VOCABS_PATH)
 
print(f"  -> {VOCABS_PATH}  ({len(vocabs)} tables)")


# ------ Build pipeline DataFrames ------
base = df[['id', 'chords']].copy()
base['chords'] = base['chords'].progress_apply(str.split)

label_mask = base['chords'].progress_apply(has_label)

# all_songs: every song, labels stripped
df_all = base.copy()
df_all['chords'] = df_all['chords'].progress_apply(remove_labels)

# sections: labeled songs only, exploded into one row per section
df_sections = base[label_mask].copy()
df_sections['chords'] = df_sections['chords'].progress_apply(extract_parts)
df_sections = df_sections.explode('chords', ignore_index=True)
df_sections["label"]  = df_sections["chords"].progress_apply(lambda x: format_label(x[0]))
df_sections['chords'] = df_sections['chords'].progress_apply(lambda x: x[1:])

# labeled: labeled songs, labels stripped, full sequence kept
df_labeled = base[label_mask].copy()
df_labeled['chords'] = df_labeled['chords'].progress_apply(remove_labels)

PIPELINES = {
    "all_songs": df_all,
    "sections": df_sections,
    "labeled": df_labeled
}

ENCODERS = {
    "1token": encode_1token,
    "3tokens": encode_3tokens,
    "4tokens": encode_4tokens
}


# ------ Encode & save ------
for pipeline_name, pipeline_df in PIPELINES.items():
    for enc_name, enc_fn in ENCODERS.items():
        print(f"Encoding & saving  {pipeline_name} / {enc_name}")
        save_encoded(pipeline_df, enc_fn, OUTPUT_PATHS[(pipeline_name, enc_name)])

print(f"Done in {time.time() - start_time:.2f}s")
