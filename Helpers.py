# ------------------ Imports ------------------
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.model_selection import train_test_split


# ------------------ Dataset ------------------
class RandomCutDataset(Dataset):
    """
    Given a list of chord sequences, returns (input, target) pairs
    by randomly cutting each sequence at a point and predicting the next chord.
 
    In chord mode   : sequences are 1D arrays of chord ids.
    In triad/tetrad : sequences are 2D arrays of shape (len, num_parts).
    """
    def __init__(self, sequences, mode, parts_map=None, seq_len_min=5, max_len=50):
        self.mode = mode
        self.parts_map = parts_map  # (part_ids tuple) -> chord_id; required for triad/tetrad
        self.seq_len_min = seq_len_min
        self.max_len = max_len
        self.sequences = [s for s in sequences if len(s) >= seq_len_min]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        cutoff = np.random.randint(self.seq_len_min - 1, len(seq))
        start = max(0, cutoff - self.max_len)

        input_seq   = torch.tensor(seq[start:cutoff], dtype=torch.long)
        full_target = seq[cutoff]
 
        if self.mode == "chord":
            return input_seq, torch.tensor(full_target, dtype=torch.long)
        
        # Structured modes: full_target is a (num_parts,) array of 1-indexed part ids
        parts_tuple  = tuple(int(p) for p in full_target)
        chord_id     = self.parts_map[parts_tuple]
        target_parts = torch.tensor([p - 1 for p in parts_tuple], dtype=torch.long)  # shift to 0-index

        return input_seq, torch.tensor(chord_id, dtype=torch.long), target_parts


def collate_fn(batch, mode, pad_token):
    """Pad sequences to the same length and stack targets."""
    if mode == "chord":
        X, Y     = zip(*batch)
        lengths  = torch.tensor([len(x) for x in X], dtype=torch.long)
        X_padded = pad_sequence(X, batch_first=True, padding_value=pad_token)
        return X_padded, torch.stack(Y), lengths
    else:
        X, Yc, Yp = zip(*batch)
        lengths   = torch.tensor([len(x) for x in X], dtype=torch.long)
        X_padded  = pad_sequence(X, batch_first=True, padding_value=0)
        return X_padded, torch.stack(Yc), torch.stack(Yp), lengths
    

# ------------------ Model ------------------
class ChordModel(nn.Module):
    """
    RNN-based chord predictor with two operating modes:
 
    chord mode   : single embedding + single output head (chord id)
    triad/tetrad : one embedding per part + one head per part,
                   combined into a final chord prediction head
    """
    def __init__(self, mode, model_type, part_sizes, chord_vocab_size, embed_dim, hidden_dim, num_layers, pad_token):
        super().__init__()

        self.mode = mode
        self.part_sizes = part_sizes

        # Embeddings
        if mode == "chord":
            self.embedding = nn.Embedding(chord_vocab_size + 1, embed_dim, padding_idx=pad_token)
            input_dim = embed_dim
        else:
            self.embeddings = nn.ModuleList([nn.Embedding(size + 1, embed_dim, padding_idx=0) for size in part_sizes])
            input_dim = embed_dim * len(part_sizes)

        # Recurrent layer
        model_type_lower = model_type.lower()
        rnn_cls = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}
        if model_type_lower not in rnn_cls:
            raise ValueError(f"Unsupported model_type '{model_type}'. Choose from: {list(rnn_cls)}")
        self.encoder = rnn_cls[model_type_lower](input_dim, hidden_dim, num_layers, batch_first=True)

        # Output heads
        if mode == "chord":
            self.fc = nn.Linear(hidden_dim, chord_vocab_size)
        else:
            self.part_heads = nn.ModuleList([nn.Linear(hidden_dim, size) for size in part_sizes])
            self.chord_head = nn.Linear(sum(part_sizes), chord_vocab_size)

    def forward(self, x, lengths):
        # Embed input
        if self.mode == "chord":
            emb = self.embedding(x)
        else:
            emb = torch.cat([self.embeddings[i](x[:, :, i]) for i in range(len(self.part_sizes))], dim=-1)

        # Run through RNN and extract last non-padded output
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.encoder(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        last_out = out[torch.arange(out.size(0), device=out.device), lengths - 1]

        if self.mode == "chord":
            return self.fc(last_out)
        
        # Part logits + combined chord logit
        part_logits  = [head(last_out) for head in self.part_heads]
        chord_logits = self.chord_head(torch.cat(part_logits, dim=-1))

        return (*part_logits, chord_logits)


# ------------------ Splits ------------------
def stratified_group_split(df, group_col='id', label_col='label', val_size=0.1, test_size=0.1, seed=None):
    """
    Split song ids into train / val / test stratified by label,
    ensuring no song appears in more than one split.
    """
    assert 0.0 <= val_size < 1.0 and 0.0 <= test_size < 1.0 and val_size + test_size < 1.0
    
    grouped = df.groupby(group_col)[label_col].agg(lambda s: s.mode().iloc[0])
    group_ids = grouped.index.to_numpy()
    group_labels = grouped.to_numpy()

    train_ids, temp_ids, _, _ = train_test_split(
        group_ids, group_labels, test_size=val_size + test_size,
        stratify=group_labels, random_state=seed
    )
    temp_labels = np.array([grouped.loc[i] for i in temp_ids])
    val_ids, test_ids = train_test_split(
        temp_ids, test_size=test_size / (val_size + test_size),
        stratify=temp_labels, random_state=seed
    )
    return np.array(train_ids), np.array(val_ids), np.array(test_ids)


def stratified_samples(df, label_col="label", seq_len_min=5, sample_size=None, random_state=None):
    """
    Filter out short sequences, then draw a stratified subsample.
    Returns the full filtered df if sample_size is None or exceeds available rows.
    """
    df_filtered = df[df['chords'].apply(len) >= seq_len_min].reset_index(drop=True)

    if sample_size is None or sample_size >= len(df_filtered):
        return df_filtered

    subsample, _ = train_test_split(
        df_filtered,
        train_size=sample_size,
        stratify=df_filtered[label_col],
        random_state=random_state,
    )

    return subsample.reset_index(drop=True)