#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for training the temporal GRU model.
    """
    p = argparse.ArgumentParser(
        description="Train a temporal GRU model for next-week raion risk prediction."
    )
    p.add_argument("--model_table_csv", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--seq-len", type=int, default=8, dest="seq_len")
    p.add_argument("--batch-size", type=int, default=128, dest="batch_size")
    p.add_argument("--hidden-dim", type=int, default=96, dest="hidden_dim")
    p.add_argument("--num-layers", type=int, default=2, dest="num_layers")
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4, dest="weight_decay")
    p.add_argument("--target-col", default="target_next_week_high_risk", dest="target_col")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(arg: str) -> torch.device:
    """
    Choose the compute device. If set to auto, prefer CUDA,
    then MPS, and otherwise fall back to CPU.
    """
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Select the dynamic and static feature columns that are present
    in the model table.
    """
    # Dynamic features go into the sequence input of the GRU
    dynamic_candidates = [
        "acled_event_count",
        "fatalities_sum",
        "events_with_fatalities",
        "violence_against_civilians_count",
        "explosions_remote_count",
        "battles_count",
        "strategic_developments_count",
        "protests_riots_count",
        "civilian_targeting_count",
        "air_drone_strike_count",
        "precise_geo_event_count",
        "any_event",
        "high_intensity_week",
        "ntl_week_mean",
        "ntl_week_median",
        "ntl_week_std_mean",
        "ntl_high_quality_share",
        "ntl_latest_hq_share",
        "ntl_obs_days",
        "ntl_valid_pixels_sum",
        "ntl_change_vs_prev_week",
        "ntl_change_vs_rolling4",
        "ntl_pct_change_vs_prev_week",
    ]
    dynamic_cols = [c for c in dynamic_candidates if c in df.columns]

    # Static or slowly changing features are fed through a separate MLP
    static_candidates = [
        "road_total_length_km",
        "road_major_length_km",
        "road_paved_length_km",
        "road_bridge_length_km",
        "rail_total_length_km",
        "road_density_km_per_100sqkm",
        "area_sqkm",
        "major_road_density_km_per_100sqkm",
        "rail_density_km_per_100sqkm",
        "major_road_share",
        "paved_road_share",
        "rail_to_road_ratio",
    ]
    static_cols = [c for c in static_candidates if c in df.columns]
    return dynamic_cols, static_cols


class SequenceDataset(Dataset):
    """
    Simple PyTorch dataset wrapping sequence features,
    static features, and targets.
    """
    def __init__(
        self,
        sequences: np.ndarray,
        statics: np.ndarray,
        targets: np.ndarray,
        meta: pd.DataFrame,
    ) -> None:
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.statics = torch.tensor(statics, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.meta = meta.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int):
        return self.sequences[idx], self.statics[idx], self.targets[idx]


class GRURiskModel(nn.Module):
    """
    Bidirectional GRU over temporal inputs, optionally fused with
    a small MLP over static features.
    """
    def __init__(
        self,
        seq_dim: int,
        static_dim: int,
        hidden_dim: int = 96,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=seq_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        self.static_mlp = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        ) if static_dim > 0 else None

        fusion_in = hidden_dim * 2 + (hidden_dim if static_dim > 0 else 0)
        self.head = nn.Sequential(
            nn.Linear(fusion_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, seq_x: torch.Tensor, static_x: torch.Tensor) -> torch.Tensor:
        # Take the final GRU time-step output as the sequence representation
        out, _ = self.gru(seq_x)
        seq_repr = out[:, -1, :]

        if self.static_mlp is not None:
            static_repr = self.static_mlp(static_x)
            fused = torch.cat([seq_repr, static_repr], dim=1)
        else:
            fused = seq_repr

        logits = self.head(fused).squeeze(1)
        return logits


def build_sequence_rows(
    df: pd.DataFrame,
    dynamic_cols: List[str],
    static_cols: List[str],
    seq_len: int,
    target_col: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Convert the raion-week table into fixed-length sequences.
    Each row becomes one training example using the previous seq_len weeks.
    """
    df = df.sort_values(["raion_id", "week_start"]).copy()

    seq_rows = []
    static_rows = []
    targets = []
    metas = []

    for raion_id, grp in df.groupby("raion_id", sort=False):
        grp = grp.sort_values("week_start").reset_index(drop=True)

        dyn = grp[dynamic_cols].fillna(0).to_numpy(dtype=float)
        stat = (
            grp[static_cols].fillna(0).to_numpy(dtype=float)
            if static_cols else np.zeros((len(grp), 0), dtype=float)
        )
        tgt = grp[target_col].to_numpy(dtype=float)

        for i in range(seq_len - 1, len(grp)):
            y = tgt[i]
            if np.isnan(y):
                continue

            seq_rows.append(dyn[i - seq_len + 1: i + 1])
            static_rows.append(stat[i])
            targets.append(y)

            metas.append({
                "raion_id": grp.loc[i, "raion_id"],
                "raion_name": grp.loc[i, "raion_name"],
                "oblast_name": grp.loc[i, "oblast_name"],
                "week_start": grp.loc[i, "week_start"],
                "split": grp.loc[i, "split"],
            })

    return (
        np.asarray(seq_rows, dtype=float),
        np.asarray(static_rows, dtype=float),
        np.asarray(targets, dtype=float),
        pd.DataFrame(metas),
    )


def fit_scalers(train_seq: np.ndarray, train_static: np.ndarray) -> Tuple[StandardScaler, StandardScaler | None]:
    """
    Fit feature scalers using only the training subset.
    """
    n, t, d = train_seq.shape

    # Flatten the sequence over time so one scaler is fit per feature dimension
    seq_scaler = StandardScaler()
    seq_scaler.fit(train_seq.reshape(n * t, d))

    static_scaler = None
    if train_static.shape[1] > 0:
        static_scaler = StandardScaler()
        static_scaler.fit(train_static)

    return seq_scaler, static_scaler


def transform_data(
    seq: np.ndarray,
    static: np.ndarray,
    seq_scaler: StandardScaler,
    static_scaler: StandardScaler | None
):
    """
    Apply the fitted scalers to sequence and static features.
    """
    n, t, d = seq.shape
    seq_t = seq_scaler.transform(seq.reshape(n * t, d)).reshape(n, t, d)

    if static_scaler is not None and static.shape[1] > 0:
        static_t = static_scaler.transform(static)
    else:
        static_t = static

    return seq_t, static_t


def build_loaders(
    seq: np.ndarray,
    static: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    batch_size: int
):
    """
    Split the prepared arrays into train/valid/test datasets and loaders.
    """
    idx_train = meta["split"] == "train"
    idx_valid = meta["split"] == "valid"
    idx_test = meta["split"] == "test"

    ds_train = SequenceDataset(seq[idx_train], static[idx_train], y[idx_train], meta[idx_train])
    ds_valid = SequenceDataset(seq[idx_valid], static[idx_valid], y[idx_valid], meta[idx_valid])
    ds_test = SequenceDataset(seq[idx_test], static[idx_test], y[idx_test], meta[idx_test])

    return (
        DataLoader(ds_train, batch_size=batch_size, shuffle=True),
        DataLoader(ds_valid, batch_size=batch_size, shuffle=False),
        DataLoader(ds_test, batch_size=batch_size, shuffle=False),
        ds_train,
        ds_valid,
        ds_test,
    )


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    """
    Run the model on one split and return true labels and predicted probabilities.
    """
    model.eval()
    ys, ps = [], []

    with torch.no_grad():
        for seq_x, static_x, y in loader:
            seq_x = seq_x.to(device)
            static_x = static_x.to(device)

            logits = model(seq_x, static_x)
            prob = torch.sigmoid(logits).cpu().numpy()

            ys.append(y.numpy())
            ps.append(prob)

    y_true = np.concatenate(ys) if ys else np.array([])
    y_prob = np.concatenate(ps) if ps else np.array([])
    return y_true, y_prob


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    Compute standard binary classification metrics.
    """
    if len(y_true) == 0:
        return {}

    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "avg_precision": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "positive_rate": float(np.mean(y_true)),
        "predicted_positive_rate": float(np.mean(y_pred)),
    }
    return metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.model_table_csv, parse_dates=["week_start"])
    if args.target_col not in df.columns:
        raise ValueError(f"Target column not found: {args.target_col}")

    dynamic_cols, static_cols = pick_feature_columns(df)
    if not dynamic_cols:
        raise ValueError("No dynamic feature columns found in model table.")

    seq, static, y, meta = build_sequence_rows(
        df, dynamic_cols, static_cols, args.seq_len, args.target_col
    )

    if len(meta) == 0:
        raise ValueError("No sequence rows created. Check sequence length and target availability.")

    # Fit scalers only on the training subset to avoid leakage
    idx_train = meta["split"] == "train"
    if idx_train.sum() == 0:
        raise ValueError("No training rows found. Check split column in model table.")

    seq_scaler, static_scaler = fit_scalers(seq[idx_train], static[idx_train])
    seq_t, static_t = transform_data(seq, static, seq_scaler, static_scaler)

    train_loader, valid_loader, test_loader, ds_train, ds_valid, ds_test = build_loaders(
        seq_t, static_t, y, meta, args.batch_size
    )

    model = GRURiskModel(
        seq_dim=seq_t.shape[2],
        static_dim=static_t.shape[1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    # Weight positives higher if the training labels are imbalanced
    pos_rate = float(ds_train.targets.mean().item()) if len(ds_train) else 0.5
    pos_weight = torch.tensor([(1 - pos_rate) / max(pos_rate, 1e-6)], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    best_state = None
    best_score = -np.inf
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []

        for seq_x, static_x, target in train_loader:
            seq_x = seq_x.to(device)
            static_x = static_x.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            logits = model(seq_x, static_x)
            loss = criterion(logits, target)
            loss.backward()

            # Gradient clipping helps keep training stable
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            losses.append(loss.item())

        y_val, p_val = evaluate(model, valid_loader, device)
        val_metrics = compute_metrics(y_val, p_val)
        val_score = val_metrics.get("f1", float("-inf"))

        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else np.nan,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        print(
            f"Epoch {epoch:02d} | "
            f"loss={np.mean(losses):.4f} | "
            f"val_f1={val_metrics.get('f1', float('nan')):.4f} | "
            f"val_ap={val_metrics.get('avg_precision', float('nan')):.4f}"
        )

        # Keep the checkpoint with the best validation F1
        if val_score > best_score:
            best_score = val_score
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation on all splits using the best saved model
    y_train, p_train = evaluate(model, train_loader, device)
    y_val, p_val = evaluate(model, valid_loader, device)
    y_test, p_test = evaluate(model, test_loader, device)

    metrics = {
        "train": compute_metrics(y_train, p_train),
        "valid": compute_metrics(y_val, p_val),
        "test": compute_metrics(y_test, p_test),
        "dynamic_cols": dynamic_cols,
        "static_cols": static_cols,
        "seq_len": args.seq_len,
        "n_train": int(len(ds_train)),
        "n_valid": int(len(ds_valid)),
        "n_test": int(len(ds_test)),
    }

    # Save model weights plus preprocessing info needed for reuse later
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "dynamic_cols": dynamic_cols,
            "static_cols": static_cols,
            "seq_len": args.seq_len,
            "seq_scaler_mean": seq_scaler.mean_,
            "seq_scaler_scale": seq_scaler.scale_,
            "static_scaler_mean": getattr(static_scaler, "mean_", None),
            "static_scaler_scale": getattr(static_scaler, "scale_", None),
            "args": vars(args),
            "metrics": metrics,
        },
        outdir / "gru_risk_model.pt",
    )

    pd.DataFrame(history).to_csv(outdir / "training_history.csv", index=False)

    # Save row-level predictions for later analysis and visualization
    preds = []
    for split_name, ds, y_true, y_prob in [
        ("train", ds_train, y_train, p_train),
        ("valid", ds_valid, y_val, p_val),
        ("test", ds_test, y_test, p_test),
    ]:
        meta_df = ds.meta.copy()
        meta_df["y_true"] = y_true
        meta_df["y_prob"] = y_prob
        meta_df["y_pred"] = (y_prob >= 0.5).astype(int)
        meta_df["split"] = split_name
        preds.append(meta_df)

    pd.concat(preds, ignore_index=True).to_csv(outdir / "predictions.csv", index=False)

    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved outputs to:", outdir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()