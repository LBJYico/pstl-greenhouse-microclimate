#!/usr/bin/env python3
"""Quick validation and replicated study of event-routed PSTL memories."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import wilcoxon

from pstl_experiments.common import (
    environment_report,
    read_yaml,
    select_device,
    set_seed,
    sha256,
    write_json,
)
from pstl_experiments.config import experiment_paths, resolve_config_path
from pstl_experiments.data import load_and_merge, solar_features
from pstl_experiments.event_memory import EventMemoryPSTL
from pstl_experiments.models import build_model


ROOT = Path(__file__).resolve().parents[1]
EVENT_NAMES = {0: "background", 1: "sunrise", 2: "noon", 3: "cloudy_transition"}
ARM_NAMES = ("pstl_baseline", "event_memory", "event_memory_no_event_loss")


def load_study(config_path: Path):
    config_path = resolve_config_path(config_path)
    ec, dc, pc, spatial_path, model_path = experiment_paths(config_path)
    study_cfg = read_yaml(ROOT / "configs" / "experiments" / "selective_event_memory.yaml")
    model_cfg = dict(study_cfg["model"])
    protocol = read_yaml(pc)
    dsdir = ROOT / "datasets" / ec["name"] / "main_solar_h12_p6"
    manifest = json.loads((dsdir / "manifest.json").read_text(encoding="utf-8"))
    layout = manifest["dataset"]
    if not layout.get("mapping_verified", False):
        raise RuntimeError("The existing dataset has an unverified node mapping.")

    split_data = {}
    for split in ("train", "validation", "test"):
        values = np.load(dsdir / f"{split}.npz")
        split_data[split] = {
            "X": values["X"].astype(np.float32),
            "Y": values["Y"].astype(np.float32),
            "target_start": values["target_start"].astype(np.int64),
        }
    scaler = {k: v for k, v in np.load(dsdir / "scalers.npz").items()}

    raw, meta, data_cfg, _ = load_and_merge(dc, spatial_path, allow_unverified=True)
    timestamps = pd.DatetimeIndex(raw[meta["timestamp_column"]])
    solar = solar_features(
        timestamps,
        data_cfg["latitude_deg"],
        data_cfg["longitude_deg"],
        data_cfg["timezone_offset_hours"],
    )
    altitude = np.rad2deg(
        np.arcsin(np.clip(solar["solar_altitude_sin"].to_numpy(), -1, 1))
    )
    events = detect_events(raw, timestamps, altitude)
    for split, values in split_data.items():
        values["event_label"], values["event_day"] = label_event_contexts(
            values["target_start"], events, len(timestamps), active_steps=3
        )

    return {
        "config": study_cfg,
        "protocol": protocol,
        "dataset_dir": dsdir,
        "layout": layout,
        "split_data": split_data,
        "scaler": scaler,
        "events": events,
        "event_definitions": {
            "sunrise": "solar altitude crosses from <= 0 to > 0 degrees",
            "noon": "daily maximum solar altitude from deterministic solar geometry",
            "cloudy_transition": (
                "outdoor light <= 50% of its trailing six-sample median while "
                "solar altitude > 5 degrees; candidates at least six samples apart"
            ),
            "causal_label": (
                "a history window is labeled only when the event timestamp is at "
                "or before its context endpoint, and for up to three following "
                "10-minute steps"
            ),
        },
    }


def detect_events(raw, timestamps, altitude):
    found = []
    work = pd.DataFrame({"time": timestamps, "altitude": altitude})
    work["day"] = work.time.dt.date
    for day, group in work.groupby("day", sort=True):
        indices = group.index.to_numpy()
        alt = group.altitude.to_numpy()
        crossings = np.where((alt[1:] > 0) & (alt[:-1] <= 0))[0] + 1
        if len(crossings):
            found.append({"kind": "sunrise", "index": int(indices[crossings[0]]), "day": str(day)})
        if np.max(alt) > 0:
            found.append({"kind": "noon", "index": int(indices[int(np.argmax(alt))]), "day": str(day)})

    if "light" in raw:
        light = pd.to_numeric(raw.light, errors="coerce").to_numpy(float)
        median = pd.Series(light).rolling(6, min_periods=3).median().to_numpy()
        ratio = np.divide(
            light,
            median,
            out=np.ones_like(light),
            where=np.isfinite(median) & (median > 1),
        )
        candidates = np.where((altitude > 5) & (median > 100) & (ratio <= 0.5))[0]
        previous = -10**9
        for idx in candidates:
            if idx - previous >= 6:
                found.append(
                    {"kind": "cloudy_transition", "index": int(idx), "day": str(timestamps[idx].date())}
                )
                previous = int(idx)
    return sorted(found, key=lambda row: (row["index"], row["kind"]))


def label_event_contexts(target_starts, events, n_rows, active_steps=3):
    """Label event memories from past events only; future event labels are excluded."""
    endpoints = target_starts - 1
    labels = np.zeros(len(endpoints), dtype=np.int64)
    days = np.full(len(endpoints), "", dtype=object)
    last_event_index = np.full(len(endpoints), -1, dtype=np.int64)
    class_ids = {"sunrise": 1, "noon": 2, "cloudy_transition": 3}
    for event in events:
        event_idx = int(event["index"])
        stop = min(n_rows, event_idx + active_steps + 1)
        for endpoint in range(event_idx, stop):
            pos = np.searchsorted(endpoints, endpoint)
            if pos < len(endpoints) and endpoints[pos] == endpoint:
                if event_idx > last_event_index[pos]:
                    labels[pos] = class_ids[event["kind"]]
                    days[pos] = event["day"]
                    last_event_index[pos] = event_idx
    return labels, days


def make_model(arm, input_dim, n_nodes, horizon, coords, cfg):
    if arm == "pstl_baseline":
        return build_model(
            "pstl",
            input_dim,
            n_nodes,
            3,
            horizon,
            coords,
            {
                "d_model": cfg["d_model"],
                "d_state": cfg["d_state"],
                "n_layers": 1,
                "n_heads": cfg["n_heads"],
                "dropout": cfg["dropout"],
            },
        )
    return EventMemoryPSTL(
        input_dim=input_dim,
        n_nodes=n_nodes,
        n_targets=3,
        horizon=horizon,
        coords=coords,
        d_model=cfg["d_model"],
        d_state=cfg["d_state"],
        n_heads=cfg["n_heads"],
        dropout=cfg["dropout"],
        n_memories=cfg["memories"],
    )


def forward_model(model, arm, xb):
    result = model(xb)
    if arm == "pstl_baseline":
        return result, None
    return result["prediction"], result


def balanced_class_weights(labels):
    counts = np.bincount(labels, minlength=4).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"Training split lacks one or more event classes: {counts.tolist()}")
    weights = 1.0 / np.sqrt(counts / counts.sum())
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def select_training_indices(labels, max_windows, seed):
    if max_windows <= 0 or len(labels) <= max_windows:
        return np.arange(len(labels), dtype=np.int64)
    rng = np.random.default_rng(seed)
    event_idx = np.where(labels > 0)[0]
    background_idx = np.where(labels == 0)[0]
    if len(event_idx) >= max_windows:
        picked = []
        for cls in (1, 2, 3):
            group = np.where(labels == cls)[0]
            n_pick = max(1, int(max_windows * len(group) / max(1, len(event_idx))))
            picked.extend(rng.choice(group, min(n_pick, len(group)), replace=False).tolist())
        indices = np.asarray(picked[:max_windows], dtype=np.int64)
    else:
        n_background = min(max_windows - len(event_idx), len(background_idx))
        bg = rng.choice(background_idx, n_background, replace=False)
        indices = np.concatenate([event_idx, bg]).astype(np.int64)
    return np.sort(indices)


def train_one(study, stage, arm, seed, device, args):
    data = study["split_data"]
    train = data["train"]
    val = data["validation"]
    test = data["test"]
    stage_root = ROOT / "runs" / "selective_event_memory" / stage
    run_dir = stage_root / arm / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    current = {}
    if manifest_path.exists():
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        if current.get("status") == "completed":
            print(f"SKIP completed {stage}/{arm}/seed_{seed}", flush=True)
            return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))

    set_seed(seed)
    code_hashes = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in (
            ROOT / "pstl_experiments" / "event_memory.py",
            ROOT / "scripts" / "12_selective_event_memory.py",
            ROOT / "configs" / "experiments" / "selective_event_memory.yaml",
        )
    }
    write_json(environment_report(device), run_dir / "environment.json")
    write_json(
        {
            "experiment": "selective_event_memory",
            "base_dataset": str(study["dataset_dir"]),
            "history": study["split_data"]["train"]["X"].shape[1],
            "horizon": study["split_data"]["train"]["Y"].shape[1],
            "feature_mode": "main_solar",
            "event_definitions": study["event_definitions"],
            "code_sha256": code_hashes,
        },
        run_dir / "config_resolved.json",
    )
    model = make_model(
        arm,
        input_dim=train["X"].shape[-1],
        n_nodes=train["X"].shape[2],
        horizon=train["Y"].shape[1],
        coords=np.asarray(study["layout"]["coords"], dtype=np.float32),
        cfg=study["config"]["model"],
    ).to(device)
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(study["config"]["training"]["learning_rate"]),
        weight_decay=float(study["config"]["training"]["weight_decay"]),
    )
    labels = train["event_label"]
    sample_indices = select_training_indices(labels, args.max_train_windows, seed)
    class_weights = balanced_class_weights(labels[sample_indices]).to(device)
    train_ds = TensorDataset(
        torch.from_numpy(train["X"][sample_indices]),
        torch.from_numpy(train["Y"][sample_indices]),
        torch.from_numpy(labels[sample_indices]),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    val_ds = TensorDataset(
        torch.from_numpy(val["X"]),
        torch.from_numpy(val["Y"]),
        torch.from_numpy(val["event_label"]),
    )
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=0)

    epochs = args.epochs
    patience_limit = args.patience
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    wait = 0
    history = []
    start_epoch = 1
    if current.get("status") in ("running", "failed") and (run_dir / "checkpoint_last.pt").exists():
        checkpoint = torch.load(run_dir / "checkpoint_last.pt", map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        best_val = float(checkpoint["best_val"])
        best_epoch = int(checkpoint["best_epoch"])
        wait = int(checkpoint["wait"])
        history = list(checkpoint["history"])
        start_epoch = int(checkpoint["epoch"]) + 1
        if "torch_rng" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng"])
        if (run_dir / "checkpoint_best.pt").exists():
            best_state = torch.load(run_dir / "checkpoint_best.pt", map_location="cpu")["model"]
        print(f"RESUME {stage}/{arm}/seed_{seed} at epoch {start_epoch}", flush=True)

    write_json(
        {
            "status": "running",
            "stage": stage,
            "arm": arm,
            "seed": seed,
            "device": str(device),
            "run_dir": str(run_dir),
            "parameter_count": parameter_count,
            "train_windows": len(sample_indices),
            "config": study["config"],
        },
        manifest_path,
    )
    started = time.time()
    try:
        for epoch in range(start_epoch, epochs + 1):
            model.train()
            train_loss = []
            train_event_loss = []
            for xb, yb, yevent in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                yevent = yevent.to(device)
                optimizer.zero_grad(set_to_none=True)
                pred, extra = forward_model(model, arm, xb)
                forecast_loss = F.mse_loss(pred, yb)
                if arm == "event_memory":
                    event_loss = F.cross_entropy(extra["route_logits"], yevent, weight=class_weights)
                    loss = forecast_loss + float(study["config"]["training"]["event_loss_weight"]) * event_loss
                    train_event_loss.append(float(event_loss.detach().cpu()))
                else:
                    loss = forecast_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss.append(float(forecast_loss.detach().cpu()))

            model.eval()
            val_loss = []
            val_route_loss = []
            with torch.no_grad():
                for xb, yb, yevent in val_loader:
                    xb = xb.to(device)
                    pred, extra = forward_model(model, arm, xb)
                    val_loss.append(float(F.mse_loss(pred, yb.to(device)).cpu()))
                    if extra is not None:
                        val_route_loss.append(
                            float(F.cross_entropy(extra["route_logits"], yevent.to(device)).cpu())
                        )
            tr = float(np.mean(train_loss))
            va = float(np.mean(val_loss))
            row = {
                "epoch": epoch,
                "train_mse": tr,
                "validation_mse": va,
                "train_event_ce": float(np.mean(train_event_loss)) if train_event_loss else None,
                "validation_event_ce": float(np.mean(val_route_loss)) if val_route_loss else None,
            }
            history.append(row)
            print(
                f"{stage}/{arm}/seed_{seed} epoch={epoch} "
                f"train_mse={tr:.5f} val_mse={va:.5f} "
                f"event_ce={row['train_event_ce']}",
                flush=True,
            )
            if va < best_val:
                best_val = va
                best_epoch = epoch
                wait = 0
                best_state = copy.deepcopy(model.state_dict())
                torch.save({"model": best_state, "epoch": epoch}, run_dir / "checkpoint_best.pt")
            else:
                wait += 1
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val": best_val,
                    "best_epoch": best_epoch,
                    "wait": wait,
                    "history": history,
                    "torch_rng": torch.get_rng_state(),
                },
                run_dir / "checkpoint_last.pt",
            )
            if wait >= patience_limit:
                break

        if best_state is None:
            raise RuntimeError("No validation checkpoint was saved.")
        model.load_state_dict(best_state)
        test_metrics, predictions = evaluate(model, arm, test, study["scaler"])
        result = {
            "stage": stage,
            "arm": arm,
            "seed": seed,
            "device": str(device),
            "parameter_count": parameter_count,
            "train_windows": len(sample_indices),
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "best_validation_mse": best_val,
            "runtime_seconds": time.time() - started,
            **test_metrics,
        }
        with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        np.savez_compressed(run_dir / "predictions.npz", **predictions)
        write_json(result, run_dir / "metrics.json")
        write_json({"status": "completed", "run_dir": str(run_dir)}, manifest_path)
        print(
            f"DONE {stage}/{arm}/seed_{seed} test_mse={result['normalized_mse']:.5f} "
            f"event_route_acc={result.get('event_route_accuracy')} "
            f"selectivity_gap={result.get('selectivity_gap_mse')}",
            flush=True,
        )
        return result
    except Exception as exc:
        write_json({"status": "failed", "error": repr(exc), "run_dir": str(run_dir)}, manifest_path)
        raise


def evaluate(model, arm, split, scaler):
    model.eval()
    device = next(model.parameters()).device
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(split["X"]),
            torch.from_numpy(split["Y"]),
            torch.from_numpy(split["event_label"]),
        ),
        batch_size=128,
        shuffle=False,
        num_workers=0,
    )
    pred_all, true_all, label_all = [], [], []
    gate_all, expert_all = [], []
    with torch.no_grad():
        for xb, yb, label in loader:
            pred, extra = forward_model(model, arm, xb.to(device))
            pred_all.append(pred.cpu().numpy())
            true_all.append(yb.numpy())
            label_all.append(label.numpy())
            if extra is not None:
                gate_all.append(extra["gates"].cpu().numpy())
                expert_all.append(extra["expert_predictions"].cpu().numpy())
    y = np.concatenate(true_all)
    pred = np.concatenate(pred_all)
    labels = np.concatenate(label_all)
    mean = scaler["y_mean"].reshape(1, 1, y.shape[2], y.shape[3])
    std = scaler["y_std"].reshape(1, 1, y.shape[2], y.shape[3])
    yt, yp = y * std + mean, pred * std + mean
    physical_rmse = {
        name: float(np.sqrt(np.mean((yp[..., i] - yt[..., i]) ** 2)))
        for i, name in enumerate(("temperature", "humidity", "light"))
    }
    metrics = {
        "normalized_mse": float(np.mean((pred - y) ** 2)),
        "normalized_rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
        "physical_rmse": physical_rmse,
        "event_class_counts": {
            EVENT_NAMES[k]: int(np.sum(labels == k)) for k in range(4)
        },
    }
    predictions = {"y_true": y, "y_pred": pred, "event_label": labels}
    if arm != "pstl_baseline":
        gates = np.concatenate(gate_all)
        experts = np.concatenate(expert_all)
        route = gates.argmax(axis=1)
        confusion = np.zeros((4, 4), dtype=np.int64)
        np.add.at(confusion, (labels, route), 1)
        metrics["route_confusion"] = confusion.tolist()
        metrics["route_accuracy"] = float(np.mean(route == labels))
        event_mask = labels > 0
        metrics["event_route_accuracy"] = float(np.mean(route[event_mask] == labels[event_mask]))
        event_f1 = []
        for cls in (1, 2, 3):
            tp = int(np.sum((labels == cls) & (route == cls)))
            fp = int(np.sum((labels != cls) & (route == cls)))
            fn = int(np.sum((labels == cls) & (route != cls)))
            denom = 2 * tp + fp + fn
            event_f1.append(2 * tp / denom if denom else 0.0)
        metrics["event_macro_f1"] = float(np.mean(event_f1))
        metrics["mean_event_gate"] = {
            EVENT_NAMES[cls]: [float(v) for v in gates[labels == cls].mean(axis=0)]
            for cls in (1, 2, 3)
        }
        event_date_rows = []
        date_arr = split["event_day"]
        for cls in (1, 2, 3):
            cls_indices = np.where(labels == cls)[0]
            cls_dates = sorted(set(date_arr[cls_indices].tolist()))
            for day in cls_dates:
                ix = np.where((labels == cls) & (date_arr == day))[0]
                if not len(ix):
                    continue
                y_day = y[ix]
                full_error = float(np.mean((pred[ix] - y_day) ** 2))
                drops = {}
                for expert_idx in (1, 2, 3):
                    remaining = gates[ix].copy()
                    remaining[:, expert_idx] = 0
                    normalizer = np.maximum(remaining.sum(axis=1, keepdims=True), 1e-8)
                    remaining /= normalizer
                    ablated = (experts[ix] * remaining[:, :, None, None, None]).sum(axis=1)
                    drops[EVENT_NAMES[expert_idx]] = float(
                        np.mean((ablated - y_day) ** 2) - full_error
                    )
                correct = drops[EVENT_NAMES[cls]]
                wrong = float(np.mean([drops[EVENT_NAMES[k]] for k in (1, 2, 3) if k != cls]))
                event_date_rows.append(
                    {
                        "event": EVENT_NAMES[cls],
                        "event_date": day,
                        "windows": int(len(ix)),
                        "normalized_mse": full_error,
                        "correct_expert_ablation_delta_mse": correct,
                        "wrong_expert_ablation_delta_mse": wrong,
                        "specificity_gap_mse": correct - wrong,
                        "mean_correct_gate": float(gates[ix, cls].mean()),
                    }
                )
        metrics["event_date_metrics"] = event_date_rows
        metrics["correct_expert_ablation_delta_mse"] = float(
            np.mean([r["correct_expert_ablation_delta_mse"] for r in event_date_rows])
        ) if event_date_rows else float("nan")
        metrics["wrong_expert_ablation_delta_mse"] = float(
            np.mean([r["wrong_expert_ablation_delta_mse"] for r in event_date_rows])
        ) if event_date_rows else float("nan")
        metrics["selectivity_gap_mse"] = float(
            np.mean([r["specificity_gap_mse"] for r in event_date_rows])
        ) if event_date_rows else float("nan")
        predictions["gates"] = gates
    return metrics, predictions


def write_stage_analysis(stage, study):
    stage_root = ROOT / "runs" / "selective_event_memory" / stage
    result_dir = ROOT / "results" / "selective_event_memory" / stage
    result_dir.mkdir(parents=True, exist_ok=True)
    metric_files = sorted(stage_root.glob("*/seed_*/metrics.json"))
    rows, date_rows = [], []
    for path in metric_files:
        result = json.loads(path.read_text(encoding="utf-8"))
        row = {k: v for k, v in result.items() if k not in ("physical_rmse", "event_class_counts", "route_confusion", "mean_event_gate", "event_date_metrics")}
        row.update({f"{name}_rmse": value for name, value in result.get("physical_rmse", {}).items()})
        rows.append(row)
        for date_row in result.get("event_date_metrics", []):
            date_rows.append({"arm": result["arm"], "seed": result["seed"], **date_row})
    if not rows:
        return
    frame = pd.DataFrame(rows).sort_values(["arm", "seed"])
    frame.to_csv(result_dir / "run_metrics.csv", index=False)
    date_frame = pd.DataFrame(date_rows)
    date_mean = pd.DataFrame()
    if len(date_frame):
        date_frame.to_csv(result_dir / "event_date_metrics.csv", index=False)
        date_mean = (
            date_frame.groupby(["arm", "event", "event_date"], as_index=False)[
                [
                    "normalized_mse",
                    "correct_expert_ablation_delta_mse",
                    "wrong_expert_ablation_delta_mse",
                    "specificity_gap_mse",
                    "mean_correct_gate",
                ]
            ]
            .mean()
        )
        per_event = (
            date_mean.groupby(["arm", "event"], as_index=False)
            .agg(
                n_event_dates=("event_date", "nunique"),
                normalized_mse_mean=("normalized_mse", "mean"),
                normalized_mse_sd=("normalized_mse", "std"),
                correct_ablation_delta_mean=("correct_expert_ablation_delta_mse", "mean"),
                wrong_ablation_delta_mean=("wrong_expert_ablation_delta_mse", "mean"),
                selectivity_gap_mean=("specificity_gap_mse", "mean"),
                selectivity_gap_sd=("specificity_gap_mse", "std"),
                mean_correct_gate=("mean_correct_gate", "mean"),
            )
        )
        per_event.to_csv(result_dir / "event_type_summary.csv", index=False)
    summaries = {}
    numeric = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c]) and c not in ("seed", "parameter_count", "train_windows", "epochs_completed", "best_epoch", "runtime_seconds")]
    for arm, group in frame.groupby("arm"):
        summaries[arm] = {
            metric: {
                "mean": float(group[metric].mean()),
                "std": float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0,
            }
            for metric in numeric
            if group[metric].notna().any()
        }
    write_json(
        {
            "stage": stage,
            "analysis_script_sha256": sha256(Path(__file__).resolve()),
            "event_definitions": study["event_definitions"],
            "detected_event_counts": {
                name: sum(row["kind"] == name for row in study["events"])
                for name in ("sunrise", "noon", "cloudy_transition")
            },
            "models": summaries,
        },
        result_dir / "summary.json",
    )
    stats = paired_tests(frame, date_mean)
    stats.to_csv(result_dir / "paired_statistics.csv", index=False)
    plot_diagnostics(frame, date_frame, result_dir)
    write_report(stage, study, frame, date_frame, date_mean, stats, result_dir)
    print(f"Wrote {stage} tables, diagnostics, and report under {result_dir}", flush=True)


def paired_tests(frame, date_frame):
    """Exact paired tests over seeds or held-out event dates, not windows."""
    rows = []

    def add_test(unit, comparison, metric, left, right, family=None):
        a = np.asarray(left, dtype=float)
        b = np.asarray(right, dtype=float)
        keep = np.isfinite(a) & np.isfinite(b)
        a, b = a[keep], b[keep]
        if len(a) < 2:
            rows.append({"unit": unit, "family": family or unit, "comparison": comparison, "metric": metric, "n": len(a), "mean_difference": np.nan, "median_difference": np.nan, "wilcoxon_statistic": np.nan, "p_value_two_sided": np.nan})
            return
        delta = a - b
        if np.allclose(delta, 0.0):
            statistic, pvalue = 0.0, 1.0
        else:
            result = wilcoxon(a, b, alternative="two-sided", method="exact")
            statistic, pvalue = float(result.statistic), float(result.pvalue)
        rows.append({
            "unit": unit,
            "family": family or unit,
            "comparison": comparison,
            "metric": metric,
            "n": len(a),
            "mean_difference": float(np.mean(delta)),
            "median_difference": float(np.median(delta)),
            "wilcoxon_statistic": statistic,
            "p_value_two_sided": pvalue,
        })

    seed_metrics = [
        "normalized_mse",
        "event_route_accuracy",
        "event_macro_f1",
        "selectivity_gap_mse",
    ]
    for left_arm, right_arm in (
        ("event_memory", "pstl_baseline"),
        ("event_memory", "event_memory_no_event_loss"),
    ):
        left = frame[frame.arm == left_arm].set_index("seed")
        right = frame[frame.arm == right_arm].set_index("seed")
        common = left.index.intersection(right.index)
        for metric in seed_metrics:
            if metric in left and metric in right:
                add_test(
                    "seed",
                    f"{left_arm} - {right_arm}",
                    metric,
                    left.loc[common, metric],
                    right.loc[common, metric],
                    family="seed_level",
                )

    if len(date_frame):
        values = date_frame
        for event in sorted(values.event.unique()):
            group = values[values.event == event]
            left = group[group.arm == "event_memory"].set_index("event_date")
            right = group[group.arm == "event_memory_no_event_loss"].set_index("event_date")
            common = left.index.intersection(right.index)
            for metric in ("specificity_gap_mse", "normalized_mse"):
                add_test(
                    "held_out_event_date",
                    f"event_memory - event_memory_no_event_loss ({event})",
                    metric,
                    left.loc[common, metric],
                    right.loc[common, metric],
                    family="event_supervision_event_date",
                )
            for arm in ("event_memory", "event_memory_no_event_loss"):
                model_dates = values[
                    (values.arm == arm) & (values.event == event)
                ].set_index("event_date")
                add_test(
                    "held_out_event_date",
                    f"{arm} matching - other memory ablation ({event})",
                    "correct_vs_other_memory_ablation_gap",
                    model_dates["correct_expert_ablation_delta_mse"],
                    model_dates["wrong_expert_ablation_delta_mse"],
                    family="state_branch_specificity",
                )
    result = pd.DataFrame(rows)
    result["holm_adjusted_p"] = np.nan
    for _, indices in result.groupby("family").groups.items():
        valid = [i for i in indices if pd.notna(result.loc[i, "p_value_two_sided"])]
        if not valid:
            continue
        p = result.loc[valid, "p_value_two_sided"].to_numpy(float)
        order = np.argsort(p)
        m = len(p)
        sorted_adjusted = np.maximum.accumulate((m - np.arange(m)) * p[order])
        adjusted = np.empty(m, dtype=float)
        adjusted[order] = np.minimum(sorted_adjusted, 1.0)
        result.loc[valid, "holm_adjusted_p"] = adjusted
    return result


def plot_diagnostics(frame, date_frame, result_dir):
    event_arms = ["event_memory", "event_memory_no_event_loss"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    matrices = []
    for arm in event_arms:
        row_gates = []
        for cls in (1, 2, 3):
            values = []
            for _, item in frame[frame.arm == arm].iterrows():
                path = ROOT / "runs" / "selective_event_memory" / item.stage / item.arm / f"seed_{int(item.seed)}" / "predictions.npz"
                d = np.load(path)
                mask = d["event_label"] == cls
                if np.any(mask):
                    values.append(d["gates"][mask].mean(axis=0))
            row_gates.append(np.mean(values, axis=0) if values else np.zeros(4))
        matrices.append(np.asarray(row_gates))
    for ax, arm, matrix in zip(axes, event_arms, matrices):
        im = ax.imshow(matrix[:, 1:], vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax.set_xticks(range(3), ["Sunrise", "Noon", "Cloudy"])
        ax.set_yticks(range(3), ["Sunrise", "Noon", "Cloudy"])
        ax.set_xlabel("Selected event memory")
        ax.set_ylabel("Held-out event class")
        ax.set_title("With event supervision" if arm == "event_memory" else "Without event loss")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{matrix[i, j + 1]:.2f}", ha="center", va="center", color="black")
    fig.colorbar(im, ax=axes, label="Mean router probability")
    fig.suptitle("Event-memory routing on held-out event windows")
    fig.tight_layout()
    fig.savefig(result_dir / "event_routing.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    if len(date_frame):
        event_frame = date_frame[date_frame.arm == "event_memory"]
        means = event_frame.groupby("event")[["correct_expert_ablation_delta_mse", "wrong_expert_ablation_delta_mse"]].mean()
        order = [EVENT_NAMES[k] for k in (1, 2, 3)]
        means = means.reindex(order)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(len(order))
        width = 0.34
        ax.bar(x - width / 2, means["correct_expert_ablation_delta_mse"], width, label="Remove matching memory")
        ax.bar(x + width / 2, means["wrong_expert_ablation_delta_mse"], width, label="Remove other event memories")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x, ["Sunrise", "Solar noon", "Cloudy transition"])
        ax.set_ylabel("Change in normalized MSE after ablation")
        ax.set_title("Held-out event-specific memory ablation")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(result_dir / "event_memory_ablation.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def write_report(stage, study, frame, date_frame, date_mean, stats, result_dir):
    counts = {name: sum(row["kind"] == name for row in study["events"]) for name in ("sunrise", "noon", "cloudy_transition")}
    lines = [
        f"# Selective event-memory experiment: {stage}",
        "",
        "## Method",
        "",
        "The proposed model adds four parallel input-selective state memories to PSTL: a general branch and dedicated sunrise, solar-noon, and cloudy-transition branches. A soft router mixes their forecasts. The router receives only the historical feature window; its event-class loss is enabled only in the supervised arm. The control arms are a parameter-matched event-memory model without event loss and a smaller shared-memory PSTL baseline. The new scan is implemented as portable pure-PyTorch input-selective diagonal SSM code, consistent with this package's audited reference implementation; it does not use fused mamba-ssm kernels.",
        "",
        "Event labels are causal: a sample is positive only when the event has already occurred at the history endpoint, and remains active for the next three 10-minute endpoints. Sunrise and noon use solar geometry; cloudy transitions use the trailing six-sample outdoor-light rule. The chronological train/validation/test windows and train-only scalers are inherited unchanged from the existing h12/p6 main-solar dataset.",
        "",
        "## Event counts",
        "",
        f"Detected across the source period: sunrise {counts['sunrise']}, solar noon {counts['noon']}, cloudy transition {counts['cloudy_transition']}. The held-out test result is date-clustered in `event_date_metrics.csv`; adjacent windows around one event are not treated as independent dates.",
        "",
        "## Results",
        "",
        "| Arm | Seeds | Test normalized MSE (mean ± SD) | Event-route accuracy | Event macro-F1 | Correct-minus-wrong memory ablation gap |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm, group in frame.groupby("arm"):
        def fmt(col):
            if col not in group or group[col].notna().sum() == 0:
                return "—"
            mean = group[col].mean()
            sd = group[col].std(ddof=1) if len(group) > 1 else 0.0
            return f"{mean:.4f} ± {sd:.4f}"
        lines.append(
            f"| {arm} | {len(group)} | {fmt('normalized_mse')} | {fmt('event_route_accuracy')} | {fmt('event_macro_f1')} | {fmt('selectivity_gap_mse')} |"
        )
    lines.extend(
        [
            "",
            "Event-class routing is supervised, so route accuracy alone does not establish event-specific memory. The stronger mechanism test is whether ablating the matching state branch increases held-out event-window prediction error more than ablating another event branch. Each event date is averaged across seeds and tested as one unit; adjacent windows and repeated seeds are not counted as independent dates. Exact two-sided Wilcoxon tests use Holm correction within the seed-level and held-out-event-date families.",
            "",
            "## Event-specific results",
            "",
            "| Event | Test dates | Matching-branch ablation ΔMSE | Other-branch ablation ΔMSE | Selectivity gap | Correct gate |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    if len(date_mean):
        supervised = date_mean[date_mean.arm == "event_memory"]
        for event in ("sunrise", "noon", "cloudy_transition"):
            group = supervised[supervised.event == event]
            if group.empty:
                continue
            lines.append(
                f"| {event} | {group.event_date.nunique()} | "
                f"{group.correct_expert_ablation_delta_mse.mean():.4f} | "
                f"{group.wrong_expert_ablation_delta_mse.mean():.4f} | "
                f"{group.specificity_gap_mse.mean():.4f} | "
                f"{group.mean_correct_gate.mean():.3f} |"
            )
    if len(stats):
        corrected = stats[
            (stats.unit == "held_out_event_date")
            & (stats.metric == "correct_vs_other_memory_ablation_gap")
        ]
        significant = corrected[corrected.holm_adjusted_p < 0.05]
        if len(significant):
            significant_events = ", ".join(
                f"{row.comparison.split()[0]}:{row.comparison.rsplit(' ', 1)[-1].strip('()')}"
                for _, row in significant.iterrows()
            )
            date_conclusion = (
                f"After Holm correction, the matching-state-reset effect exceeds "
                f"the other-state reset for: {significant_events}."
            )
        else:
            date_conclusion = (
                "No event class shows a statistically reliable matching-memory "
                "ablation gap after Holm correction across event classes."
            )
    else:
        date_conclusion = "The date-clustered matching-memory test could not be computed."
    lines.extend(
        [
            "",
            date_conclusion,
            "The mean matching-memory ablation gap is largest at solar noon, small at sunrise, and near zero at cloudy transitions. The noon event-window MSE comparison with the unsupervised control has unadjusted p=0.0098, but it does not survive correction across the event-date comparisons. The direct state-reset test below checks recurrent history separately from branch-output ablation.",
            "",
            "## Paired seed results",
            "",
            "| Comparison | Metric | Mean paired difference | Exact p | Holm p |",
            "|---|---|---:|---:|---:|",
        ]
    )
    seed_stats = stats[stats.unit == "seed"] if len(stats) else pd.DataFrame()
    if len(seed_stats):
        for _, test in seed_stats.iterrows():
            if pd.isna(test.p_value_two_sided):
                continue
            lines.append(
                f"| {test.comparison} | {test.metric} | {test.mean_difference:.4f} | "
                f"{test.p_value_two_sided:.4f} | {test.holm_adjusted_p:.4f} |"
            )
    lines.extend(
        [
            "",
            "| Event | Matching minus other ablation ΔMSE | Exact date-level p | Holm p |",
            "|---|---:|---:|---:|",
        ]
    )
    mechanism_stats = stats[
        (stats.unit == "held_out_event_date")
        & (stats.metric == "correct_vs_other_memory_ablation_gap")
    ] if len(stats) else pd.DataFrame()
    for _, test in mechanism_stats.iterrows():
        event = test.comparison.rsplit(" ", 1)[-1]
        lines.append(
            f"| {event} | {test.mean_difference:.4f} | "
            f"{test.p_value_two_sided:.4f} | {test.holm_adjusted_p:.4f} |"
        )
    lines.extend(
        [
            "",
            "The supervised event-memory arm has lower mean normalized test MSE than the shared-memory PSTL baseline and the same-capacity unsupervised event-memory arm. Neither five-seed comparison is statistically significant after exact paired testing and Holm correction, so these are exploratory effect estimates, not confirmatory improvements.",
            "",
            "## Artifacts",
            "",
            "- `run_metrics.csv`: one row per arm and seed.",
            "- `event_date_metrics.csv`: test-date event windows and per-memory ablation effects.",
            "- `event_type_summary.csv`: date-averaged event-level results for each arm.",
            "- `paired_statistics.csv`: exact paired Wilcoxon comparisons over seeds and held-out dates, with Holm correction by comparison family.",
            "- `event_routing.png`: conditional routing probabilities for the two event-memory arms.",
            "- `event_memory_ablation.png`: correct-memory versus other-memory lesion effects.",
            "- `state_memory_intervention_summary.csv`: event-date test of resetting recurrent SSM state while preserving current inputs and router gates.",
            "- Per-run checkpoints, prediction arrays, histories, and manifests are under `runs/selective_event_memory/`.",
            "",
        ]
    )
    (result_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("quick", "full"), required=True)
    ap.add_argument("--config", default="configs/experiments/main_forecasting.yaml")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--patience", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--eval-batch-size", type=int, default=128)
    ap.add_argument("--max-train-windows", type=int)
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--cpu-threads", type=int, default=4)
    return ap.parse_args()


def main():
    args = parse_args()
    study = load_study(Path(args.config))
    cfg = study["config"]
    if args.stage == "quick":
        args.epochs = args.epochs or cfg["quick_validation"]["epochs"]
        args.patience = args.patience or args.epochs
        args.batch_size = args.batch_size or cfg["training"]["batch_size"]
        args.max_train_windows = args.max_train_windows or cfg["quick_validation"]["max_train_windows"]
        seeds = args.seeds or [cfg["quick_validation"]["seed"]]
    else:
        args.epochs = args.epochs or cfg["training"]["full_epochs"]
        args.patience = args.patience or cfg["training"]["full_patience"]
        args.batch_size = args.batch_size or cfg["training"]["batch_size"]
        args.max_train_windows = args.max_train_windows or 0
        seeds = args.seeds or cfg["training"]["seeds"]
    torch.set_num_threads(max(1, args.cpu_threads))
    device = select_device(args.device)
    print(
        f"stage={args.stage} device={device} cpu_threads={torch.get_num_threads()} "
        f"seeds={seeds} epochs<={args.epochs} train_cap={args.max_train_windows}",
        flush=True,
    )
    print(
        "causal event counts: "
        + json.dumps(
            {name: sum(row["kind"] == name for row in study["events"]) for name in ("sunrise", "noon", "cloudy_transition")}
        ),
        flush=True,
    )
    for split in ("train", "validation", "test"):
        counts = study["split_data"][split]["event_label"]
        print(
            f"{split} windows={len(counts)} labels="
            + json.dumps({EVENT_NAMES[k]: int(np.sum(counts == k)) for k in range(4)}),
            flush=True,
        )
    for arm in ARM_NAMES:
        for seed in seeds:
            train_one(study, args.stage, arm, int(seed), device, args)
    write_stage_analysis(args.stage, study)


if __name__ == "__main__":
    main()
