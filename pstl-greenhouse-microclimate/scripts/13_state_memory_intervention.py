#!/usr/bin/env python3
"""Intervene on trained Mamba recurrence while preserving present inputs/router."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from torch.utils.data import DataLoader, TensorDataset

from pstl_experiments.common import select_device, write_json
from pstl_experiments.event_memory import EventMemoryPSTL

ROOT = Path(__file__).resolve().parents[1]
EVENTS = {1: "sunrise", 2: "noon", 3: "cloudy_transition"}
ARMS = ("event_memory", "event_memory_no_event_loss")


def load_training_driver():
    path = ROOT / "scripts" / "12_selective_event_memory.py"
    spec = importlib.util.spec_from_file_location("selective_event_study", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def intervene(model, split, device, batch_size):
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(split["X"]),
            torch.from_numpy(split["Y"]),
            torch.from_numpy(split["event_label"]),
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    outputs = {"normal": [], 1: [], 2: [], 3: []}
    gates, targets, labels = [], [], []
    model.eval()
    with torch.no_grad():
        for xb, yb, yevent in loader:
            xb = xb.to(device)
            normal = model(xb)
            outputs["normal"].append(normal["prediction"].cpu().numpy())
            for branch in (1, 2, 3):
                ablated = model(xb, reset_memory_index=branch)
                outputs[branch].append(ablated["prediction"].cpu().numpy())
            gates.append(normal["gates"].cpu().numpy())
            targets.append(yb.numpy())
            labels.append(yevent.numpy())
    return (
        {k: np.concatenate(v, axis=0) for k, v in outputs.items()},
        np.concatenate(gates, axis=0),
        np.concatenate(targets, axis=0),
        np.concatenate(labels, axis=0),
    )


def by_event_date(arm, seed, split, outputs, gates, targets, labels):
    records = []
    dates = split["event_day"]
    base = outputs["normal"]
    for cls, event in EVENTS.items():
        for day in sorted(set(dates[(labels == cls)].tolist())):
            ix = np.where((labels == cls) & (dates == day))[0]
            if not len(ix):
                continue
            normal_error = float(np.mean((base[ix] - targets[ix]) ** 2))
            deltas = {}
            for branch in (1, 2, 3):
                reset_error = float(np.mean((outputs[branch][ix] - targets[ix]) ** 2))
                deltas[branch] = reset_error - normal_error
            correct = deltas[cls]
            other = float(np.mean([deltas[b] for b in (1, 2, 3) if b != cls]))
            records.append(
                {
                    "arm": arm,
                    "seed": int(seed),
                    "event": event,
                    "event_date": day,
                    "event_windows": int(len(ix)),
                    "normal_mse": normal_error,
                    "matching_state_reset_delta_mse": correct,
                    "other_state_reset_delta_mse": other,
                    "state_selectivity_gap_mse": correct - other,
                    "mean_matching_event_gate": float(gates[ix, cls].mean()),
                }
            )
    return records


def paired_stats(frame):
    daily = (
        frame.groupby(["arm", "event", "event_date"], as_index=False)[
            [
                "normal_mse",
                "matching_state_reset_delta_mse",
                "other_state_reset_delta_mse",
                "state_selectivity_gap_mse",
            ]
        ]
        .mean()
    )
    tests = []

    def add(family, comparison, metric, a, b):
        x, y = np.asarray(a, float), np.asarray(b, float)
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        if len(x) < 2:
            p, stat, mean = np.nan, np.nan, np.nan
        elif np.allclose(x, y):
            p, stat, mean = 1.0, 0.0, 0.0
        else:
            result = wilcoxon(x, y, alternative="two-sided", method="exact")
            p, stat, mean = float(result.pvalue), float(result.statistic), float(np.mean(x - y))
        tests.append(
            {
                "family": family,
                "comparison": comparison,
                "metric": metric,
                "n_event_dates": len(x),
                "mean_difference": mean,
                "p_value_two_sided": p,
                "wilcoxon_statistic": stat,
            }
        )

    for arm in ARMS:
        for event in EVENTS.values():
            group = daily[(daily.arm == arm) & (daily.event == event)]
            add(
                "within_model_memory_specificity",
                f"{arm} matching-state reset - other-state reset ({event})",
                "state_selectivity_gap_mse",
                group["matching_state_reset_delta_mse"],
                group["other_state_reset_delta_mse"],
            )
    for event in EVENTS.values():
        left = daily[(daily.arm == "event_memory") & (daily.event == event)].set_index("event_date")
        right = daily[(daily.arm == "event_memory_no_event_loss") & (daily.event == event)].set_index("event_date")
        common = left.index.intersection(right.index)
        add(
            "event_supervision_effect",
            f"event_memory - event_memory_no_event_loss ({event})",
            "state_selectivity_gap_mse",
            left.loc[common, "state_selectivity_gap_mse"],
            right.loc[common, "state_selectivity_gap_mse"],
        )
    tests = pd.DataFrame(tests)
    tests["holm_adjusted_p"] = np.nan
    for _, indices in tests.groupby("family").groups.items():
        valid = [i for i in indices if pd.notna(tests.loc[i, "p_value_two_sided"])]
        if not valid:
            continue
        p = tests.loc[valid, "p_value_two_sided"].to_numpy()
        order = np.argsort(p)
        m = len(p)
        values = np.maximum.accumulate((m - np.arange(m)) * p[order])
        adjusted = np.empty(m)
        adjusted[order] = np.minimum(values, 1.0)
        tests.loc[valid, "holm_adjusted_p"] = adjusted
    return daily, tests


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--stage", default="full")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 11, 22, 33, 44])
    args = ap.parse_args()
    driver = load_training_driver()
    study = driver.load_study(Path("configs/experiments/main_forecasting.yaml"))
    device = select_device(args.device)
    torch.set_num_threads(4)
    records = []
    run_root = ROOT / "runs" / "selective_event_memory" / args.stage
    test = study["split_data"]["test"]
    for arm in ARMS:
        for seed in args.seeds:
            run = run_root / arm / f"seed_{seed}"
            checkpoint_path = run / "checkpoint_best.pt"
            if not checkpoint_path.exists():
                raise FileNotFoundError(checkpoint_path)
            model = EventMemoryPSTL(
                input_dim=test["X"].shape[-1],
                n_nodes=test["X"].shape[2],
                n_targets=3,
                horizon=test["Y"].shape[1],
                coords=np.asarray(study["layout"]["coords"], np.float32),
                **{
                    "d_model": study["config"]["model"]["d_model"],
                    "d_state": study["config"]["model"]["d_state"],
                    "n_heads": study["config"]["model"]["n_heads"],
                    "dropout": study["config"]["model"]["dropout"],
                    "n_memories": study["config"]["model"]["memories"],
                },
            ).to(device)
            try:
                state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            except TypeError:
                state = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(state["model"])
            outputs, gates, targets, labels = intervene(model, test, device, args.batch_size)
            records.extend(by_event_date(arm, seed, test, outputs, gates, targets, labels))
            print(f"state intervention done {arm} seed={seed} device={device}", flush=True)

    result_root = ROOT / "results" / "selective_event_memory" / args.stage
    result_root.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(result_root / "state_memory_intervention_by_seed_date.csv", index=False)
    daily, stats = paired_stats(frame)
    daily.to_csv(result_root / "state_memory_intervention_by_date.csv", index=False)
    stats.to_csv(result_root / "state_memory_intervention_statistics.csv", index=False)

    summary = (
        daily.groupby(["arm", "event"], as_index=False)
        .agg(
            n_event_dates=("event_date", "nunique"),
            matching_state_reset_delta_mse=("matching_state_reset_delta_mse", "mean"),
            other_state_reset_delta_mse=("other_state_reset_delta_mse", "mean"),
            state_selectivity_gap_mse=("state_selectivity_gap_mse", "mean"),
        )
    )
    summary.to_csv(result_root / "state_memory_intervention_summary.csv", index=False)
    write_json(
        {
            "purpose": "Test whether recurrent state history itself supports each event class while preserving current input features and router gates.",
            "method": "At inference, reset one event branch's SSM state to zero before every time step. The branch still processes the current token; all other branches and the router are unchanged.",
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "n_runs": len(ARMS) * len(args.seeds),
        },
        result_root / "state_memory_intervention.json",
    )
    print(summary.to_string(index=False), flush=True)
    print(stats.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
