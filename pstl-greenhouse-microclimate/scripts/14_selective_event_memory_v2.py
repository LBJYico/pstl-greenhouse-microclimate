#!/usr/bin/env python3
"""Train and evaluate physics-gated event memories with delayed state-only recall."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon
from torch.utils.data import DataLoader, TensorDataset

from pstl_experiments.common import environment_report, select_device, set_seed, sha256, write_json
from pstl_experiments.event_memory_v2 import WriteGatedEventMemory

ROOT = Path(__file__).resolve().parents[1]
MODES = ("gated", "ungated", "no_recurrence")
EVENT_CLASSES = {"sunrise": 0, "noon": 1, "cloudy_transition": 2}


def import_v1_driver():
    spec = importlib.util.spec_from_file_location(
        "selective_event_study", ROOT / "scripts" / "12_selective_event_memory.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_recall_data(driver, study):
    layout = study["layout"]
    global_cols = layout["global_columns"]
    global_offset = 3 + global_cols.index("light")
    input_indices = (0, 1, 2, global_offset)
    result = {}
    for split, data in study["split_data"].items():
        x = data["X"]
        starts = data["target_start"]
        flags = np.zeros((len(x), x.shape[1], 3), dtype=np.float32)
        targets = np.zeros((len(x), 3, x.shape[2], 4), dtype=np.float32)
        mask = np.zeros((len(x), 3), dtype=bool)
        days = np.full((len(x), 3), "", dtype=object)
        event_indices = {name: [] for name in EVENT_CLASSES}
        for event in study["events"]:
            event_indices[event["kind"]].append(event)
        for row, target_start in enumerate(starts):
            start = int(target_start) - x.shape[1]
            end = start + x.shape[1]
            for name, cls in EVENT_CLASSES.items():
                # Every already observed event may write, including recent
                # events. Only delayed events are scored by the recall task.
                for observed_event in event_indices[name]:
                    observed_pos = int(observed_event["index"]) - start
                    if 0 <= observed_pos < x.shape[1]:
                        flags[row, observed_pos, cls] = 1.0
                candidates = [
                    event
                    for event in event_indices[name]
                    if start + 1 <= int(event["index"]) <= start + 7
                ]
                if not candidates:
                    continue
                event = max(candidates, key=lambda item: item["index"])
                pos = int(event["index"]) - start
                delta = x[row, pos][:, input_indices] - x[row, pos - 1][:, input_indices]
                targets[row, cls] = delta
                mask[row, cls] = True
                days[row, cls] = event["day"]
        result[split] = {
            **data,
            "event_flags": flags,
            "recall_target": targets,
            "recall_mask": mask,
            "event_day_v2": days,
        }

    train = result["train"]
    values = train["recall_target"][train["recall_mask"]]
    scale = values.reshape(-1, 4).std(axis=0).clip(1e-3).astype(np.float32)
    for data in result.values():
        data["recall_target"] = data["recall_target"] / scale.reshape(1, 1, 1, 4)
    meta = {"input_indices": list(input_indices), "recall_scale": scale.tolist()}
    return result, meta


def model_for(mode, study, recall_meta):
    x = study["split_data"]["train"]["X"]
    return WriteGatedEventMemory(
        input_dim=x.shape[-1],
        n_nodes=x.shape[2],
        horizon=study["split_data"]["train"]["Y"].shape[1],
        coords=np.asarray(study["layout"]["coords"], dtype=np.float32),
        innovation_scale=np.asarray(recall_meta["recall_scale"], dtype=np.float32),
        innovation_indices=recall_meta["input_indices"],
        mode=mode,
        d_model=study["config"]["training"]["d_model"],
        d_state=study["config"]["training"]["d_state"],
        dropout=study["config"]["training"]["dropout"],
    )


def run_model(model, data, device, batch_size, reset_branch=None):
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(data["X"]),
            torch.from_numpy(data["Y"]),
            torch.from_numpy(data["event_flags"]),
            torch.from_numpy(data["recall_target"]),
            torch.from_numpy(data["recall_mask"]),
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    model.eval()
    pred, recalls, ys, rys, masks = [], [], [], [], []
    with torch.no_grad():
        for xb, yb, flags, rtarget, rmask in loader:
            output = model(xb.to(device), flags.to(device), reset_branch=reset_branch)
            pred.append(output["prediction"].cpu().numpy())
            recalls.append(output["recall"].cpu().numpy())
            ys.append(yb.numpy())
            rys.append(rtarget.numpy())
            masks.append(rmask.numpy())
    return {
        "prediction": np.concatenate(pred),
        "recall": np.concatenate(recalls),
        "target": np.concatenate(ys),
        "recall_target": np.concatenate(rys),
        "recall_mask": np.concatenate(masks),
    }


def evaluate(model, mode, seed, data, study, device, batch_size):
    normal = run_model(model, data, device, batch_size)
    resets = {
        cls: run_model(model, data, device, batch_size, reset_branch=cls + 1)
        for cls in range(3)
    }
    pred, y = normal["prediction"], normal["target"]
    result = {
        "mode": mode,
        "seed": int(seed),
        "normalized_mse": float(np.mean((pred - y) ** 2)),
        "recall_mse_by_event": {},
    }
    if mode == "no_recurrence":
        # Still evaluate recall performance; this controls for the current token
        # or explicit event indicator directly predicting the observed change.
        pass
    for cls, name in enumerate(EVENT_CLASSES):
        sample_mask = normal["recall_mask"][:, cls]
        if not sample_mask.any():
            continue
        target = normal["recall_target"][:, cls]
        estimate = normal["recall"][:, cls]
        err = float(np.mean((estimate[sample_mask] - target[sample_mask]) ** 2))
        baseline = float(np.mean(target[sample_mask] ** 2))
        result["recall_mse_by_event"][name] = err
        result.setdefault("recall_r2_by_event", {})[name] = float(1 - err / max(baseline, 1e-8))
    dates = data["event_day_v2"]
    day_rows = []
    for cls, name in enumerate(EVENT_CLASSES):
        day_values = sorted(set(dates[normal["recall_mask"][:, cls], cls].tolist()))
        for day in day_values:
            ix = np.where(normal["recall_mask"][:, cls] & (dates[:, cls] == day))[0]
            if not len(ix):
                continue
            normal_recall = float(np.mean((normal["recall"][ix, cls] - normal["recall_target"][ix, cls]) ** 2))
            ablated_recall = float(np.mean((resets[cls]["recall"][ix, cls] - normal["recall_target"][ix, cls]) ** 2))
            other_recall = []
            for other in range(3):
                if other == cls:
                    continue
                other_recall.append(
                    float(np.mean((resets[other]["recall"][ix, cls] - normal["recall_target"][ix, cls]) ** 2))
                )
            normal_forecast = float(np.mean((normal["prediction"][ix] - normal["target"][ix]) ** 2))
            reset_forecast = float(np.mean((resets[cls]["prediction"][ix] - normal["target"][ix]) ** 2))
            day_rows.append(
                {
                    "mode": mode,
                    "seed": int(seed),
                    "event": name,
                    "event_date": day,
                    "windows": len(ix),
                    "normal_recall_mse": normal_recall,
                    "matching_state_reset_recall_delta": ablated_recall - normal_recall,
                    "other_state_reset_recall_delta": float(np.mean(other_recall)) - normal_recall,
                    "recall_selectivity_gap": (ablated_recall - normal_recall) - (float(np.mean(other_recall)) - normal_recall),
                    "matching_state_reset_forecast_delta": reset_forecast - normal_forecast,
                }
            )
    result["event_date_metrics"] = day_rows
    result["recall_selectivity_gap"] = float(np.mean([row["recall_selectivity_gap"] for row in day_rows])) if day_rows else np.nan
    return result


def paired_date_tests(date_frame):
    daily = (
        date_frame.groupby(["mode", "event", "event_date"], as_index=False)[
            ["normal_recall_mse", "matching_state_reset_recall_delta", "other_state_reset_recall_delta", "recall_selectivity_gap", "matching_state_reset_forecast_delta"]
        ].mean()
    )
    rows = []

    def add(family, comparison, metric, a, b=None):
        x = np.asarray(a, dtype=float)
        if b is None:
            y = np.zeros_like(x)
        else:
            y = np.asarray(b, dtype=float)
        keep = np.isfinite(x) & np.isfinite(y)
        x, y = x[keep], y[keep]
        if len(x) < 2:
            stat, pvalue, mean = np.nan, np.nan, np.nan
        elif np.allclose(x, y):
            stat, pvalue, mean = 0.0, 1.0, 0.0
        else:
            test = wilcoxon(x, y, alternative="two-sided", method="exact")
            stat, pvalue, mean = float(test.statistic), float(test.pvalue), float(np.mean(x - y))
        rows.append({"family": family, "comparison": comparison, "metric": metric, "n_event_dates": len(x), "mean_difference": mean, "p_value_two_sided": pvalue, "wilcoxon_statistic": stat})

    for mode in MODES:
        for event in EVENT_CLASSES:
            group = daily[(daily["mode"] == mode) & (daily["event"] == event)]
            if mode == "gated":
                # Primary mechanistic test: resetting the matching recurrent state
                # should increase state-only delayed-recall error on event dates.
                add(
                    "gated_state_dependence",
                    f"gated matching-state reset vs intact ({event})",
                    "matching_state_reset_recall_delta",
                    group["matching_state_reset_recall_delta"],
                )
                add(
                    "forecast_state_dependence",
                    f"gated matching-state reset forecast effect ({event})",
                    "matching_state_reset_forecast_delta",
                    group["matching_state_reset_forecast_delta"],
                )
    for event in EVENT_CLASSES:
        left = daily[(daily["mode"] == "gated") & (daily["event"] == event)].set_index("event_date")
        for mode in ("ungated", "no_recurrence"):
            right = daily[(daily["mode"] == mode) & (daily["event"] == event)].set_index("event_date")
            common = left.index.intersection(right.index)
            add(
                "gated_vs_control",
                f"gated - {mode} ({event})",
                "normal_recall_mse",
                left.loc[common, "normal_recall_mse"],
                right.loc[common, "normal_recall_mse"],
            )
    tests = pd.DataFrame(rows)
    tests["holm_adjusted_p"] = np.nan
    for _, indices in tests.groupby("family").groups.items():
        valid = [idx for idx in indices if pd.notna(tests.loc[idx, "p_value_two_sided"])]
        if not valid:
            continue
        p = tests.loc[valid, "p_value_two_sided"].to_numpy()
        order = np.argsort(p)
        m = len(p)
        h = np.maximum.accumulate((m - np.arange(m)) * p[order])
        adjusted = np.empty(m)
        adjusted[order] = np.minimum(h, 1.0)
        tests.loc[valid, "holm_adjusted_p"] = adjusted
    tests = tests.rename(columns={"n_event_dates": "n_units"})
    return daily, tests


def seed_level_tests(seed_frame):
    rows = []
    for control in ("ungated", "no_recurrence"):
        left = seed_frame[seed_frame["mode"] == "gated"].set_index("seed")
        right = seed_frame[seed_frame["mode"] == control].set_index("seed")
        common = left.index.intersection(right.index)
        for metric in ("normalized_mse", "recall_selectivity_gap"):
            x = left.loc[common, metric].to_numpy(float)
            y = right.loc[common, metric].to_numpy(float)
            if len(x) < 2:
                statistic, pvalue = np.nan, np.nan
            elif np.allclose(x, y):
                statistic, pvalue = 0.0, 1.0
            else:
                result = wilcoxon(x, y, alternative="two-sided", method="exact")
                statistic, pvalue = float(result.statistic), float(result.pvalue)
            rows.append({
                "family": "seed_level",
                "comparison": f"gated - {control}",
                "metric": metric,
                "n_units": len(common),
                "mean_difference": float(np.mean(x - y)),
                "p_value_two_sided": pvalue,
                "wilcoxon_statistic": statistic,
                "holm_adjusted_p": np.nan,
            })
    result = pd.DataFrame(rows)
    p = result["p_value_two_sided"].to_numpy(float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    h = np.maximum.accumulate((m - np.arange(m)) * p[order])
    adjusted[order] = np.minimum(h, 1.0)
    result["holm_adjusted_p"] = adjusted
    return result


def write_v2_report(stage, seed_frame, event_summary, stats, result_root, recall_meta):
    if stage == "full":
        result_interpretation = (
            "The full study supports retained event-time information in the noon and cloudy-transition memories: resetting those states raises delayed-recall error on their held-out dates (Holm p=0.0029 for noon; p=0.0313 for cloudy transition). Sunrise shows no such effect (Holm p=0.9219). The event-gated memory improves noon recall over both continuous-write and no-recurrence controls (Holm p=0.0059 and 0.0488). Cloudy-transition recall is better on average than both controls, but neither comparison passes the family correction (Holm p=0.0625); sunrise does not improve. Thus the evidence supports event-specific memory for noon and stored cloudy-event information, but not all three event classes or a general advantage of event gating."
        )
        forecast_interpretation = (
            "This state memory has not been shown to improve forecasting. Resetting the noon state lowers forecast MSE by 0.0307 on average (Holm p=0.0410), while forecast-state effects for sunrise and cloudy transition are not significant. The five-seed full-forecast comparisons do not establish a forecast-MSE advantage for gated memory over either control. Treat V2 as evidence for selective retention on the auxiliary recall task, not as evidence that those memories improve the greenhouse forecasts."
        )
    else:
        result_interpretation = (
            "This quick validation uses one seed and does not establish a population-level or paper-confirmatory effect. Read the date-paired results as screening evidence for the planned full study."
        )
        forecast_interpretation = (
            "The recall intervention and forecast intervention are separate outcomes; a quick-stage recall effect does not imply improved forecasting."
        )
    lines = [
        f"# Selective event memory V2: {stage}",
        "",
        "The model uses an input-dependent diagonal selective SSM, with a stable negative diagonal transition and input-dependent Δ, B and C, following the selective-state-space design in Mamba ([Gu & Dao, 2023](https://arxiv.org/abs/2312.00752)). It is a custom pure-PyTorch Mamba-inspired implementation, not the fused official mamba-ssm package. Four branches hold general, sunrise, noon, and cloudy-transition information. Event flags are supplied from the causal physical definitions below; this experiment tests retention conditional on a detected event, not autonomous event discovery. Event branches accept state writes only at already observed events. A state-only readout is trained to recover four normalized observation innovations (indoor temperature, humidity, light, and outdoor light) from the event token after at least four 10-minute steps. The readout sees only the final recurrent state. The slower positive-Δ initialization follows the inverse-softplus initialization pattern in the [official Mamba implementation](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py), using the range set in the frozen V2 protocol.",
        "",
        "Controls are continuous writes to all branches (`ungated`) and no recurrence (`no_recurrence`). At evaluation, one branch's recurrent history is reset at every step while its event token, all observations, and all other branches remain unchanged. This tests retained historical information separately from event classification. The chronological split was already inspected in V1, so this study is exploratory, not an independent confirmatory holdout.",
        "",
        "The initial seed-42 quick run exposed an implementation issue: only recall-eligible events were being written. Before full training, the script was corrected so every already observed event in the context can write, while the recall loss still requires a four-step delay; that first quick run is excluded. The corrected quick screen used seed 41, and the full experiment used the same corrected code.",
        "",
        "Recall targets are standardized by train-only event-innovation scales: " + ", ".join(f"{v:.4f}" for v in recall_meta["recall_scale"]) + ".",
        "",
        "## Held-out event-date results",
        "",
        "| Event | Dates | Gated recall MSE | Reset matching state Δ recall MSE | Holm p | Gated minus ungated recall MSE (Holm p) | Gated minus no-recurrence (Holm p) | Reset matching state Δ forecast MSE | Holm p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    event_order = list(EVENT_CLASSES)
    for event in event_order:
        item = event_summary[(event_summary["mode"] == "gated") & (event_summary["event"] == event)]
        if item.empty:
            continue
        row = item.iloc[0]
        reset_test = stats[(stats["family"] == "gated_state_dependence") & stats["comparison"].str.endswith(f"({event})")].iloc[0]
        forecast_test = stats[(stats["family"] == "forecast_state_dependence") & stats["comparison"].str.endswith(f"({event})")].iloc[0]
        control_tests = {}
        for control in ("ungated", "no_recurrence"):
            control_tests[control] = stats[(stats["family"] == "gated_vs_control") & (stats["comparison"] == f"gated - {control} ({event})")].iloc[0]
        lines.append(
            f"| {event} | {int(row.n_dates)} | {row.normal_recall_mse:.4f} | "
            f"{reset_test.mean_difference:.4f} | {reset_test.holm_adjusted_p:.4f} | "
            f"{control_tests['ungated'].mean_difference:.4f} ({control_tests['ungated'].holm_adjusted_p:.4f}) | "
            f"{control_tests['no_recurrence'].mean_difference:.4f} ({control_tests['no_recurrence'].holm_adjusted_p:.4f}) | "
            f"{row.forecast_reset_delta:.5f} | {forecast_test.holm_adjusted_p:.4f} |"
        )
    lines.extend([
        "",
        "State-reset and gated-versus-control tests are paired by held-out event date after averaging seeds. Exact two-sided Wilcoxon tests are Holm-corrected within each predeclared family. A positive recall-reset effect means recalling the event-time observation became harder when the corresponding recurrent state was erased. For the forecast column, a negative effect means resetting the state lowered forecast error. Branch-reset versus other-branch-reset effects are not used as evidence because the branches are architecturally separated.",
        "",
        "## Full-forecast performance (five seeds)",
        "",
        "| Mode | Normalized test MSE (mean ± SD) |",
        "|---|---:|",
    ])
    for mode, group in seed_frame.groupby("mode"):
        lines.append(f"| {mode} | {group.normalized_mse.mean():.4f} ± {group.normalized_mse.std(ddof=1):.4f} |")
    physical_path = result_root / "physical_forecast_metrics.csv"
    if physical_path.exists():
        physical = pd.read_csv(physical_path)
        lines.extend([
            "",
            "| Mode | Temperature RMSE | Humidity RMSE | Light RMSE |",
            "|---|---:|---:|---:|",
        ])
        for mode, group in physical.groupby("mode"):
            values = []
            for metric in ("temperature_rmse", "humidity_rmse", "light_rmse"):
                sd = group[metric].std(ddof=1) if len(group) > 1 else 0.0
                values.append(f"{group[metric].mean():.3f} ± {sd:.3f}")
            lines.append(f"| {mode} | {values[0]} | {values[1]} | {values[2]} |")
    lines.extend([
        "",
        "## Paired seed results",
        "",
        "| Comparison | Metric | Mean paired difference | Seeds | Exact p | Holm p |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for _, row in stats[stats["family"] == "seed_level"].iterrows():
        lines.append(
            f"| {row.comparison} | {row.metric} | {row.mean_difference:.4f} | {int(row.n_units)} | "
            f"{row.p_value_two_sided:.4f} | {row.holm_adjusted_p:.4f} |"
        )
    lines.extend([
        "",
        result_interpretation,
        "",
        forecast_interpretation,
        "",
        "Artifacts: `run_metrics.csv`, `physical_forecast_metrics.csv`, `event_type_summary.csv`, `event_date_metrics.csv`, `paired_statistics.csv`, per-seed `predictions.npz` (physical/normalized forecasts, recall targets and outputs, event flags and target indices), `history.csv`, checkpoints, environment reports, evaluation manifests, and resolved code hashes.",
        "",
    ])
    (result_root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def train_run(mode, seed, stage, study, data, recall_meta, device, args):
    train, valid, test = data["train"], data["validation"], data["test"]
    run = ROOT / "runs" / "selective_event_memory_v2" / stage / mode / f"seed_{seed}"
    run.mkdir(parents=True, exist_ok=True)
    manifest_path = run / "manifest.json"
    existing = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if existing.get("status") == "completed":
        return json.loads((run / "metrics.json").read_text())
    set_seed(seed)
    cfg = study["config"]["training"]
    model = model_for(mode, study, recall_meta).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=0.001)
    indices = np.arange(len(train["X"]))
    if args.max_train_windows and len(indices) > args.max_train_windows:
        rng = np.random.default_rng(seed)
        event_ix = np.where(train["recall_mask"].any(axis=1))[0]
        other_ix = np.where(~train["recall_mask"].any(axis=1))[0]
        keep_events = rng.choice(event_ix, min(len(event_ix), args.max_train_windows // 2), replace=False)
        keep_other = rng.choice(other_ix, max(0, args.max_train_windows - len(keep_events)), replace=False)
        indices = np.sort(np.concatenate((keep_events, keep_other)))
    loader = DataLoader(
        TensorDataset(*[
            torch.from_numpy(train[key][indices])
            for key in ("X", "Y", "event_flags", "recall_target", "recall_mask")
        ]),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_set = TensorDataset(*[
        torch.from_numpy(valid[key])
        for key in ("X", "Y", "event_flags", "recall_target", "recall_mask")
    ])
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    best, wait, best_state, history, start_epoch = float("inf"), 0, None, [], 1
    if existing.get("status") in ("running", "failed") and (run / "checkpoint_last.pt").exists():
        ck = torch.load(run / "checkpoint_last.pt", map_location="cpu")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        best, wait, history, start_epoch = ck["best"], ck["wait"], ck["history"], ck["epoch"] + 1
        if (run / "checkpoint_best.pt").exists():
            best_state = torch.load(run / "checkpoint_best.pt", map_location="cpu")["model"]
    write_json({"status": "running", "mode": mode, "seed": seed, "device": str(device)}, manifest_path)
    write_json(environment_report(device), run / "environment.json")
    code_paths = (
        ROOT / "pstl_experiments" / "event_memory_v2.py",
        ROOT / "scripts" / "14_selective_event_memory_v2.py",
        ROOT / "configs" / "experiments" / "selective_event_memory_v2.yaml",
        ROOT / "reports" / "EVENT_MEMORY_V2_PROTOCOL.md",
    )
    write_json(
        {
            "mode": mode,
            "seed": seed,
            "stage": stage,
            "training_config": cfg,
            "recall_scale": recall_meta["recall_scale"],
            "innovation_indices": recall_meta["input_indices"],
            "code_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in code_paths},
        },
        run / "config_resolved.json",
    )
    started = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        losses, recalls = [], []
        for xb, yb, flags, rtarget, rmask in loader:
            xb, yb, flags = xb.to(device), yb.to(device), flags.to(device)
            rtarget, rmask = rtarget.to(device), rmask.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(xb, flags)
            forecast_loss = F.mse_loss(output["prediction"], yb)
            mask = rmask[:, :, None, None].expand_as(output["recall"])
            recall_loss = F.mse_loss(output["recall"][mask], rtarget[mask]) if mask.any() else forecast_loss * 0
            loss = forecast_loss + cfg["recall_loss_weight"] * recall_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(forecast_loss.detach().cpu()))
            recalls.append(float(recall_loss.detach().cpu()))
        model.eval()
        val_loss, val_recall = [], []
        with torch.no_grad():
            for xb, yb, flags, rtarget, rmask in val_loader:
                xb, yb, flags = xb.to(device), yb.to(device), flags.to(device)
                rtarget, rmask = rtarget.to(device), rmask.to(device)
                out = model(xb, flags)
                val_loss.append(float(F.mse_loss(out["prediction"], yb).cpu()))
                mask = rmask[:, :, None, None].expand_as(out["recall"])
                if mask.any():
                    val_recall.append(float(F.mse_loss(out["recall"][mask], rtarget[mask]).cpu()))
        row = {
            "epoch": epoch,
            "train_mse": float(np.mean(losses)),
            "train_recall_mse": float(np.mean(recalls)),
            "validation_mse": float(np.mean(val_loss)),
            "validation_recall_mse": float(np.mean(val_recall)) if val_recall else np.nan,
        }
        history.append(row)
        print(f"{stage}/{mode}/seed_{seed} epoch={epoch} forecast={row['validation_mse']:.4f} recall={row['validation_recall_mse']:.4f}", flush=True)
        # Joint criterion prevents tuning only to either forecast or the recall task.
        score = row["validation_mse"] + cfg["recall_loss_weight"] * row["validation_recall_mse"]
        if score < best:
            best, wait, best_state = score, 0, copy.deepcopy(model.state_dict())
            torch.save({"model": best_state, "epoch": epoch}, run / "checkpoint_best.pt")
        else:
            wait += 1
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "best": best, "wait": wait, "history": history}, run / "checkpoint_last.pt")
        if wait >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("No V2 checkpoint produced")
    model.load_state_dict(best_state)
    metrics = evaluate(model, mode, seed, test, study, device, args.eval_batch_size)
    metrics.update({"stage": stage, "device": str(device), "parameter_count": sum(p.numel() for p in model.parameters()), "train_windows": len(indices), "epochs_completed": len(history), "runtime_seconds": time.time() - started})
    with (run / "history.csv").open("w", newline="") as f:
        pd.DataFrame(history).to_csv(f, index=False)
    write_json(metrics, run / "metrics.json")
    write_json({"status": "completed", "run_dir": str(run)}, manifest_path)
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("quick", "full"), required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--patience", type=int)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--eval-batch-size", type=int, default=32)
    ap.add_argument("--max-train-windows", type=int)
    ap.add_argument("--seeds", nargs="*", type=int)
    args = ap.parse_args()
    driver = import_v1_driver()
    study = driver.load_study(Path("configs/experiments/main_forecasting.yaml"))
    v2_config = driver.read_yaml(ROOT / "configs" / "experiments" / "selective_event_memory_v2.yaml")
    study["config"]["training"] = v2_config["training"]
    data, recall_meta = make_recall_data(driver, study)
    args.epochs = args.epochs or (v2_config["training"]["quick_epochs"] if args.stage == "quick" else v2_config["training"]["full_epochs"])
    args.patience = args.patience or (args.epochs if args.stage == "quick" else v2_config["training"]["patience"])
    args.max_train_windows = args.max_train_windows or (v2_config["training"]["quick_max_train_windows"] if args.stage == "quick" else 0)
    seeds = args.seeds or ([42] if args.stage == "quick" else v2_config["training"]["seeds"])
    device = select_device(args.device)
    torch.set_num_threads(4)
    print(f"V2 stage={args.stage} device={device} seeds={seeds} max_train={args.max_train_windows}", flush=True)
    for split in ("train", "validation", "test"):
        print(split, {k: int(data[split]["recall_mask"][:, i].sum()) for i, k in enumerate(EVENT_CLASSES)}, flush=True)
    rows = []
    for mode in MODES:
        for seed in seeds:
            metrics = train_run(mode, seed, args.stage, study, data, recall_meta, device, args)
            rows.append(metrics)
    result_root = ROOT / "results" / "selective_event_memory_v2" / args.stage
    result_root.mkdir(parents=True, exist_ok=True)
    flat = [{k: v for k, v in r.items() if k != "event_date_metrics" and k not in ("recall_mse_by_event", "recall_r2_by_event")} for r in rows]
    pd.DataFrame(flat).to_csv(result_root / "run_metrics.csv", index=False)
    event_rows = [date for row in rows for date in row.get("event_date_metrics", [])]
    date_frame = pd.DataFrame(event_rows)
    date_frame.to_csv(result_root / "event_date_metrics_by_seed.csv", index=False)
    daily, stats = paired_date_tests(date_frame)
    seed_stats = seed_level_tests(pd.DataFrame(flat))
    stats = pd.concat([stats, seed_stats], ignore_index=True)
    daily.to_csv(result_root / "event_date_metrics.csv", index=False)
    stats.to_csv(result_root / "paired_statistics.csv", index=False)
    summary = date_frame.groupby(["mode", "event"]).agg(
        n_dates=("event_date", "nunique"),
        normal_recall_mse=("normal_recall_mse", "mean"),
        recall_reset_delta=("matching_state_reset_recall_delta", "mean"),
        wrong_reset_delta=("other_state_reset_recall_delta", "mean"),
        state_selectivity_gap=("recall_selectivity_gap", "mean"),
        forecast_reset_delta=("matching_state_reset_forecast_delta", "mean"),
    ).reset_index()
    summary.to_csv(result_root / "event_type_summary.csv", index=False)
    write_v2_report(args.stage, pd.DataFrame(flat), summary, stats, result_root, recall_meta)
    write_json({
        "method": "physics-gated event-token writes plus delayed state-only observation-innovation recall; all labels from past/current deterministic event definitions",
        "training_config": v2_config["training"],
        "recall_scale": recall_meta["recall_scale"],
        "input_indices": recall_meta["input_indices"],
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "runs": len(MODES) * len(seeds),
        "analysis_limit": "The time split was already inspected by V1, so this follow-up remains exploratory.",
    }, result_root / "protocol_and_environment.json")
    print(summary.to_string(index=False), flush=True)
    print(stats.to_string(index=False), flush=True)
if __name__ == "__main__":
    main()
