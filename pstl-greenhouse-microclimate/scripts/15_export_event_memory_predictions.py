#!/usr/bin/env python3
"""Export frozen V2 forecasts and delayed-recall arrays without retraining."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from pstl_experiments.common import select_device, write_json, sha256

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--stage", default="full")
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location("event_memory_v2_study", ROOT / "scripts" / "14_selective_event_memory_v2.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    v1 = driver.import_v1_driver()
    study = v1.load_study(Path("configs/experiments/main_forecasting.yaml"))
    study["config"]["training"] = v1.read_yaml(ROOT / "configs" / "experiments" / "selective_event_memory_v2.yaml")["training"]
    data, recall_meta = driver.make_recall_data(v1, study)
    device = select_device(args.device)
    torch.set_num_threads(4)
    rows = []
    for run in sorted((ROOT / "runs" / "selective_event_memory_v2" / args.stage).glob("*/seed_*")):
        if not (run / "metrics.json").exists():
            continue
        old = json.loads((run / "metrics.json").read_text())
        mode = old["mode"]
        seed = old["seed"]
        if args.stage == "quick" and seed != 41:
            continue
        model = driver.model_for(mode, study, recall_meta).to(device)
        try:
            ck = torch.load(run / "checkpoint_best.pt", map_location="cpu", weights_only=True)
        except TypeError:
            ck = torch.load(run / "checkpoint_best.pt", map_location="cpu")
        model.load_state_dict(ck["model"])
        out = driver.run_model(model, data["test"], device, 32)
        n = out["target"].shape[2]
        scaler = study["scaler"]
        mean = scaler["y_mean"].reshape(1, 1, n, 3)
        std = scaler["y_std"].reshape(1, 1, n, 3)
        yphysical = out["target"] * std + mean
        pphysical = out["prediction"] * std + mean
        recalculated = float(np.mean((out["target"] - out["prediction"]) ** 2))
        if abs(recalculated - old["normalized_mse"]) > 1e-5:
            raise RuntimeError(f"Frozen checkpoint prediction differs from saved metric: {run}")
        np.savez_compressed(
            run / "predictions.npz",
            y_true=yphysical,
            y_pred=pphysical,
            y_true_normalized=out["target"],
            y_pred_normalized=out["prediction"],
            recall_target=out["recall_target"],
            recall_prediction=out["recall"],
            recall_mask=out["recall_mask"],
            target_start=data["test"]["target_start"],
            event_flags=data["test"]["event_flags"],
            event_day=data["test"]["event_day_v2"].astype(str),
        )
        row = {"mode": mode, "seed": int(seed), "normalized_mse": recalculated}
        for i, name in enumerate(("temperature", "humidity", "light")):
            row[f"{name}_rmse"] = float(np.sqrt(np.mean((pphysical[..., i] - yphysical[..., i]) ** 2)))
        rows.append(row)
        write_json(row, run / "physical_metrics.json")
        write_json({
            "device": str(device),
            "export_script_sha256": sha256(Path(__file__).resolve()),
            "model_sha256": sha256(ROOT / "pstl_experiments" / "event_memory_v2.py"),
            "prediction_mse_matches_training_evaluation": True,
            "checkpoint_sha256": sha256(run / "checkpoint_best.pt"),
        }, run / "evaluation_manifest.json")
        print(f"exported {mode} seed={seed} mse={recalculated:.5f}", flush=True)
    result_root = ROOT / "results" / "selective_event_memory_v2" / args.stage
    result_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(result_root / "physical_forecast_metrics.csv", index=False)
    print(pd.DataFrame(rows).groupby("mode").agg({"temperature_rmse": ["mean", "std"], "humidity_rmse": ["mean", "std"], "light_rmse": ["mean", "std"]}).to_string(), flush=True)


if __name__ == "__main__":
    main()
