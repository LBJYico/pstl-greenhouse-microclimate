# PSTL for Greenhouse Microclimate Forecasting

Reproducible PyTorch code and winter input data for evaluating physically structured spatiotemporal forecasting on a 27-sensor greenhouse network. The repository contains model code, experiment configurations, data preparation and evaluation scripts, and the input CSV files used by the package. Generated experiment outputs are not versioned here.

## Research scope

This project studies artificial intelligence and intelligent sensing for agricultural and biological environmental systems. It combines timestamp-derived solar geometry, multivariate greenhouse sensor histories, and an input-dependent diagonal selective state-space model with spatial attention. The implementation is a portable PyTorch selective SSM inspired by Mamba; it does not require the fused `mamba-ssm` package. The repository supports experiments on forecasting accuracy, calendar-versus-solar representations, component ablations, and event-conditioned state-memory interventions.

The A1–A9 layout represents nine plan-view measurement sites, each with sensors at 1, 2, and 4 m. Site membership and relative coordinates were transcribed from the supplied plan-view figure; the coordinates are not surveyed distances. The code records this limitation in the spatial configuration.

## Data

`data/raw/` contains three winter model-input files: smoothed indoor measurements, outdoor meteorology with solar-related variables, and outdoor meteorology without the solar feature set. Indoor and outdoor timestamps use different source formats; the loader parses them explicitly and aligns exact timestamps. The data manifest records byte sizes and SHA-256 checksums. The analyzed inputs cover the winter period available in the files; see the configuration and audit output for the exact aligned interval and missingness.

The files are included for research reproducibility. No reuse license is asserted in this repository. Please contact the authors before redistributing the data.

## Environment

Python 3.8 or later is recommended. Install a PyTorch build that matches your device first, then install the common dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install PyTorch for the local CPU/MPS or CUDA environment.
python -m pip install -r requirements-common.txt
```

For optional development dependencies, use `requirements-dev.txt`. CEEMDAN ablations additionally require `EMD-signal` (PyEMD), as noted in the configuration and scripts.

## Reproduce a data audit and smoke check

From the repository root:

```bash
python scripts/00_audit_data.py --config configs/data/winter_solar.yaml
python scripts/00_smoke_models.py --device auto
```

The audit is read-only. It reports timestamp alignment, cadence, missingness, target channels, and the status of the figure-derived spatial mapping.

## Build and run experiments

Dataset construction and training are separate commands. For example, to build and train one PSTL configuration:

```bash
python scripts/01_build_dataset.py --config configs/experiments/main_forecasting.yaml --history 12 --horizon 6 --allow-unverified-layout
python scripts/02_train.py --config configs/experiments/main_forecasting.yaml --model pstl --history 12 --horizon 6 --seed 42 --device auto --allow-unverified-layout
```

For a CUDA host, pass `--device cuda`; `--device auto` selects CUDA, then Apple MPS, then CPU. The experiment scripts write generated datasets and outputs under `datasets/`, `runs/`, and `results/`, which are excluded from Git. Review the configuration and compute budget before launching a full matrix.

Key entry points are in `scripts/`; model, data, and reproducibility utilities are in `pstl_experiments/`; protocol and model parameters are in `configs/`. The default protocol uses chronological 70/10/20 train/validation/test splits, training-only scaling, and five seeds. Each run saves its resolved configuration, environment report, logs, metrics, and checkpoint locally.

## Repository contents

- `configs/`: data, model, split, spatial-layout, and experiment definitions
- `data/raw/`: winter indoor and outdoor input CSVs
- `environment/`: runtime and environment reporting
- `pstl_experiments/`: models, data loading, and experiment utilities
- `scripts/`: audit, training, evaluation, aggregation, and plotting entry points
- `tests/`: data-protocol checks

No checkpoints, run logs, generated predictions, result tables, or result figures are included.

## Citation

If you use this code or data, cite the associated PSTL manuscript once its publication details are available and acknowledge the source dataset. A `CITATION.cff` file is included for repository citation metadata.
