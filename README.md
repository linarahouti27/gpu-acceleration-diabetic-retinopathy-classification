# GPU Acceleration of a Diabetic Retinopathy Classification Pipeline

Engineering project: GPU Programming and Performance Optimization (2026).

## Project objective

Profile a CPU reference pipeline for 5-class diabetic retinopathy (DR) classification
(3,000 balanced retinal fundus images), identify its bottlenecks, and redesign/accelerate
selected stages on the GPU while preserving or improving classification quality.

Methodology: **profile → hypothesize → optimize → validate → re-profile**.

## Team members

- TODO
- TODO
- TODO

## Repository structure

```
configs/        Experiment configurations (baseline vs optimized)
src/            CPU reference (*_cpu.py) and GPU-optimized (*_gpu.py) implementations
benchmarks/     Profiling and benchmarking scripts
tests/          Correctness and CPU/GPU equivalence tests
results/        Timings, metrics and figures
report/         Final technical report
environment/    Hardware and software documentation
```

## Environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

See `environment/hardware_software.md` for the exact hardware and software versions used.

## Dataset

The dataset is not included in this repository. Set its location in `configs/*.yaml`
(`data.root`). Expected layout:

```
<data.root>/
├── images/            # RGB fundus images
└── labels.csv         # columns: image_id, patient_id, label (0–4)
```

Labels: 0 = No DR, 1 = Mild, 2 = Moderate, 3 = Severe, 4 = Proliferative DR.

Splits (train / val / test) are created **before** any model development, grouped by
patient so that all images from a patient stay in the same split. The test set is never
used for hyperparameter selection.

## Running the CPU baseline

```bash
python -m src.train_cpu --config configs/baseline.yaml
```

## Running the optimized implementation

```bash
python -m src.train_gpu --config configs/optimized.yaml
```

## Reproducing profiling and benchmarking

```bash
python -m benchmarks.profile_baseline --config configs/baseline.yaml
python -m benchmarks.benchmark_stages --config configs/optimized.yaml
python -m benchmarks.benchmark_end_to_end --config configs/optimized.yaml
```

Results are written to `results/timing.csv`, `results/metrics.json` and `results/figures/`.

## Reproducing model evaluation

```bash
python -m src.evaluate --config configs/optimized.yaml --checkpoint <path/to/checkpoint.pt>
```

## Tests

```bash
pytest tests/
```

## Known compatibility issues and hardware requirements

- An NVIDIA GPU with CUDA is required for the `*_gpu.py` modules and GPU benchmarks.
- CuPy must match the installed CUDA version (e.g. `cupy-cuda12x`).
- TODO: document any issues encountered.
