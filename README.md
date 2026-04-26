# Point-supervised remote sensing segmentation

Personal short test project implementing **partial cross-entropy (pCE)** and **partial focal cross-entropy (pfCE)** on **simulated point labels**, trained on **[LoveDA](https://github.com/Junjue-Wang/LoveDA)** (real remote sensing) or a **synthetic** benchmark for fast checks.

## What is included

| Item | Description |
|------|-------------|
| `src/losses.py` | Masked pCE and pfCE (normalized by number of labeled pixels). |
| `src/data.py` | LoveDA loader (TorchGeo), random/balanced point simulation, synthetic dataset. |
| `src/model.py` | U-Net + ResNet-34 (ImageNet weights) via `segmentation_models_pytorch`. |
| `train.py` | Training and **full-mask validation mIoU** (LoveDA label `0` = no-data excluded from IoU mean). |
| `run_experiments.py` | Experiment grid → `experiments/logs/results.csv`. |
| `tests/test_losses.py` | Sanity checks for the loss implementations. |
| `TECHNICAL_REPORT.md` | Method, experiments (purpose / hypothesis / procedure), and results. |

## Requirements

- **Python 3.10+** (3.11 recommended)
- **Disk:** a few MB for code; **~4 GB+ per split** if you download LoveDA train/val
- **GPU:** optional; CPU runs work for small epochs and synthetic data

## Setup (virtual environment)

Do **not** install dependencies system-wide.

```bash
cd /path/to/TestProject
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For GPU PyTorch, install the matching `torch` / `torchvision` wheels from [pytorch.org](https://pytorch.org/) (then install the rest of `requirements.txt` if needed).

## Quick verification

```bash
source .venv/bin/activate   # or: .venv/bin/python tests/test_losses.py
python tests/test_losses.py
```

Expected: `ok`  
If your shell does not load the venv (some CI setups), call the interpreter explicitly: `.venv/bin/python tests/test_losses.py`.

## Training

Run from the **project root** (the repo root that contains `src/` and `train.py`).

### Synthetic (no download; good for CI and smoke tests)

```bash
python train.py --dataset synthetic --epochs 8 --batch-size 8 --crop-size 256 \
  --num-points 50 --loss focal --sampling random --seed 42
```

### LoveDA (real remote sensing)

First run downloads Zenodo archives into `--data-root` (large). Use caps while developing:

```bash
python train.py --dataset loveda --data-root ./data/loveda --download \
  --epochs 10 --batch-size 4 --crop-size 512 \
  --num-points 50 --loss focal --sampling random \
  --max-train-samples 400 --max-val-samples 150 --seed 42
```

**Note on mask PNGs:** segmentation masks use **small integer class IDs** (0–7). Many viewers show them as **almost black**; that is normal. See **TECHNICAL_REPORT.md** (FAQ).

## Experiments (CSV)

Runs Experiment A (point density × sampling, pfCE) and Experiment B (pCE vs pfCE at fixed settings):

```bash
python run_experiments.py --dataset synthetic --epochs 3 --batch-size 8 --crop-size 256 \
  --max-train-samples 200 --max-val-samples 80 --seed 42
```

LoveDA (after data is present):

```bash
python run_experiments.py --dataset loveda --data-root ./data/loveda --epochs 5 \
  --max-train-samples 500 --max-val-samples 200
# Add --download only if ./data/loveda is not populated yet.
```

Output: `experiments/logs/results.csv`

## Main CLI options (`train.py`)

| Option | Meaning |
|--------|---------|
| `--dataset` | `synthetic` or `loveda` |
| `--download` | Download LoveDA into `--data-root` (huge) |
| `--num-points` | Simulated clicks per crop (`N`) |
| `--sampling` | `random` (PDF baseline) or `balanced` (experiment) |
| `--loss` | `focal` (pfCE) or `ce` (pCE) |
| `--epochs`, `--batch-size`, `--lr`, `--crop-size` | Training knobs |
| `--max-train-samples`, `--max-val-samples` | Subset caps for speed |

## Creating a project zip

Exclude the venv and any downloaded data. From the project root:

```bash
./scripts/package_submission.sh ./submission.zip
```

Or manually zip the repository **without** `.venv/`, `data/`, and `__pycache__/`.

The produced archive should include: `src/`, `train.py`, `run_experiments.py`, `requirements.txt`, `tests/`, `experiments/logs/results.csv` (after you run experiments), `TECHNICAL_REPORT.md`, and this `README.md`.

## Citation (LoveDA)

If you use LoveDA, cite:

> Wang et al., *LoveDA: A Remote Sensing Land-Cover Dataset for Domain Adaptive Semantic Segmentation*, arXiv:2110.08733.

## License

This repository is a personal short test project. **LoveDA** has its own license; follow the [LoveDA repository](https://github.com/Junjue-Wang/LoveDA) terms when using the data.
# pfce-point-segmentation
