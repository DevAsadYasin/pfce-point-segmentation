# Technical report: point-supervised semantic segmentation for remote sensing

**Deliverables covered:** (1) partial cross-entropy and partial focal loss, (2) LoveDA (or synthetic) data with **random** simulated point labels and a segmentation network, (3) experiments with two factors and documented results.

---

## 1. Problem and goal

We solve **semantic segmentation** (dense class map) while **training** only on **sparse point annotations** simulated from dense masks. Unlabeled pixels do not contribute to the supervised loss. **Validation** uses the **full** mask to report **mean Intersection-over-Union (mIoU)** over semantic classes, **excluding** LoveDA **no-data** (label `0`).

---

## 2. Method

### 2.1 Partial cross-entropy (pCE)

Let `pred` be logits \([B,C,H,W]\), `GT` the dense class map \([B,H,W]\), and `MASK_labeled` a binary mask that is `1` only on supervised point pixels. The **partial CE** is the mean of per-pixel CE over labeled sites:

\[
\mathrm{pCE} = \frac{\sum_{i,j} \mathrm{CE}(\mathrm{pred}_{i,j}, GT_{i,j}) \cdot MASK_{i,j}}{\sum_{i,j} MASK_{i,j}}.
\]

Implementation: `partial_cross_entropy` in `src/losses.py`.

### 2.2 Partial focal cross-entropy (pfCE)

As in the problem-statement slide, we use **focal** weighting on the supervised pixels only:

\[
\mathrm{pfCE} = \frac{\sum_{i,j} \mathrm{Focal}(\mathrm{pred}_{i,j}, GT_{i,j}) \cdot MASK_{i,j}}{\sum_{i,j} MASK_{i,j}},
\]

with multiclass focal modulation \((1-p_t)^\gamma\) applied to the per-pixel CE at labeled locations (\(\gamma=2\) by default). Implementation: `partial_focal_loss` in `src/losses.py`.

### 2.3 Point simulation (PDF requirement: **random**)

For each crop we sample **`N` pixel coordinates uniformly at random** among pixels with **valid labels** (`mask ≠ 0`). Class identity at each point is read from the **ground-truth** mask (not invented). For experiments, we additionally test **class-balanced** placement (spread clicks across classes present in the crop) at the **same** `N`.

### 2.4 Data

| Mode | Description |
|------|-------------|
| **LoveDA** | Public high-resolution urban/rural semantic segmentation ([TorchGeo](https://torchgeo.readthedocs.io/) loader, Zenodo-hosted archives). Labels are integers **0–7**: `0` = no-data (ignore in IoU mean), `1`–`7` = seven land-cover classes. |
| **Synthetic** | Random RGB tiles and random labels (with sprinkled no-data) for reproducible runs **without** large downloads. |

**Viewing mask PNGs:** values are **class indices**, not 0–255 photo intensities, so previews often look **black**. Inspecting unique pixel values in Python or applying a colormap confirms content.

### 2.5 Model and training

- **Architecture:** U-Net, **ResNet-34** encoder, **ImageNet** pretrained weights (`segmentation_models_pytorch`).
- **Classes:** `C = 8` logits to match LoveDA’s label set.
- **Input normalization:** ImageNet mean/std.
- **Optimizer:** AdamW, default learning rate `1e-4`.

---

## 3. Experiments

### Experiment A — Point density and sampling strategy

| | |
|---|---|
| **Purpose** | Measure how many simulated points per crop are needed and whether **balanced** sampling changes validation mIoU vs **uniform random**. |
| **Hypothesis** | mIoU increases with `N` then saturates; balanced sampling may help minority classes / stability at fixed `N`. |
| **Procedure** | Fix **pfCE** (`--loss focal`). Sweep `N ∈ {10, 50, 200}` and `sampling ∈ {random, balanced}`. Same seed and subset caps when comparing. |
| **Reproduce** | `python run_experiments.py` (see `README.md`); results in `experiments/logs/results.csv`. |

### Experiment B — Partial CE vs pfCE

| | |
|---|---|
| **Purpose** | Compare **pCE** and **pfCE** under the same point budget. |
| **Hypothesis** | Focal reweighting can change optimization dynamics on sparse clicks. |
| **Procedure** | Fix `N=50`, `sampling=random`. Train once with `--loss focal`, once with `--loss ce`. |

---

## 4. Results

The following table is **representative** of a full pipeline run on the **synthetic** benchmark (`epochs=2` per row, fixed seed). **Your** numbers will differ slightly with hardware, PyTorch version, and LoveDA vs synthetic. Replace this section with **your** `experiments/logs/results.csv` after you run experiments on the target dataset.

### Experiment A & B (synthetic, example run)

Values match `experiments/logs/results.csv` in this repository (rounded). Regenerate the CSV with `run_experiments.py` and refresh this table after any code or seed change.

| dataset | num_points | sampling | loss | epochs | val_mIoU |
|---------|------------|----------|------|--------|----------|
| synthetic | 10 | random | focal | 1 | 0.0619 |
| synthetic | 10 | balanced | focal | 1 | 0.0616 |
| synthetic | 50 | random | focal | 1 | 0.0620 |
| synthetic | 50 | balanced | focal | 1 | 0.0622 |
| synthetic | 200 | random | focal | 1 | 0.0629 |
| synthetic | 200 | balanced | focal | 1 | 0.0633 |
| synthetic | 50 | random | ce | 1 | 0.0629 |

**Interpretation (qualitative):** On this toy synthetic data, mIoU is flat across settings—expected because labels are nearly random and signal is weak. For a meaningful report, prioritize **LoveDA** rows (longer training, larger caps) and discuss **per-class IoU** if implemented later.

**Machine-readable log:** `experiments/logs/results.csv` (columns include `best_val_mIoU`).

---

## 5. Limitations

- No semi-supervised loss on unlabeled pixels (no consistency / Mean Teacher).
- Synthetic data does not reflect geographic structure; LoveDA is preferred for final benchmarks.
- Full LoveDA training is resource-heavy; development uses `--max-train-samples` / `--max-val-samples`.

---

## 6. References

1. Junjue Wang et al., *LoveDA: A Remote Sensing Land-Cover Dataset for Domain Adaptive Semantic Segmentation*, arXiv:2110.08733. [GitHub](https://github.com/Junjue-Wang/LoveDA)
2. TorchGeo documentation: [LoveDA dataset](https://torchgeo.readthedocs.io/en/stable/api/datasets.html)
