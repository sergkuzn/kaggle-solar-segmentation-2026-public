# filseg: solar filament instance segmentation

My final submission (`v4__unet_r34_v2_bce_x2`) for the Kaggle competition
[filament-segmentation-2026](https://www.kaggle.com/competitions/filament-segmentation-2026):
instance segmentation of solar filaments on 2048×2048 GONG H-alpha images, scored with
Panoptic Quality (PQ).

| | PQ |
|---|---|
| Nested-crossfit out-of-fold PQ (707 training images) | 0.4643 |
| Public leaderboard | 0.40 |

## Where everything is

- **Notebook (start here):** [`notebooks/filseg_pipeline.ipynb`](notebooks/filseg_pipeline.ipynb):
  the whole pipeline end to end, **saved with all its outputs**: data → preprocessing →
  training or weight download → inference → post-processing → evaluation (PQ, IoU / Dice
  distributions, one-to-many / many-to-one relations, inference time) → figures.
- **Trained weights:** GitHub release
  [`weights-v1`](https://github.com/sergkuzn/kaggle-solar-segmentation-2026-public/releases/tag/weights-v1)
  of this repository (936 MB, all 10 networks + LightGBM scorer + tuned parameters).
  The notebook and `make weights` download them automatically. [Details](docs/weights.md).
- **Data:** the official competition data, downloaded by the notebook or `make data` via the
  Kaggle CLI (mounted automatically when run on Kaggle). No external data is used.
- **Code:** the `filseg` package in [`src/filseg`](src/filseg).

## Method in one paragraph

A U-Net (ResNet-34 encoder) sees the full-resolution solar disk as two channels (raw and
background-flattened) and predicts the fraction of annotators marking each pixel. 5 folds × 2
seeds with 8-way TTA are averaged; connected components of the map are candidate filaments, and a
LightGBM scorer keeps those whose predicted IoU is ≥ 0.23. Full description: [docs/method.md](docs/method.md).

## Quick start

Configs, all CLI commands, training from scratch: [docs/usage.md](docs/usage.md).
