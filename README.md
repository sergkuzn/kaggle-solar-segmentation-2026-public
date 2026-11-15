# filseg: solar filament instance segmentation

Code for the Kaggle competition
[filament-segmentation-2026](https://www.kaggle.com/competitions/filament-segmentation-2026):
instance segmentation of solar filaments on 2048×2048 GONG H-alpha images, scored with
Panoptic Quality (PQ) per annotator record.

This repository holds the pipeline of my final submission, `v4__unet_r34_v2_bce_x2`, and nothing
else. The walkthrough is [`notebooks/filseg_pipeline.ipynb`](notebooks/filseg_pipeline.ipynb):
data → preprocessing → training (or downloading the trained weights, which are stored in this
repository's GitHub release) → inference → post-processing → evaluation (PQ, the IoU / Dice
distributions, the one-to-many / many-to-one relations, inference time) → figures.

| | PQ |
|---|---|
| Nested-crossfit out-of-fold PQ (707 training images) | 0.4643 |
| Public leaderboard | 0.40 |

## The method

1. **Targets.** There are 2 targets.
   - Main target: soft labels: for each pixel, the *fraction* of annotators who marked it
     (0, 1/3, 1/2, 2/3, 1).
   - Additional target: the same average, but over the annotated filament spines.
   - Images are divided into 5 folds. Folds are grouped by physical image (all records of an image in one fold) and stratified by station and number of annotators.
2. **Inputs.**
   - The solar disk is found separately in each image: a circle is fitted to the disk edge
     (the limb) with RANSAC, starting from the typical GONG disk (centre ≈ (1023, 1024),
     radius ≈ 900 px). If the fit fails (clouds, blank frame), that typical disk is used.
     The disk is used to standardize the two channels below and, in step 6, for the
     "position on the disk" features.
   - Two channels, each standardized over the disk: `raw` intensity and `flat` (intensity minus a
     disk-aware 48 px Gaussian background - done to make illumination more uniform).
   - The model sees the centred 1888×1888 crop, which contains every annotated filament, at full
     resolution: no resizing, no tiling.
3. **Model.**
   - U-Net with an ImageNet ResNet-34 encoder
     ([segmentation_models_pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)),
     2 outputs (filament, spine).
   - Loss: BCE on the soft target + 0.3 × BCE on the spine target.
   - Training: random 768 px crops (70 % centred on a filament pixel), AdamW with warm-up +
     cosine decay, 80 epochs × 200 steps; each fold keeps the epoch with the best validation PQ.
   - Augmentation:
     - each crop is randomly rotated by 0/90/180/270° and maybe flipped (8 possible
       orientations, the "dihedral" group D4);
     - each input channel gets a small random brightness change: multiplied by 0.85–1.15 (gain)
       and shifted by a random amount with std 0.1 (offset).
4. **Ensemble.**
   - 5 folds × 2 seeds = 10 networks.
   - Each seed's test map is the mean over its 5 fold models and over all 8 rotated/flipped
     copies of the image (test-time augmentation; each prediction is rotated back before
     averaging).
   - The final map is the mean of the two seeds' maps.
5. **Instances.**
   - Candidate filaments are the connected components of the map at a threshold `t_low`, after a
     binary closing. They are disjoint by construction, so predicted masks never overlap.
   - `t_low` and the closing radius come from a grid search on the out-of-fold maps (`tune`).
     All post-processing parameters are listed in the table below.
6. **Learned selection.** Each submitted instance costs ½ in the PQ denominator per annotator
   record, so weak candidates must be dropped.
   - A bag of 5 LightGBM regressors predicts each candidate's expected matched IoU from 22
     features of the map and the image: shape, intensity, contrast with its surroundings,
     position on the disk, neighbours, and how the blob grows and merges at lower thresholds.
   - Candidates scoring ≥ τ = 0.23 are kept. τ is not hardcoded: `fit-scorer` tries every
     value from 0.05 to 0.50 (step 0.01) and keeps the one with the best PQ on the out-of-fold
     scores. This matches theory: a candidate is worth keeping when its expected IoU is above
     about PQ / 2 ≈ 0.46 / 2 ≈ 0.23.
   - The scorer and τ are evaluated by nested crossfit on the out-of-fold maps, which gives the
     honest PQ above.

### Post-processing parameters

| Parameter | What it does | Value | Used in the submission? |
|---|---|---|---|
| `t_low` | threshold that cuts the map into candidate blobs | tuned on grid 0.25–0.45 | yes |
| `close_radius` | binary closing before labelling (joins nearby pieces) | tuned on grid {2, 4, 6, 8} px | yes |
| `scorer.min_area` | drops tiny blobs before scoring | 20 px | yes |
| τ | minimum scorer value to keep a candidate | 0.23 (tuned on grid 0.05–0.50, best OOF PQ) | yes |
| `t_high` | blob must contain a pixel with p ≥ `t_high` | tuned | no |
| `min_area` | minimum blob area | tuned | no |
| `min_score` | minimum mean probability of the blob | tuned | no |

Tuned values are written to `post_params.json` (shipped with the released weights).
The last three are only for the simple rule-based selection (`scorer.mode: off`). The
submission uses the learned scorer (step 6) instead, so it ignores them.

## Quick start

Python 3.12 and [uv](https://docs.astral.sh/uv/). A GPU is strongly recommended for inference
(on a CPU each test image takes about 4–5 minutes per seed).

```bash
uv sync --extra train        # or: pip install -r requirements.txt && pip install -e .
make data                    # competition data via the Kaggle CLI (API token + accepted rules)
make weights                 # released checkpoints + scorer -> outputs/ (936 MB, SHA-256 checked)
make infer                   # -> outputs/v4__unet_r34_v2_bce_x2/submission.csv
```

To reproduce everything from scratch instead (about 10 GPU hours per seed), `make train-all`:
train both members (which writes their out-of-fold maps) and their test maps, blend, tune,
fit the scorer, predict. (This command was written but not tested though.)

The weights are the assets of this repository's GitHub release
[`weights-v1`](https://github.com/sergkuzn/kaggle-solar-segmentation-2026-public/releases/tag/weights-v1)
(936 MB). `make weights` and the notebook download them automatically:

- **Network checkpoints**: 10 files, `<member>--folds--fold<k>--model_<epoch>.pt`, about 98 MB each.
  For each of the 2 members and 5 folds, the epoch with the best validation PQ.
- **Member configs**: `<member>--config.yaml`, the resolved training config of each seed.
- **Member OOF metrics**: `<member>--oof_metrics.json`, each seed's out-of-fold PQ / SQ / RQ and
  calibration stats.
- **Blend**: `v4__unet_r34_v2_bce_x2--blend.json`, a record of the blend: its members, their
  calibration temperatures, and how many maps were blended.
- **Post-processing**: `v4__unet_r34_v2_bce_x2--post_params.json`, the tuned `t_low`,
  `close_radius` and the rest.
- **Scorer**: `v4__unet_r34_v2_bce_x2--scorer--model.txt` and `--model_b{1..4}_h0.txt`
  (the 5 LightGBM bags), plus `--scorer--params.json`: the candidate settings (`t_low`, `close_radius`,
  `min_area`), the 22 feature names and τ = 0.23.
- **`manifest.json`**: where each asset goes under `outputs/` and its SHA-256.
  `filseg fetch-weights` follows it and checks every hash.

## Configs

Two networks, three configs. (`v2`/`v4` are just development iteration prefixes.)

| Config | What it is |
|---|---|
| `configs/v2__unet_r34_v2_bce.yaml` | member 1: the U-Net above, seed 42, 5 folds |
| `configs/v4__unet_r34_v2_bce_s43.yaml` | member 2: the same model, seed 43 (the only difference) |
| `configs/v4__unet_r34_v2_bce_x2.yaml` | the submitted pipeline: blend of the two members' maps, tuning grid, LightGBM scorer. It trains nothing. |

## Commands

`uv run filseg <command> --help` documents each one.

| Command | What it does |
|---|---|
| `download`, `extract` | competition zip (Kaggle CLI) → `data/raw/` |
| `prepare` | soft targets, the ground truth in the official format (`gt.csv`), grouped folds (`folds.csv`) |
| `train CONFIG` | fold models (best-validation-PQ checkpoints) and their out-of-fold maps |
| `write-maps CONFIG --split oof\|test\|both` | probability maps from the fold checkpoints, with the config's TTA |
| `blend CONFIG [--split test]` | mean of the members' maps; `--split test` needs no member OOF maps |
| `tune CONFIG` | grid search of the component rule on the OOF maps → `post_params.json` |
| `fit-scorer CONFIG` | learned scorer with nested crossfit → `scorer/` |
| `predict CONFIG` | test maps → `submission.csv`, validated (format, decodable, non-overlapping) |
| `evaluate CSV` | PQ of a submission-format CSV of training predictions |
| `validate CSV` | check a submission CSV (format, decodable, non-overlapping) |
| `viz STEM` | image + annotator contours + predictions → PNG |
| `release-pack CONFIG`, `fetch-weights --repo OWNER/NAME` | pack / download the release assets |

`filseg --set section.key=value <command> ...` overrides a config field for one run.
`FILSEG_DATA_DIR` / `FILSEG_OUTPUT_DIR` relocate `data/` and `outputs/`.



`filseg.competition.metric` reproduces the organizers'
[self-evaluation notebook](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook);
`tests/test_pq.py` checks it against a literal port. `PQResult.pairs()` and
`PQResult.relations()` give the IoU / Dice of the overlapping pairs and the one-to-many /
many-to-one counts the judges look at. Inference uses only the single test JPEG;
no external annotations are used.
