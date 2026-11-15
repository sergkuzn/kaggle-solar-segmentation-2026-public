# Usage

## Setup and inference

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

## Configs

Two networks, three configs. (`v2`/`v4` are just development iteration prefixes.)

| Config | What it is |
|---|---|
| `configs/v2__unet_r34_v2_bce.yaml` | member 1: the U-Net, seed 42, 5 folds |
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
