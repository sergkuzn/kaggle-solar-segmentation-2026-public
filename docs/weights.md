# Trained weights

The trained weights are the assets of this repository's GitHub release
[`weights-v1`](https://github.com/sergkuzn/kaggle-solar-segmentation-2026-public/releases/tag/weights-v1)
(936 MB). `make weights` and the [notebook](../notebooks/filseg_pipeline.ipynb) download them
automatically and check every file's SHA-256. They are exactly the weights behind the submitted
results.

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
