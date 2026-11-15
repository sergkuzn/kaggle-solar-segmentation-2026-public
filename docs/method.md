# Method

The final submission `v4__unet_r34_v2_bce_x2`, step by step. The
[notebook](../notebooks/filseg_pipeline.ipynb) runs the same steps with their outputs.

1. **Targets.** There are 2 targets.
   - Main target: soft labels: for each pixel, the *fraction* of annotators who marked it
     (0, 1/3, 1/2, 2/3, 1).
   - Additional target: the same average, but over the annotated filament spines.
   - Images are divided into 5 folds. Folds are grouped by physical image (all records of an
     image in one fold) and stratified by station and number of annotators.
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
     honest out-of-fold PQ of 0.4643.

## Post-processing parameters

| Parameter | What it does | Value | Used in the submission? |
|---|---|---|---|
| `t_low` | threshold that cuts the map into candidate blobs | tuned on grid 0.25–0.45 | yes |
| `close_radius` | binary closing before labelling (joins nearby pieces) | tuned on grid {2, 4, 6, 8} px | yes |
| `scorer.min_area` | drops tiny blobs before scoring | 20 px | yes |
| τ | minimum scorer value to keep a candidate | 0.23 (tuned on grid 0.05–0.50, best OOF PQ) | yes |
| `t_high` | blob must contain a pixel with p ≥ `t_high` | tuned | no |
| `min_area` | minimum blob area | tuned | no |
| `min_score` | minimum mean probability of the blob | tuned | no |

Tuned values are written to `post_params.json` (shipped with the [released weights](weights.md)).
The last three are only for the simple rule-based selection (`scorer.mode: off`). The
submission uses the learned scorer (step 6) instead, so it ignores them.

## Metric

`filseg.competition.metric` reproduces the organizers'
[self-evaluation notebook](https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook).
`PQResult.pairs()` and `PQResult.relations()` give the IoU / Dice of the overlapping pairs and
the one-to-many / many-to-one counts the judges look at. Inference uses only the single test
JPEG; no external annotations are used.
