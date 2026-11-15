# Final pipeline: two seeds of the same U-Net (MEMBERS), blended, then the learned scorer.
MEMBERS = v2__unet_r34_v2_bce v4__unet_r34_v2_bce_s43
FINAL = v4__unet_r34_v2_bce_x2
REPO ?= sergkuzn/kaggle-solar-segmentation-2026-public
FILSEG = uv run filseg

.PHONY: setup setup-cpu check lint test data weights infer train-all release-weights requirements

setup:          ## full env incl. torch
	uv sync --extra train

setup-cpu:      ## light env: data prep, post-processing, metric
	uv sync

check: lint test

lint:
	uv run ruff format --check src tests
	uv run ruff check src tests
	uv run mypy src

test:
	uv run pytest

data:           ## competition zip (Kaggle CLI auth) -> data/raw, targets, GT table, folds
	$(FILSEG) download
	$(FILSEG) extract
	$(FILSEG) prepare

weights:        ## released checkpoints + scorer -> outputs/
	$(FILSEG) fetch-weights --repo $(REPO)

infer:          ## test maps of both members -> blend -> outputs/$(FINAL)/submission.csv
	for m in $(MEMBERS); do $(FILSEG) write-maps configs/$$m.yaml --split test || exit 1; done
	$(FILSEG) blend configs/$(FINAL).yaml --split test
	$(FILSEG) predict configs/$(FINAL).yaml

train-all:      ## everything from scratch (~20 GPU hours): train members, blend, tune, scorer, predict
	for m in $(MEMBERS); do $(FILSEG) train configs/$$m.yaml && $(FILSEG) write-maps configs/$$m.yaml --split test || exit 1; done
	$(FILSEG) blend configs/$(FINAL).yaml
	$(FILSEG) tune configs/$(FINAL).yaml
	$(FILSEG) fit-scorer configs/$(FINAL).yaml
	$(FILSEG) predict configs/$(FINAL).yaml

release-weights: ## dist/weights/: checkpoints + scorer + manifest.json of $(FINAL), for a GitHub release
	$(FILSEG) release-pack configs/$(FINAL).yaml --out dist/weights

requirements:   ## pinned requirements.txt
	uv export --frozen --no-hashes --no-dev --extra train --no-emit-project -o requirements.txt
