import numpy as np
import pandas as pd
import pytest

from conftest import blob
from filseg.competition import rle
from filseg.competition.metric import evaluate, evaluate_subset


def official_pq(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> float:
    """Literal numpy port of the organizers' self-evaluation notebook (get_pq_score)."""
    gt_ids = gt_df["filament_id"].str.split("_", n=1).str[0]
    pred_ids = pred_df["filament_id"].str.split("_", n=1).str[0]
    tp_scores: list[float] = []
    fp = fn = 0
    for ai in gt_ids.unique():
        image_id = ai.split("-", maxsplit=1)[1]
        g = [
            rle.decode(c).astype(np.float32).ravel()
            for c in gt_df.loc[gt_ids == ai, "segmentation_rle"]
        ]
        p = [
            rle.decode(c).astype(np.float32).ravel()
            for c in pred_df.loc[pred_ids == image_id, "segmentation_rle"]
        ]
        n_gt, n_pred = len(g), len(p)
        if n_gt == 0:
            fp += n_pred
            continue
        if n_pred == 0:
            fn += n_gt
            continue
        G, P = np.stack(g), np.stack(p)
        inter = G @ P.T
        union = G.sum(1)[:, None] + P.sum(1)[None, :] - inter
        iou = np.where(union == 0, 0.0, inter / np.maximum(union, 1))
        hit = iou > 0.5
        tp_scores.extend(iou[hit].tolist())
        fp += int((hit.sum(0) == 0).sum())
        fn += int((hit.sum(1) == 0).sum())
    denom = len(tp_scores) + 0.5 * fp + 0.5 * fn
    return sum(tp_scores) / denom if denom > 0 else 0.0


def _df(rows: list[tuple[str, np.ndarray]]) -> pd.DataFrame:
    return pd.DataFrame(
        [(fid, rle.encode(m)) for fid, m in rows], columns=["filament_id", "segmentation_rle"]
    )


def test_perfect_prediction_scores_one() -> None:
    masks = [blob(500, 500, 60, 20), blob(1200, 900, 30, 80)]
    gt = _df([(f"010101-S1Bh_{i}", m) for i, m in enumerate(masks)])
    pred = _df([(f"S1Bh_{i}", m) for i, m in enumerate(masks)])
    assert evaluate(gt, pred).pq == pytest.approx(1.0)


def test_empty_prediction_scores_zero() -> None:
    gt = _df([("010101-S1Bh_1", blob(500, 500, 60, 20))])
    pred = pd.DataFrame(columns=["filament_id", "segmentation_rle"])
    res = evaluate(gt, pred)
    assert res.pq == 0.0 and res.fn == 1


def test_evaluate_subset_with_no_predictions() -> None:
    # an untrained model (EMA weights at epoch 1) predicts nothing; this used to raise KeyError
    gt = _df([("010101-S1Bh_1", blob(500, 500, 60, 20))])
    pred = pd.DataFrame(columns=["filament_id", "segmentation_rle"])
    res = evaluate_subset(gt, pred, {"S1Bh"})
    assert res.pq == 0.0 and res.fn == 1


def test_matches_official_port_multi_annotator(rng: np.random.Generator) -> None:
    gt_rows, pred_rows = [], []
    for img in range(3):
        stem = f"2015010100000{img}Bh"
        centers = [(int(rng.integers(300, 1700)), int(rng.integers(300, 1700))) for _ in range(5)]
        for ann in range(int(rng.integers(1, 4))):  # 1-3 annotators per image
            for k, (cx, cy) in enumerate(centers):
                if rng.random() < 0.8:  # annotators disagree on which filaments exist
                    jitter = rng.integers(-15, 16, size=2)
                    gt_rows.append(
                        (f"0101{ann:02d}-{stem}_{k}", blob(cx + jitter[0], cy + jitter[1], 50, 18))
                    )
        for k, (cx, cy) in enumerate(centers[:4]):
            pred_rows.append((f"{stem}_{k}", blob(cx + 10, cy - 5, 45, 22)))
        pred_rows.append((f"{stem}_fp", blob(100 + 300 * img, 1900, 20, 20)))
    gt, pred = _df(gt_rows), _df(pred_rows)
    assert evaluate(gt, pred).pq == pytest.approx(official_pq(gt, pred), abs=1e-9)


def test_predictions_for_unlabelled_images_are_ignored() -> None:
    m = blob(500, 500, 60, 20)
    gt = _df([("010101-S1Bh_1", m)])
    pred = _df([("S1Bh_1", m), ("OTHERBh_1", m)])
    assert evaluate(gt, pred).pq == pytest.approx(1.0)


def _rect(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    m = np.zeros((2048, 2048), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_pairs_and_relations() -> None:
    """A split into 3 pieces (1:3), B + C merged (2:1), D missed, E exact, one unannotated pred."""
    gt = _df(
        [
            ("010101-S1Bh_0", _rect(100, 120, 100, 190)),  # A
            ("010101-S1Bh_1", _rect(300, 320, 100, 150)),  # B
            ("010101-S1Bh_2", _rect(300, 320, 160, 210)),  # C
            ("010101-S1Bh_3", _rect(500, 520, 500, 520)),  # D
            ("010101-S1Bh_4", _rect(700, 720, 700, 760)),  # E
        ]
    )
    pred = _df(
        [
            ("S1Bh_0", _rect(100, 120, 100, 130)),
            ("S1Bh_1", _rect(100, 120, 130, 160)),
            ("S1Bh_2", _rect(100, 120, 160, 190)),
            ("S1Bh_3", _rect(300, 320, 100, 210)),
            ("S1Bh_4", _rect(900, 920, 900, 920)),
            ("S1Bh_5", _rect(700, 720, 700, 760)),
        ]
    )
    res = evaluate(gt, pred)
    rel = res.relations()
    assert rel["gt_filaments"].tolist() == [1, 3, 0, 1]  # D; B, C, E; -; A
    assert rel["predictions"].tolist() == [1, 4, 1, 0]  # S1Bh_4; the A pieces and E; the merge
    pairs = res.pairs()
    assert len(pairs) == 6 and pairs["tp"].sum() == res.tp == 1
    merge_b = pairs[(pairs.gt_idx == 1) & (pairs.pred_idx == 3)].iloc[0]
    assert merge_b.iou == pytest.approx(1000 / 2200)
    assert merge_b.dice == pytest.approx(2 * 1000 / (1000 + 2200))


def test_pairs_and_relations_without_predictions() -> None:
    gt = _df([("010101-S1Bh_1", blob(500, 500, 60, 20))])
    res = evaluate(gt, pd.DataFrame(columns=["filament_id", "segmentation_rle"]))
    assert res.pairs().empty
    assert res.relations().to_dict("list") == {"gt_filaments": [1], "predictions": [0]}
