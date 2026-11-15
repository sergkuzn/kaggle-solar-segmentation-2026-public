"""CPU smoke tests of the torch stack (skipped when the [train] extra is not installed)."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("segmentation_models_pytorch")

from filseg.config import TTA, ExperimentConfig, ModelConfig  # noqa: E402
from filseg.infra.paths import Paths  # noqa: E402
from filseg.model.arch import build_model  # noqa: E402
from filseg.model.checkpoint import save_checkpoint  # noqa: E402
from filseg.model.predictor import Predictor  # noqa: E402
from filseg.preprocess import Preprocessor  # noqa: E402
from filseg.training.losses import compute_loss  # noqa: E402

pytestmark = pytest.mark.train


def _cfg(tta: TTA) -> ExperimentConfig:
    cfg = ExperimentConfig(
        name="smoke", model=ModelConfig(encoder="resnet18", encoder_weights=None)
    )
    cfg.data.crop = 256
    cfg.infer.tta = tta
    return cfg


def test_forward_and_loss() -> None:
    cfg = _cfg(TTA.NONE)
    model = build_model(cfg.model, in_channels=2, pretrained=False)
    x = torch.randn(2, 2, 128, 128)
    y = torch.rand(2, 2, 128, 128)
    out = model(x)
    assert out.shape == (2, 2, 128, 128)
    loss, parts = compute_loss(out, y, cfg)
    loss.backward()
    assert set(parts) == {"bce", "spine"}


@pytest.mark.parametrize("tta", list(TTA))
def test_predictor_shapes(tta: TTA) -> None:
    cfg = _cfg(tta)
    model = build_model(cfg.model, in_channels=2, pretrained=False).eval()
    img = np.full((2048, 2048), 120, np.uint8)
    pre = Preprocessor.from_config(cfg.data)
    prob = Predictor([model], pre, torch.device("cpu"), tta=tta).prob(img)
    assert prob.shape == (2048, 2048)
    assert prob.min() >= 0 and prob.max() <= 1
    assert prob[:10, :10].max() == 0, "outside the crop stays zero"


def test_predictor_from_experiment(tmp_path: Path) -> None:
    cfg = _cfg(TTA.NONE)
    paths = Paths(data_root=tmp_path / "data", output_root=tmp_path / "outputs")
    model = build_model(cfg.model, in_channels=2, pretrained=False).eval()
    save_checkpoint(model, paths.experiment(cfg.name).checkpoint(0, 1, last=True), meta={})
    img = np.full((2048, 2048), 120, np.uint8)
    predictor = Predictor.from_experiment(cfg, torch.device("cpu"), [0], paths)
    direct = Predictor([model], Preprocessor.from_config(cfg.data), torch.device("cpu"))
    np.testing.assert_allclose(predictor.prob(img), direct.prob(img))


def test_write_prob_maps(tmp_path: Path) -> None:
    import cv2

    from filseg.imaging import read_prob
    from filseg.model.predictor import write_prob_maps

    cfg = _cfg(TTA.FLIP)
    models = [build_model(cfg.model, in_channels=2, pretrained=False).eval() for _ in range(2)]
    pred = Predictor(models, Preprocessor.from_config(cfg.data), torch.device("cpu"), tta=TTA.FLIP)
    cv2.imwrite(str(tmp_path / "img.jpeg"), np.full((2048, 2048), 120, np.uint8))
    write_prob_maps(pred, tmp_path, ["img"], tmp_path / "maps")
    saved = read_prob(tmp_path / "maps" / "img.png")
    np.testing.assert_allclose(saved, pred.prob_file(tmp_path / "img.jpeg"), atol=0.5 / 255 + 1e-6)
