"""Tests for lazy, version-pinned model loading. No real backend is touched."""

from __future__ import annotations

import hashlib
import sys
import types
from pathlib import Path

import pytest

from tracking import models
from tracking.config import ModelConfig


def _fake_backend(created: list, payload: bytes | None = None):
    """Builds a stand-in for the ultralytics module.

    Its `YOLO` records the path it was given and, like the real one,
    "downloads" the weights to that path when they are missing.
    """

    class FakeYolo:
        def __init__(self, path):
            created.append(path)
            if payload is not None and not Path(path).is_file():
                Path(path).write_bytes(payload)

    return types.SimpleNamespace(YOLO=FakeYolo, __version__="9.9.9")


def _register(monkeypatch, payload: bytes) -> ModelConfig:
    """Registers a fake person model pinned to `payload`'s checksum."""
    spec = models.ModelSpec(
        name="fake-person",
        task="person",
        backend="fake_backend",
        weights="fake.pt",
        release="test",
        licence="none",
        sha256=hashlib.sha256(payload).hexdigest(),
    )
    monkeypatch.setitem(models.REGISTRY, spec.name, spec)
    return ModelConfig(person_model=spec.name)


def test_importing_the_tracker_loads_no_ml_library():
    import tracking  # noqa: F401
    import tracking.ball  # noqa: F401
    import tracking.players  # noqa: F401

    assert "torch" not in sys.modules
    assert "ultralytics" not in sys.modules


def test_the_registry_pins_the_person_model_and_its_licence():
    spec = models.model_spec("yolo11n")

    assert spec is models.PERSON_YOLO
    assert spec.weights == "yolo11n.pt"
    assert spec.licence == "AGPL-3.0"
    assert len(spec.sha256) == 64


def test_an_unknown_model_name_lists_the_known_ones():
    with pytest.raises(models.ModelError, match="yolo11n"):
        models.model_spec("yolo99")


def test_a_missing_backend_is_a_typed_error_with_install_advice():
    with pytest.raises(models.BackendMissingError, match="pip install"):
        models.import_backend("a_backend_that_is_not_installed")


def test_requesting_yolo_without_ultralytics_raises_only_then(monkeypatch, tmp_path):
    # None in sys.modules makes `import ultralytics` raise ImportError.
    monkeypatch.setitem(sys.modules, "ultralytics", None)

    with pytest.raises(models.BackendMissingError, match="ultralytics"):
        models.load_person_model(ModelConfig(weights_dir=tmp_path))


@pytest.mark.parametrize(
    ("platform", "environment", "expected"),
    [
        ("win32", {"LOCALAPPDATA": "C:/Local"}, Path("C:/Local/rally-ai/models")),
        ("linux", {"XDG_CACHE_HOME": "/xdg"}, Path("/xdg/rally-ai/models")),
        ("linux", {}, Path.home() / ".cache" / "rally-ai" / "models"),
    ],
)
def test_weights_default_to_a_per_user_folder(monkeypatch, platform, environment, expected):
    monkeypatch.setattr(sys, "platform", platform)
    for name in ("LOCALAPPDATA", "XDG_CACHE_HOME"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    assert models.default_weights_dir() == expected


def test_the_default_folder_is_outside_the_repository():
    repo = Path(__file__).resolve().parents[2]

    assert repo not in models.default_weights_dir().resolve().parents


def test_loading_downloads_to_an_absolute_path_and_names_the_version(monkeypatch, tmp_path):
    created: list = []
    config = _register(monkeypatch, b"weights")
    monkeypatch.setattr(models, "import_backend", lambda name: _fake_backend(created, b"weights"))
    monkeypatch.setattr(models, "default_weights_dir", lambda: tmp_path / "cache")

    loaded = models.load_person_model(config)

    assert Path(created[0]).is_absolute()
    assert Path(created[0]) == (tmp_path / "cache" / "fake.pt").resolve()
    assert loaded.source == "fake-person@fake_backend-9.9.9"
    assert loaded.weights_path == Path(created[0])
    assert loaded.spec.name == "fake-person"


def test_a_downloaded_file_that_is_not_the_pinned_one_is_refused(monkeypatch, tmp_path):
    config = _register(monkeypatch, b"weights")
    monkeypatch.setattr(models, "import_backend", lambda name: _fake_backend([], b"tampered"))

    with pytest.raises(models.WeightsChecksumError, match="fake.pt"):
        models.load_person_model(
            ModelConfig(person_model=config.person_model, weights_dir=tmp_path)
        )


def test_an_existing_wrong_file_is_refused_before_it_is_ever_loaded(monkeypatch, tmp_path):
    created: list = []
    config = _register(monkeypatch, b"weights")
    (tmp_path / "fake.pt").write_bytes(b"something else")
    monkeypatch.setattr(models, "import_backend", lambda name: _fake_backend(created))

    with pytest.raises(models.WeightsChecksumError):
        models.load_person_model(
            ModelConfig(person_model=config.person_model, weights_dir=tmp_path)
        )
    assert created == []


def test_an_existing_pinned_file_loads(monkeypatch, tmp_path):
    created: list = []
    config = _register(monkeypatch, b"weights")
    (tmp_path / "fake.pt").write_bytes(b"weights")
    monkeypatch.setattr(models, "import_backend", lambda name: _fake_backend(created))

    loaded = models.load_person_model(
        ModelConfig(person_model=config.person_model, weights_dir=tmp_path)
    )

    assert len(created) == 1
    assert loaded.model is not None


def test_checksum_enforcement_can_be_turned_off(monkeypatch, tmp_path):
    config = _register(monkeypatch, b"weights")
    monkeypatch.setattr(models, "import_backend", lambda name: _fake_backend([], b"tampered"))

    loaded = models.load_person_model(
        ModelConfig(
            person_model=config.person_model, weights_dir=tmp_path, verify_checksum=False
        )
    )

    assert loaded.weights_path.read_bytes() == b"tampered"


def test_a_model_that_is_not_a_person_detector_is_refused(tmp_path):
    with pytest.raises(models.ModelError, match="not a person detector"):
        models.load_person_model(
            ModelConfig(person_model="ball-heatmap-v0", weights_dir=tmp_path)
        )


def test_the_learned_ball_model_says_it_has_no_trained_weights():
    with pytest.raises(models.WeightsMissingError, match="No trained weights"):
        models.load_ball_heatmap()
    assert models.BALL_HEATMAP.weights is None


def test_weights_checksum_is_sha256(tmp_path):
    path = tmp_path / "w.bin"
    path.write_bytes(b"abc")

    assert models.weights_checksum(path) == hashlib.sha256(b"abc").hexdigest()


class _FakeTensor:
    """Just enough of a torch tensor for the adapter."""

    def __init__(self, array):
        self.array = array

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


def _fake_torch(loaded: list):
    """A stand-in for torch: `jit.load` records its path; the module halves the middle frame."""

    class Module:
        def __init__(self):
            self.evaluated = False

        def eval(self):
            self.evaluated = True

        def __call__(self, tensor):
            return _FakeTensor(tensor.array[1, :, :, 0] / 510.0)

    class NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    def load(path, map_location=None):
        loaded.append((path, map_location))
        module = Module()
        loaded.append(module)
        return module

    return types.SimpleNamespace(
        jit=types.SimpleNamespace(load=load),
        no_grad=NoGrad,
        from_numpy=_FakeTensor,
    )


def _ball_spec(monkeypatch, payload: bytes, *, sha=True, weights="ball.pt", task="ball_heatmap"):
    """Registers a trained-looking ball model and returns its config's name."""
    spec = models.ModelSpec(
        name="ball-test",
        task=task,
        backend="torch",
        weights=weights,
        release="first-party 2026-12-01",
        licence="first-party",
        sha256=hashlib.sha256(payload).hexdigest() if sha else None,
    )
    monkeypatch.setitem(models.REGISTRY, spec.name, spec)
    return spec.name


def test_a_pinned_trained_ball_model_loads_on_cpu_and_predicts(monkeypatch, tmp_path):
    import numpy as np

    name = _ball_spec(monkeypatch, b"scripted")
    (tmp_path / "ball.pt").write_bytes(b"scripted")
    loaded = []
    monkeypatch.setattr(models, "import_backend", lambda backend: _fake_torch(loaded))
    model = models.load_ball_heatmap(ModelConfig(ball_model=name, weights_dir=tmp_path))
    assert model.source == "ball-test@first-party 2026-12-01"
    assert loaded[0] == (str((tmp_path / "ball.pt").resolve()), "cpu") and loaded[1].evaluated
    frames = np.full((3, 4, 5, 3), 255, dtype=np.uint8)
    heat = model.model.predict(frames)
    assert heat.shape == (4, 5) and np.allclose(heat, 0.5)


def test_a_ball_model_file_that_is_missing_or_not_pinned_is_refused(monkeypatch, tmp_path):
    name = _ball_spec(monkeypatch, b"scripted")
    config = ModelConfig(ball_model=name, weights_dir=tmp_path)
    with pytest.raises(models.WeightsMissingError, match="copy the exported file there"):
        models.load_ball_heatmap(config)
    (tmp_path / "ball.pt").write_bytes(b"tampered")
    with pytest.raises(models.WeightsChecksumError):
        models.load_ball_heatmap(config)
    name = _ball_spec(monkeypatch, b"tampered", sha=False)
    with pytest.raises(models.WeightsChecksumError, match="pins no SHA-256"):
        models.load_ball_heatmap(config)
    loaded = []
    monkeypatch.setattr(models, "import_backend", lambda backend: _fake_torch(loaded))
    unchecked = ModelConfig(ball_model=name, weights_dir=tmp_path, verify_checksum=False)
    assert models.load_ball_heatmap(unchecked).weights_path.name == "ball.pt"


def test_a_ball_model_entry_must_be_a_ball_model(monkeypatch, tmp_path):
    with pytest.raises(models.ModelError, match="is not a ball model"):
        models.load_ball_heatmap(ModelConfig(ball_model="yolo11n", weights_dir=tmp_path))


def test_ball_weights_are_looked_for_in_the_per_user_folder_by_default(monkeypatch, tmp_path):
    name = _ball_spec(monkeypatch, b"scripted")
    monkeypatch.setattr(models, "default_weights_dir", lambda: tmp_path / "cache")
    with pytest.raises(models.WeightsMissingError, match="cache"):
        models.load_ball_heatmap(ModelConfig(ball_model=name))
