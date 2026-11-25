"""Learned models: which ones, which version, and loading them lazily.

Nothing here imports a machine-learning library at module import time.
`import tracking` - and the whole existing app - works on a machine with no
torch and no ultralytics; a missing backend is reported, as a
`BackendMissingError`, only at the moment that backend is asked for.

**Version pinning.** Every model the tracker can load is an entry in
`REGISTRY`: a `ModelSpec` naming the weights file, the release it belongs
to, its licence, and (once known) the SHA-256 of the file. A loaded model
comes back as a `LoadedModel` whose `source` string - e.g.
`"yolo11n@ultralytics-8.3.40"` - is what detections carry as their
provenance `source`, so a result can be traced to the exact model and
library that produced it.

**Weights are never in the repository.** They are downloaded to, and read
from, a per-user cache folder (`default_weights_dir`) or the folder
`ModelConfig.weights_dir` names. The file is always addressed by absolute
path, so the backend never drops a `.pt` in the working directory.

Licence note: Ultralytics YOLO is AGPL-3.0. That is acceptable here because
this project is open and non-commercial (spec section 12, decision 1); it
would not be for a closed distribution.
"""

from __future__ import annotations

import hashlib
import importlib
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import ModelConfig

logger = logging.getLogger(__name__)

__all__ = [
    "BALL_HEATMAP",
    "PERSON_YOLO",
    "POSE_YOLO",
    "REGISTRY",
    "BackendMissingError",
    "LoadedModel",
    "TorchScriptHeatmap",
    "ModelError",
    "ModelSpec",
    "WeightsChecksumError",
    "WeightsMissingError",
    "default_weights_dir",
    "import_backend",
    "load_ball_heatmap",
    "load_person_model",
    "load_pose_model",
    "model_spec",
    "weights_checksum",
]


class ModelError(RuntimeError):
    """A learned model could not be provided."""


class BackendMissingError(ModelError):
    """The library a model needs is not installed."""


class WeightsMissingError(ModelError):
    """There are no trained weights for the model asked for."""


class WeightsChecksumError(ModelError):
    """A weights file is not the pinned one."""


@dataclass(frozen=True)
class ModelSpec:
    """One pinned model.

    Attributes:
        name: The registry key, e.g. "yolo11n".
        task: What it is for: "person", "pose" or "ball_heatmap".
        backend: The Python package that runs it.
        weights: The weights file's name, or None when none exist yet.
        release: The upstream release the weights belong to.
        licence: The licence the code and weights are under.
        sha256: The weights file's SHA-256, or None when not pinned.
    """

    name: str
    task: str
    backend: str
    weights: str | None
    release: str
    licence: str
    sha256: str | None = None


PERSON_YOLO = ModelSpec(
    name="yolo11n",
    task="person",
    backend="ultralytics",
    weights="yolo11n.pt",
    release="ultralytics assets v8.4.0",
    licence="AGPL-3.0",
    sha256="0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
)

BALL_HEATMAP = ModelSpec(
    name="ball-heatmap-v0",
    task="ball_heatmap",
    backend="torch",
    weights=None,
    release="untrained",
    licence="first-party",
)

POSE_YOLO = ModelSpec(
    name="yolo11n-pose",
    task="pose",
    backend="ultralytics",
    weights="yolo11n-pose.pt",
    release="ultralytics assets v8.4.0",
    licence="AGPL-3.0",
    sha256="869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0",
)

REGISTRY: dict[str, ModelSpec] = {
    spec.name: spec for spec in (PERSON_YOLO, BALL_HEATMAP, POSE_YOLO)
}


@dataclass(frozen=True)
class LoadedModel:
    """A model ready to run, with where it came from.

    Attributes:
        spec: The pinned entry it was loaded from.
        model: The backend's own model object.
        source: "name@backend-version", for provenance.
        weights_path: The file it was loaded from.
    """

    spec: ModelSpec
    model: Any
    source: str
    weights_path: Path | None


def model_spec(name: str) -> ModelSpec:
    """Looks a model up in the registry.

    Args:
        name: The registry key.

    Returns:
        Its pinned entry.

    Raises:
        ModelError: If no such model is registered.
    """
    try:
        return REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(REGISTRY))
        raise ModelError(f"Unknown model {name!r}; registered: {known}.") from None


def default_weights_dir() -> Path:
    """Names the per-user folder weights are kept in, outside any repository.

    Returns:
        `%LOCALAPPDATA%/rally-ai/models` on Windows, otherwise
        `$XDG_CACHE_HOME/rally-ai/models` or `~/.cache/rally-ai/models`.
        Not created here.
    """
    if sys.platform == "win32" and os.environ.get("LOCALAPPDATA"):
        base = Path(os.environ["LOCALAPPDATA"])
    elif os.environ.get("XDG_CACHE_HOME"):
        base = Path(os.environ["XDG_CACHE_HOME"])
    else:
        base = Path.home() / ".cache"
    return base / "rally-ai" / "models"


def import_backend(name: str) -> Any:
    """Imports a model backend at the moment it is needed.

    Args:
        name: The package, e.g. "ultralytics".

    Returns:
        The imported module.

    Raises:
        BackendMissingError: If it is not installed (or fails to import),
            saying how to install it.
    """
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise BackendMissingError(
            f"The {name!r} package is needed for this model but could not be imported "
            f"({error}). Install it with `pip install -r requirements.txt`, or use the "
            "classic detector, which needs no learned model."
        ) from error


def weights_checksum(path: Path | str) -> str:
    """Hashes a weights file.

    Args:
        path: The file.

    Returns:
        Its SHA-256 as lowercase hex.
    """
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_person_model(config: ModelConfig | None = None) -> LoadedModel:
    """Loads the pinned person detector, downloading its weights if absent.

    The weights live in `config.weights_dir` (default: the per-user cache
    folder), addressed by absolute path so nothing lands in the working
    directory.

    Args:
        config: Which model, where its weights are kept, and whether the
            file's checksum is enforced.

    Returns:
        The loaded model with its provenance string.

    Raises:
        ModelError: If the configured model is not a person detector.
        BackendMissingError: If ultralytics (or torch under it) is not
            installed.
        WeightsChecksumError: If the weights file is not the pinned one.
    """
    config = config or ModelConfig()
    spec = model_spec(config.person_model)
    if spec.task != "person" or spec.weights is None:
        raise ModelError(f"{spec.name!r} is not a person detector.")
    return _load_yolo(spec, config)


def load_pose_model(config: ModelConfig | None = None) -> LoadedModel:
    """Loads the pinned pose model, downloading its weights if absent.

    Same rules as `load_person_model`: per-user folder, absolute path,
    checksum enforced.

    Args:
        config: Which model, where its weights are kept, and whether the
            file's checksum is enforced.

    Returns:
        The loaded model with its provenance string.

    Raises:
        ModelError: If the configured model is not a pose model.
        BackendMissingError: If ultralytics (or torch under it) is not
            installed.
        WeightsChecksumError: If the weights file is not the pinned one.
    """
    config = config or ModelConfig()
    spec = model_spec(config.pose_model)
    if spec.task != "pose" or spec.weights is None:
        raise ModelError(f"{spec.name!r} is not a pose model.")
    return _load_yolo(spec, config)


def _load_yolo(spec: ModelSpec, config: ModelConfig) -> LoadedModel:
    """Loads one pinned Ultralytics model from the weights folder."""
    backend = import_backend(spec.backend)
    folder = Path(config.weights_dir) if config.weights_dir else default_weights_dir()
    folder.mkdir(parents=True, exist_ok=True)
    path = (folder / spec.weights).resolve()
    check = config.verify_checksum and spec.sha256 is not None
    present = path.is_file()
    if check and present:
        _verify(path, spec)
    # Given an absolute path that does not exist yet, the backend downloads
    # the pinned asset to exactly that path - never the working directory.
    model = backend.YOLO(str(path))
    if check and not present and path.is_file():
        _verify(path, spec)
    version = getattr(backend, "__version__", "unknown")
    source = f"{spec.name}@{spec.backend}-{version}"
    logger.info("Loaded %s model %s from %s", spec.task, source, path)
    return LoadedModel(spec=spec, model=model, source=source, weights_path=path)


def _verify(path: Path, spec: ModelSpec) -> None:
    """Raises `WeightsChecksumError` unless the file is the pinned one."""
    found = weights_checksum(path)
    if found != spec.sha256:
        raise WeightsChecksumError(
            f"{path.name}: checksum {found[:12]}... is not the pinned "
            f"{str(spec.sha256)[:12]}... for {spec.name} ({spec.release}). Delete the file to "
            "download it again, or turn off ModelConfig.verify_checksum."
        )


class TorchScriptHeatmap:
    """Adapts an exported TorchScript ball model to `ball.HeatmapModel`.

    The exported module takes the three frames exactly as the tracker
    holds them - a (3, H, W, 3) uint8 tensor - and returns an (H, W)
    heatmap in 0..1. Resizing and normalising live *inside* the exported
    module (docs/ball-model-training.md), so they are pinned by the same
    SHA-256 as the weights and cannot drift from what was trained.

    Attributes:
        module: The loaded TorchScript module.
    """

    def __init__(self, module: Any, torch: Any) -> None:
        """Wraps a loaded module.

        Args:
            module: The TorchScript module, in eval mode.
            torch: The torch package.
        """
        self.module = module
        self._torch = torch

    def predict(self, frames: Any) -> Any:
        """Scores every pixel of the middle frame.

        Args:
            frames: (3, H, W, 3) uint8 RGB: earlier, judged, later.

        Returns:
            (H, W) float heatmap.
        """
        import numpy as np

        with self._torch.no_grad():
            heat = self.module(self._torch.from_numpy(np.ascontiguousarray(frames)))
        return np.asarray(heat.detach().cpu().numpy(), dtype=float)


def load_ball_heatmap(config: ModelConfig | None = None) -> LoadedModel:
    """Loads the learned ball detector (Track B) from pinned, local weights.

    Weights are never downloaded and never committed: a trained file is
    copied into the weights folder by hand, and its registry entry must
    name the file and pin its SHA-256 (docs/ball-model-training.md, "Pin
    the weights"). Until then this raises.

    Args:
        config: Which ball model (`ModelConfig.ball_model`), where weights
            are kept, and whether the checksum is enforced.

    Returns:
        The loaded model, wrapped as a `ball.HeatmapModel`, with its
        provenance string ("name@release").

    Raises:
        ModelError: If the configured entry is not a ball model.
        WeightsMissingError: If the entry names no trained weights, or the
            file is not in the weights folder.
        WeightsChecksumError: If the file is not the pinned one, or the
            entry pins no checksum while checksums are enforced.
        BackendMissingError: If torch is not installed.
    """
    config = config or ModelConfig()
    spec = model_spec(config.ball_model)
    if spec.task != "ball_heatmap":
        raise ModelError(f"{spec.name!r} is not a ball model.")
    if spec.weights is None:
        raise WeightsMissingError(
            f"No trained weights exist for {spec.name!r}. The learned ball detector (Track B) "
            "has not been trained; use the classic `ball.MotionBallDetector`."
        )
    folder = Path(config.weights_dir) if config.weights_dir else default_weights_dir()
    path = (folder / spec.weights).resolve()
    if not path.is_file():
        raise WeightsMissingError(
            f"{spec.name!r} expects its trained weights at {path}; copy the exported file there "
            "(weights are never downloaded automatically and never committed)."
        )
    if config.verify_checksum:
        if spec.sha256 is None:
            raise WeightsChecksumError(
                f"{spec.name!r} pins no SHA-256, so {path.name} cannot be verified. Add its "
                "checksum (tracking.models.weights_checksum) to the registry entry."
            )
        _verify(path, spec)
    torch = import_backend(spec.backend)
    module = torch.jit.load(str(path), map_location="cpu")
    module.eval()
    source = f"{spec.name}@{spec.release}"
    logger.info("Loaded ball model %s from %s", source, path)
    return LoadedModel(
        spec=spec, model=TorchScriptHeatmap(module, torch), source=source, weights_path=path
    )
