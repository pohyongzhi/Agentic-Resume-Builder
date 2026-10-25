"""Configuration loading and path safety for V4 training."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/v4/qwen35-4b-lora.json"
ALLOWED_OUTPUT_ROOTS = tuple((ROOT / name).resolve() for name in (
    "runs", "artifacts", "models", "checkpoints"
))


class TrainingConfigError(ValueError):
    """Raised when a training invariant is not satisfied."""


def load_config(path=DEFAULT_CONFIG):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    value = json.loads(path.read_text(encoding="utf-8"))
    model = value.get("model", {})
    if model.get("id") != "Qwen/Qwen3.5-4B":
        raise TrainingConfigError("Training must use official Qwen/Qwen3.5-4B")
    revision = model.get("revision", "")
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
        raise TrainingConfigError("Model revision must be an exact lowercase 40-character commit SHA")
    if value.get("sequence", {}).get("max_length") != 4096:
        raise TrainingConfigError("The V4 pilot sequence budget must be 4096 tokens")
    if value.get("sequence", {}).get("packing") is not False:
        raise TrainingConfigError("Packing must remain disabled for the initial pilot")
    return value


def resolve_repo_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def require_safe_output_dir(value):
    path = Path(value).expanduser().resolve()
    if path == ROOT:
        raise TrainingConfigError("Output directory cannot be the repository root")
    if path.is_relative_to(ROOT) and not any(
            path == allowed or path.is_relative_to(allowed) for allowed in ALLOWED_OUTPUT_ROOTS):
        raise TrainingConfigError(
            "Repository-local outputs must be under runs/, artifacts/, models/, or checkpoints/")
    return path
