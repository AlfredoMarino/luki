from pathlib import Path


def repo_root() -> Path:
    """Walk up from this file until pyproject.toml is found."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Could not locate repo root (no pyproject.toml found)")


def config_path(relative: str = "config/base.yaml") -> Path:
    return repo_root() / relative
