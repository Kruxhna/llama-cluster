"""
Tests for configuration parsing and relative path resolution.
"""

from pathlib import Path
import pytest
from llama_cluster.config import Config, get_config, REPO_ROOT


def test_repo_root_resolution():
    """Verify REPO_ROOT points to valid repository directory."""
    assert REPO_ROOT.exists()
    assert (REPO_ROOT / "pyproject.toml").exists()


def test_config_defaults():
    """Verify default configuration attributes."""
    cfg = Config()
    assert isinstance(cfg.host, str)
    assert isinstance(cfg.port, int)
    assert isinstance(cfg.model_dir, Path)
    assert isinstance(cfg.llama_cpp_dir, Path)
    assert cfg.model_dir.is_absolute()
    assert cfg.llama_cpp_dir.is_absolute()


def test_get_model_path():
    """Verify GGUF model path calculation."""
    cfg = Config()
    path = cfg.get_model_path("test_model.gguf")
    assert path == cfg.model_dir / "test_model.gguf"
