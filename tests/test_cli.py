"""
Tests for CLI argument parsing.
"""

from llama_cluster.cli import main
import pytest


def test_cli_version_flag(capsys):
    """Verify --version argument outputs program version."""
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "aeromesh" in captured.out or "0.1.0" in captured.out


def test_cli_help(capsys):
    """Verify --help flag prints usage info."""
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "AeroMesh" in captured.out or "llama-cluster" in captured.out
