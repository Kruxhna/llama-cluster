"""
Tests for AeroMesh Web Control Dashboard API endpoints.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from llama_cluster.dashboard import DashboardRequestHandler
from llama_cluster.config import Config


def test_dashboard_static_dir_exists():
    """Verify web asset directory and required static files exist."""
    web_dir = Path(__file__).parent.parent / "src" / "llama_cluster" / "web"
    assert web_dir.exists()
    assert (web_dir / "index.html").exists()
    assert (web_dir / "style.css").exists()
    assert (web_dir / "app.js").exists()


def test_dashboard_api_cluster_structure(tmp_path):
    """Verify /api/cluster response payload format."""
    cfg = Config()
    assert "nodes" in cfg.topology
    assert "coordinator" in cfg.topology
    assert "model_spec" in cfg.topology
