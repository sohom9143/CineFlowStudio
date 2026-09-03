import json
import os
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from app import CineFlowApp, build_fastapi_app
from desktop_app import get_saved_colab_url, verify_colab_url


@pytest.fixture
def test_app():
    config_path = "configs/colab_t4_config.yaml"
    app_instance = CineFlowApp(config_path=config_path)
    fastapi_app = build_fastapi_app(app_instance)
    return TestClient(fastapi_app)


def test_api_health_endpoint(test_app):
    """Verifies that /api/health returns 200 OK and healthy status structure."""
    response = test_app.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "device" in data
    assert "is_cuda" in data


def test_connect_html_page_served(test_app):
    """Verifies that the /connect portal endpoint serves HTML."""
    response = test_app.get("/connect")
    assert response.status_code == 200
    assert "CineFlow-AI" in response.text
    assert "Connect Cloud GPU" in response.text


def test_connection_save_and_retrieve(test_app, tmp_path):
    """Verifies storing and retrieving Colab URL via API."""
    test_url = "https://cineflow-test-tunnel.trycloudflare.com"
    response = test_app.post("/api/connection", json={"colab_url": test_url})
    assert response.status_code == 200
    assert response.json()["status"] == "success"

    get_resp = test_app.get("/api/connection")
    assert get_resp.status_code == 200
    assert get_resp.json()["colab_url"] == test_url


def test_desktop_verify_colab_url_invalid():
    """Verifies that an invalid or unreachable URL returns False safely without crashing."""
    assert verify_colab_url("") is False
    assert verify_colab_url("http://127.0.0.1:9999", timeout=0.5) is False
