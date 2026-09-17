"""
Integration tests for CineFlow NLE Studio FastAPI Endpoints.
Verifies all REST API routes for timeline, color wheels, Fairlight audio, meme vault, transitions/fonts, agentic planning, and token telemetry.
"""

import pytest
from fastapi.testclient import TestClient
from app import CineFlowApp, build_fastapi_app, FASTAPI_AVAILABLE


@pytest.fixture
def api_client():
    if not FASTAPI_AVAILABLE:
        pytest.skip("FastAPI not available")
    app_instance = CineFlowApp(config_path="configs/colab_t4_config.yaml")
    fastapi_app = build_fastapi_app(app_instance)
    return TestClient(fastapi_app)


def test_api_nle_project(api_client):
    res = api_client.get("/api/nle/project")
    assert res.status_code == 200
    data = res.json()
    assert "tracks" in data
    assert "V1" in data["tracks"]
    assert "A1" in data["tracks"]
    assert data["fps"] == 60.0


def test_api_nle_ast(api_client):
    res = api_client.get("/api/nle/ast")
    assert res.status_code == 200
    data = res.json()
    assert "tracks" in data
    assert "dur_s" in data


def test_api_nle_cut(api_client):
    res = api_client.post("/api/nle/cut", json={"track_id": "V1", "frame": 200})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"


def test_api_nle_color_luts_and_grade(api_client):
    res = api_client.get("/api/nle/luts")
    assert res.status_code == 200
    luts = res.json()
    assert len(luts) >= 7

    res_grade = api_client.post("/api/nle/color-grade", json={
        "temperature": 15.0,
        "contrast": 1.15,
        "lut_preset": "Kodak 2383 Film Print"
    })
    assert res_grade.status_code == 200
    data = res_grade.json()
    assert data["status"] == "success"
    assert "scopes" in data


def test_api_nle_audio_channels(api_client):
    res = api_client.get("/api/nle/audio-channels")
    assert res.status_code == 200
    channels = res.json()
    assert "A1" in channels
    assert "eqCurve" in channels["A1"]


def test_api_nle_memes_and_preview(api_client):
    res = api_client.get("/api/nle/memes?category=India")
    assert res.status_code == 200
    memes = res.json()
    assert len(memes) > 0

    res_prev = api_client.get(f"/api/nle/memes/{memes[0]['id']}/preview")
    assert res_prev.status_code == 200
    assert "sampleCount" in res_prev.json()


def test_api_nle_transitions_and_fonts(api_client):
    res_t = api_client.get("/api/nle/transitions")
    assert res_t.status_code == 200
    assert len(res_t.json()) >= 8

    res_f = api_client.get("/api/nle/fonts")
    assert res_f.status_code == 200
    assert any("Bengali" in f["category"] for f in res_f.json())

    res_p = api_client.get("/api/nle/subtitles/presets")
    assert res_p.status_code == 200
    assert len(res_p.json()) >= 4


def test_api_nle_agent_plan_and_execution(api_client):
    # 1. Propose plan ("AI Asks Before Editing")
    res_prop = api_client.post("/api/nle/agent/propose-plan", json={
        "prompt": "Auto-cut silences, grade shot 04, insert Hera Pheri meme, and add captions."
    })
    assert res_prop.status_code == 200
    plan = res_prop.json()
    assert "planId" in plan
    assert plan["approvedByUser"] is False

    # 2. Execute full plan loop
    res_exec = api_client.post("/api/nle/agent/execute-full-plan", json={
        "plan_id": plan["planId"]
    })
    assert res_exec.status_code == 200
    exec_data = res_exec.json()
    assert exec_data["isCompleted"] is True


def test_api_nle_token_telemetry(api_client):
    res = api_client.get("/api/nle/token-telemetry")
    assert res.status_code == 200
    data = res.json()
    assert data["isLessThanOneThird"] is True
    assert data["percentageReduction"] >= 66.7
