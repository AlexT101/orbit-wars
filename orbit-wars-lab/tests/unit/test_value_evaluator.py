from __future__ import annotations

from pathlib import Path

from orbit_wars_app import value_evaluator


def test_resolve_value_model_path_maps_host_bots_path_to_zoo_root(tmp_path, monkeypatch):
    zoo_root = tmp_path / "agents"
    model = (
        zoo_root
        / "mine"
        / "trojan_horse"
        / "train"
        / "weights"
        / "xgb_46p12e88t11_latest.json"
    )
    model.parent.mkdir(parents=True)
    model.write_text("{}")

    monkeypatch.setattr(value_evaluator, "_candidate_zoo_roots", lambda: [zoo_root])

    host_path = Path(
        "/not-mounted/pantheow/bots/mine/trojan_horse/train/weights/xgb_46p12e88t11_latest.json"
    )

    assert value_evaluator.resolve_value_model_path(host_path) == model


def test_resolve_value_model_path_maps_relative_bots_path_to_zoo_root(tmp_path, monkeypatch):
    zoo_root = tmp_path / "agents"
    model = zoo_root / "mine" / "trojan_horse" / "train" / "weights" / "model.json"
    model.parent.mkdir(parents=True)
    model.write_text("{}")

    monkeypatch.setattr(value_evaluator, "_candidate_zoo_roots", lambda: [zoo_root])

    assert (
        value_evaluator.resolve_value_model_path(
            "bots/mine/trojan_horse/train/weights/model.json"
        )
        == model
    )
