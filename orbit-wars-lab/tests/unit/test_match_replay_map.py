from __future__ import annotations

from types import SimpleNamespace

from orbit_wars_app.match import (
    _inject_replay_comet_group,
    _replay_comet_schedule_by_step,
    _replay_map_initial_step,
)


def test_replay_map_initial_step_preserves_planets_for_each_player():
    replay_map = {
        "planets": [
            [0, 0, 10, 10, 1, 10, 1],
            [1, 1, 90, 90, 1, 10, 1],
        ],
        "initial_planets": [
            [0, -1, 10, 10, 1, 5, 1],
            [1, -1, 90, 90, 1, 5, 1],
        ],
        "angular_velocity": 0.031,
    }

    step = _replay_map_initial_step(replay_map, 2)

    assert len(step) == 2
    assert step[0]["observation"]["player"] == 0
    assert step[1]["observation"]["player"] == 1
    assert step[0]["observation"]["planets"] == replay_map["planets"]
    assert step[1]["observation"]["planets"] == replay_map["planets"]
    assert step[0]["observation"]["initial_planets"] == replay_map["initial_planets"]
    assert "step" in step[0]["observation"]
    assert "step" not in step[1]["observation"]


def test_replay_map_initial_step_rejects_empty_planets():
    try:
        _replay_map_initial_step({"planets": [], "angular_velocity": 0.03}, 2)
    except ValueError as e:
        assert "planets is empty" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_replay_comet_schedule_normalizes_by_spawn_step():
    replay_map = {
        "comet_schedule": [
            {
                "spawn_step": 50,
                "paths": [
                    [[1, 2], [3, 4]],
                    [[5, 6], [7, 8]],
                    [[9, 10], [11, 12]],
                    [[13, 14], [15, 16]],
                ],
                "ships": 13,
            }
        ]
    }

    schedule = _replay_comet_schedule_by_step(replay_map)

    assert sorted(schedule) == [50]
    assert schedule[50]["ships"] == 13
    assert schedule[50]["paths"][0][0] == [1.0, 2.0]


def test_inject_replay_comet_group_uses_next_planet_ids_and_offboard_start():
    obs = SimpleNamespace(
        planets=[
            [0, 0, 10, 10, 1, 10, 1],
            [1, 1, 90, 90, 1, 10, 1],
        ],
        initial_planets=[
            [0, 0, 10, 10, 1, 10, 1],
            [1, 1, 90, 90, 1, 10, 1],
        ],
        comets=[],
        comet_planet_ids=[],
    )
    paths = [
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]],
        [[9, 10], [11, 12]],
        [[13, 14], [15, 16]],
    ]

    group = _inject_replay_comet_group(obs, {"paths": paths, "ships": 13})

    assert group["planet_ids"] == [2, 3, 4, 5]
    assert group["path_index"] == -1
    assert obs.comet_planet_ids == [2, 3, 4, 5]
    assert [p[:7] for p in obs.planets[-4:]] == [
        [2, -1, -99, -99, 1.0, 13, 1],
        [3, -1, -99, -99, 1.0, 13, 1],
        [4, -1, -99, -99, 1.0, 13, 1],
        [5, -1, -99, -99, 1.0, 13, 1],
    ]
    paths[0][0][0] = 999
    assert group["paths"][0][0] == [1, 2]
