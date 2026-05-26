from __future__ import annotations

from app import service as app_service
from gui.state import GuiState


def test_gui_state_prefers_chan_and_uniform_sphere(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        app_service,
        "ALGORITHMS",
        {"bruteforce_degenerate": object, "chan": object},
    )
    monkeypatch.setattr(
        app_service,
        "SCENARIO_MAP",
        {"uniform_sphere": object, "uniform_cube": object},
    )
    monkeypatch.setattr(
        app_service,
        "SCENARIOS",
        [("uniform_sphere", object), ("uniform_cube", object)],
    )

    state = GuiState()

    assert state.algorithm == "chan"
    assert state.scenario == "uniform_sphere"
    assert state.n == 20
    assert state.seed is None
    assert state.points is None
    assert state.result is None
    assert state.last_error is None
    assert state.last_used_seed is None
    assert state.last_build_seconds is None


def test_gui_state_falls_back_to_first_sorted_algorithm_and_first_scenario(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(app_service, "ALGORITHMS", {"zeta": object, "alpha": object})
    monkeypatch.setattr(app_service, "SCENARIO_MAP", {"gamma": object, "beta": object})
    monkeypatch.setattr(app_service, "SCENARIOS", [("gamma", object), ("beta", object)])

    state = GuiState()

    assert state.algorithm == "alpha"
    assert state.scenario == "gamma"
