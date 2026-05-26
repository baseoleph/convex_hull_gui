"""Smoke tests for bench_snapshot_chan_scipy.py.

Runs a minimal benchmark matrix (1 algo x 1 scenario x 1 size x 1 seed)
so that broken imports or adapter regressions are caught by CI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

# Skip the entire module when bench dependencies (matplotlib, scipy) are absent.
pytest.importorskip("matplotlib")
pytest.importorskip("scipy")

# Importing the script exercises all its top-level imports (adapters.chan_adapter,
# app.service, verification.canonical, etc.) — the most common failure mode.
from scripts.bench_snapshot_chan_scipy import (
    Cell,
    _median_by_cell,
    run_benchmark,
    summarize_rows,
    write_jsonl,
)


def _minimal_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        algorithms=["chan"],
        scenarios=["points_on_sphere"],
        sizes=["10"],
        seeds=["0"],
        repeats=1,
        verify="none",
        oracle_max_n=50,
        show_algorithm_stdout=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_import_exposes_key_symbols() -> None:
    from scripts.bench_snapshot_chan_scipy import (  # noqa: F401
        DEFAULT_SCENARIOS,
        build_report,
        render_plots,
        run_benchmark,
        summarize_rows,
        write_jsonl,
    )


def test_run_benchmark_returns_one_row() -> None:
    rows = run_benchmark(_minimal_args())
    assert len(rows) == 1
    row = rows[0]
    assert row["algorithm"] == "chan"
    assert row["scenario"] == "points_on_sphere"
    assert row["n"] == 10


def test_run_benchmark_row_has_ok_status() -> None:
    rows = run_benchmark(_minimal_args())
    assert rows[0]["status"] == "ok", rows[0].get("error")


def test_run_benchmark_row_has_timing() -> None:
    rows = run_benchmark(_minimal_args())
    assert rows[0]["time_total_ms"] is not None
    assert rows[0]["time_total_ms"] >= 0.0


def test_run_benchmark_row_has_hull_counts() -> None:
    rows = run_benchmark(_minimal_args())
    row = rows[0]
    assert isinstance(row["h_result"], int)
    assert row["h_result"] > 0
    assert isinstance(row["faces_result"], int)
    assert row["faces_result"] > 0


def test_summarize_rows_produces_per_n_and_slope() -> None:
    rows = run_benchmark(_minimal_args(sizes=["10", "20"], seeds=["0", "1"]))
    summary = summarize_rows(rows)
    kinds = {r["kind"] for r in summary}
    assert "per_n" in kinds
    assert "slope" in kinds


def test_write_jsonl_roundtrip(tmp_path: Path) -> None:
    rows = run_benchmark(_minimal_args())
    out = tmp_path / "out.jsonl"
    write_jsonl(out, rows)
    lines = [json.loads(ln) for ln in out.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1
    assert lines[0]["algorithm"] == rows[0]["algorithm"]


def test_median_by_cell_aggregates_correctly() -> None:
    rows = run_benchmark(_minimal_args(seeds=["0", "1"]))
    ok_rows = [r for r in rows if r["status"] == "ok"]
    medians = _median_by_cell(ok_rows)
    assert len(medians) == 1
    cell = next(iter(medians))
    assert isinstance(cell, Cell)
    assert cell.algorithm == "chan"


def test_benchmark_supports_chan() -> None:
    rows = run_benchmark(_minimal_args(algorithms=["chan"]))
    assert len(rows) == 1
    assert rows[0]["algorithm"] == "chan"
    assert rows[0]["status"] == "ok"


def test_benchmark_supports_chan_v2() -> None:
    if not Path("convex_hull/src/chan_v2.py").exists():
        pytest.skip("convex_hull/src/chan_v2.py not present")
    rows = run_benchmark(_minimal_args(algorithms=["chan_v2"]))
    assert len(rows) == 1
    assert rows[0]["algorithm"] == "chan_v2"
    assert rows[0]["status"] == "ok"


def test_benchmark_supports_scipy() -> None:
    pytest.importorskip("scipy")
    rows = run_benchmark(_minimal_args(algorithms=["scipy"]))
    assert len(rows) == 1
    assert rows[0]["algorithm"] == "scipy"
    assert rows[0]["status"] == "ok"
