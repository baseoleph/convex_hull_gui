"""One-command performance snapshot for legacy chan.py vs SciPy.

Usage from repository root:

    poetry run python scripts/bench_snapshot_chan_scipy.py

The script:
  * runs chan/scipy benchmarks;
  * writes raw JSONL and summary JSONL;
  * renders time-vs-n plots by scenario;
  * prints and saves a compact markdown report with scaling/signature notes.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import statistics
import sys
import traceback
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent if THIS_FILE.parent.name == "scripts" else Path.cwd()
SCRIPTS_DIR = REPO_ROOT / "scripts"
SRC_DIR = REPO_ROOT / "src"

sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SRC_DIR))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from adapters.chan_adapter import ChanHull3D  # noqa: E402
from app.service import ALGORITHMS as _SERVICE_ALGORITHMS  # noqa: E402
from app.service import generate_points  # noqa: E402
from domain.entities import HullResult3D, Point3D  # noqa: E402
from verification.canonical import polygonal_hulls_equal  # noqa: E402
from verification.scipy_oracle import SciPyOracle3D  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_ALGORITHMS: tuple[str, ...] = ("chan", "scipy")
DEFAULT_SCENARIOS: tuple[str, ...] = (
    "controlled_h_8",
    "controlled_h_32",
    "controlled_h_128",
    "controlled_h_sqrt_n",
    "controlled_h_half_n",
    "points_on_sphere",
)
DEFAULT_SIZES: tuple[int, ...] = (128, 256, 512, 1024)
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2)

ALGORITHM_NAMES: tuple[str, ...] = ("chan", "chan_v2", "scipy")



class _GeneratedInputError(RuntimeError):
    """Raised when a scenario cannot generate one requested matrix cell."""


def _parse_tokens(values: Sequence[str] | None, *, default: Sequence[str]) -> tuple[str, ...]:
    if not values:
        return tuple(default)
    result: list[str] = []
    for value in values:
        for token in value.split(","):
            stripped = token.strip()
            if stripped:
                result.append(stripped)
    return tuple(result)


def _parse_int_tokens(
    values: Sequence[str] | None, *, default: Sequence[int]
) -> tuple[int, ...]:
    return tuple(int(t) for t in _parse_tokens(values, default=[str(x) for x in default]))


def _build_algorithm(name: str) -> Any:
    if name == "chan":
        return ChanHull3D()
    if name == "chan_v2":
        return ChanHull3D(chan_path=Path("convex_hull/src/chan_v2.py"))
    if name == "scipy":
        return SciPyOracle3D()
    if name in _SERVICE_ALGORITHMS:
        return _SERVICE_ALGORITHMS[name]()
    raise ValueError(f"unknown algorithm: {name!r}")


def _safe_log2(value: int) -> float:
    return math.log2(max(2, value))


def _normalization_fields(*, n: int, h: int, total_ms: float | None) -> dict[str, Any]:
    h_safe = max(2, h)
    n_log_h = n * _safe_log2(h_safe)
    n_log_n = n * _safe_log2(n)
    n_h = n * h_safe
    if total_ms is None:
        return {
            "n_log2_h": round(n_log_h, 6),
            "n_log2_n": round(n_log_n, 6),
            "n_h": n_h,
            "time_per_n_log2_h_us": None,
            "time_per_n_log2_n_us": None,
            "time_per_n_h_us": None,
        }
    return {
        "n_log2_h": round(n_log_h, 6),
        "n_log2_n": round(n_log_n, 6),
        "n_h": n_h,
        "time_per_n_log2_h_us": round(total_ms * 1000.0 / n_log_h, 6),
        "time_per_n_log2_n_us": round(total_ms * 1000.0 / n_log_n, 6),
        "time_per_n_h_us": round(total_ms * 1000.0 / n_h, 6),
    }


def _hull_size_from_reference(
    points: Sequence[Point3D], *, seed: int
) -> tuple[int, int, HullResult3D]:
    result = SciPyOracle3D().compute(points, seed=seed)
    return len(result.vertex_indices), len(result.faces), result


def _compute_algorithm(
    algorithm: Any,
    points: Sequence[Point3D],
    *,
    seed: int,
    capture_stdout: bool,
) -> tuple[HullResult3D, str]:
    if capture_stdout:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            result = algorithm.compute(points, seed=seed)
        return result, buffer.getvalue()
    result = algorithm.compute(points, seed=seed)
    return result, ""


def _empty_row(
    *,
    algorithm: str,
    scenario: str,
    n: int,
    seed: int,
    repeat_id: int,
    verify: str,
) -> dict[str, Any]:
    return {
        "algorithm": algorithm,
        "scenario": scenario,
        "n": n,
        "seed": seed,
        "repeat_id": repeat_id,
        "verify": verify,
        "status": "ok",
        "error": None,
        "n_input": None,
        "h_ref": None,
        "faces_ref": None,
        "h_result": None,
        "faces_result": None,
        "correct": None,
        "time_total_ms": None,
        "algorithm_stdout_bytes": 0,
    }


def _run_one(
    *,
    algorithm_name: str,
    scenario: str,
    n: int,
    seed: int,
    repeat_id: int,
    verify: str,
    oracle_max_n: int,
    capture_stdout: bool,
    reference_result: HullResult3D | None,
) -> dict[str, Any]:
    row = _empty_row(
        algorithm=algorithm_name,
        scenario=scenario,
        n=n,
        seed=seed,
        repeat_id=repeat_id,
        verify=verify,
    )

    try:
        points = generate_points(scenario, n, seed=seed)
    except Exception as exc:
        row["status"] = "skipped_generation"
        row["error"] = f"generation failed: {exc}"
        return row

    row["n_input"] = len(points)

    if reference_result is None:
        try:
            h_ref, faces_ref, reference_result = _hull_size_from_reference(points, seed=seed)
        except Exception as exc:
            raise _GeneratedInputError(
                f"SciPy reference failed for {scenario}/n={n}: {exc}"
            ) from exc
    else:
        h_ref = len(reference_result.vertex_indices)
        faces_ref = len(reference_result.faces)

    row["h_ref"] = h_ref
    row["faces_ref"] = faces_ref
    row.update(_normalization_fields(n=len(points), h=h_ref, total_ms=None))

    if verify == "scipy-small-only" and len(points) > oracle_max_n:
        reference_for_correctness = None
    elif verify in ("scipy-small-only", "scipy-all"):
        reference_for_correctness = reference_result
    else:
        reference_for_correctness = None

    algorithm = _build_algorithm(algorithm_name)
    started_at = perf_counter()
    try:
        result, captured = _compute_algorithm(
            algorithm, points, seed=seed, capture_stdout=capture_stdout
        )
    except Exception as exc:
        total_ms = (perf_counter() - started_at) * 1000.0
        row["status"] = "failed"
        row["error"] = traceback.format_exception_only(type(exc), exc)[0].strip()
        row["time_total_ms"] = round(total_ms, 3)
        row.update(_normalization_fields(n=len(points), h=h_ref, total_ms=total_ms))
        return row

    total_ms = (perf_counter() - started_at) * 1000.0
    row["time_total_ms"] = round(total_ms, 3)
    row["h_result"] = len(result.vertex_indices)
    row["faces_result"] = len(result.faces)
    row["algorithm_stdout_bytes"] = len(captured.encode("utf-8"))
    row.update(_normalization_fields(n=len(points), h=h_ref, total_ms=total_ms))

    if reference_for_correctness is not None:
        row["correct"] = polygonal_hulls_equal(result, reference_for_correctness)
        if not row["correct"]:
            row["status"] = "correctness_failed"
            row["error"] = "result does not match SciPy reference"
    else:
        row["correct"] = None

    return row


def _bench_slope(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom == 0.0:
        return None
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True)) / denom


def run_benchmark(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Run the configured asymptotic benchmark matrix and return raw rows."""
    algorithms = _parse_tokens(args.algorithms, default=DEFAULT_ALGORITHMS)
    unknown = sorted(set(algorithms).difference(ALGORITHM_NAMES).difference(_SERVICE_ALGORITHMS))
    if unknown:
        raise ValueError(f"unsupported algorithms: {unknown!r}")

    scenarios = _parse_tokens(args.scenarios, default=DEFAULT_SCENARIOS)
    sizes = _parse_int_tokens(args.sizes, default=DEFAULT_SIZES)
    seeds = _parse_int_tokens(args.seeds, default=DEFAULT_SEEDS)

    rows: list[dict[str, Any]] = []
    matrix = [
        (scenario, n, seed, repeat_id, algorithm)
        for scenario in scenarios
        for n in sizes
        for seed in seeds
        for repeat_id in range(args.repeats)
        for algorithm in algorithms
    ]

    reference_cache: dict[tuple[str, int, int], HullResult3D | None] = {}
    total = len(matrix)
    for index, (scenario, n, seed, repeat_id, algorithm) in enumerate(matrix, start=1):
        print(
            f"[{index}/{total}] {algorithm} scenario={scenario} n={n} seed={seed} rep={repeat_id}",
            flush=True,
        )
        reference_key = (scenario, n, seed)
        if reference_key not in reference_cache:
            try:
                points = generate_points(scenario, n, seed=seed)
                _, _, reference = _hull_size_from_reference(points, seed=seed)
            except Exception:
                reference = None
            reference_cache[reference_key] = reference

        row = _run_one(
            algorithm_name=algorithm,
            scenario=scenario,
            n=n,
            seed=seed,
            repeat_id=repeat_id,
            verify=args.verify,
            oracle_max_n=args.oracle_max_n,
            capture_stdout=not args.show_algorithm_stdout,
            reference_result=reference_cache[reference_key],
        )
        rows.append(row)

    return rows


def summarize_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build compact asymptotic summary rows from detailed run records."""
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") != "ok" or row.get("time_total_ms") is None:
            continue
        key = (str(row["algorithm"]), str(row["scenario"]), int(row["n"]))
        grouped.setdefault(key, []).append(row)

    per_n: list[dict[str, Any]] = []
    for (algorithm, scenario, n), group_rows in sorted(grouped.items()):
        times = [float(row["time_total_ms"]) for row in group_rows]
        h_values = [int(row["h_ref"]) for row in group_rows if row.get("h_ref") is not None]
        row0 = group_rows[0]
        per_n.append(
            {
                "algorithm": algorithm,
                "scenario": scenario,
                "n": n,
                "h_median": _bench_median([float(x) for x in h_values]),
                "repeat_count": len(group_rows),
                "median_ms": round(float(statistics.median(times)), 3),
                "min_ms": round(min(times), 3),
                "max_ms": round(max(times), 3),
                "median_time_per_n_log2_h_us": _bench_median(
                    [float(row["time_per_n_log2_h_us"]) for row in group_rows]
                ),
                "median_time_per_n_h_us": _bench_median(
                    [float(row["time_per_n_h_us"]) for row in group_rows]
                ),
                "n_log2_h": row0.get("n_log2_h"),
                "n_h": row0.get("n_h"),
            }
        )

    by_algo_scenario: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in per_n:
        by_algo_scenario.setdefault((row["algorithm"], row["scenario"]), []).append(row)

    summary: list[dict[str, Any]] = []
    for (algorithm, scenario), group_rows in sorted(by_algo_scenario.items()):
        ordered = sorted(group_rows, key=lambda r: int(r["n"]))
        xs_n = [math.log(float(r["n"])) for r in ordered if r["median_ms"] > 0]
        ys_t = [math.log(float(r["median_ms"])) for r in ordered if r["median_ms"] > 0]
        xs_n_log_h = [math.log(float(r["n_log2_h"])) for r in ordered if r["median_ms"] > 0]
        xs_n_h = [math.log(float(r["n_h"])) for r in ordered if r["median_ms"] > 0]
        summary.append(
            {
                "algorithm": algorithm,
                "scenario": scenario,
                "points": len(ordered),
                "slope_log_time_vs_log_n": (
                    None if (s := _bench_slope(xs_n, ys_t)) is None else round(s, 3)
                ),
                "slope_log_time_vs_log_n_log_h": (
                    None if (s := _bench_slope(xs_n_log_h, ys_t)) is None else round(s, 3)
                ),
                "slope_log_time_vs_log_n_h": (
                    None if (s := _bench_slope(xs_n_h, ys_t)) is None else round(s, 3)
                ),
                "smallest_n": ordered[0]["n"],
                "largest_n": ordered[-1]["n"],
                "largest_median_ms": ordered[-1]["median_ms"],
                "largest_time_per_n_log2_h_us": ordered[-1]["median_time_per_n_log2_h_us"],
                "largest_time_per_n_h_us": ordered[-1]["median_time_per_n_h_us"],
            }
        )

    return [{"kind": "per_n", **r} for r in per_n] + [{"kind": "slope", **r} for r in summary]


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write benchmark rows to a JSON Lines file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")


# ---------------------------------------------------------------------------
# Report / plotting helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """Unique key for one (algorithm, scenario, n) cell in the benchmark matrix."""

    algorithm: str
    scenario: str
    n: int


def _tokens(values: Sequence[str] | None, default: Sequence[str]) -> tuple[str, ...]:
    return _parse_tokens(values, default=default)


def _int_tokens(values: Sequence[str] | None, default: Sequence[int]) -> tuple[int, ...]:
    return _parse_int_tokens(values, default=default)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def _bench_median(values: Iterable[float]) -> float | None:
    materialized = list(values)
    if not materialized:
        return None
    return float(statistics.median(materialized))


def _ok_timed_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("status") == "ok" and row.get("time_total_ms") is not None
    ]


def _median_by_cell(rows: Sequence[dict[str, Any]]) -> dict[Cell, dict[str, float]]:
    grouped: dict[Cell, list[dict[str, Any]]] = defaultdict(list)
    for row in _ok_timed_rows(rows):
        grouped[
            Cell(
                algorithm=str(row["algorithm"]),
                scenario=str(row["scenario"]),
                n=int(row["n"]),
            )
        ].append(row)

    medians: dict[Cell, dict[str, float]] = {}
    for cell, bucket in grouped.items():
        medians[cell] = {
            "median_ms": float(statistics.median(float(r["time_total_ms"]) for r in bucket)),
            "h_median": float(
                statistics.median(
                    float(r["h_ref"]) for r in bucket if r.get("h_ref") is not None
                )
            ),
            "faces_median": float(
                statistics.median(
                    float(r["faces_ref"]) for r in bucket if r.get("faces_ref") is not None
                )
            ),
            "time_per_n_log2_h_us": float(
                statistics.median(
                    float(r["time_per_n_log2_h_us"])
                    for r in bucket
                    if r.get("time_per_n_log2_h_us") is not None
                )
            ),
            "repeat_count": float(len(bucket)),
        }
    return medians


def _slope_log_time_vs_log_n(points: dict[int, float]) -> float | None:
    usable = [(n, t) for n, t in sorted(points.items()) if n > 0 and t > 0]
    if len(usable) < 2:
        return None
    xs = [math.log(float(n)) for n, _ in usable]
    ys = [math.log(float(t)) for _, t in usable]
    return _bench_slope(xs, ys)


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def _plot_series(
    path: Path,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    series: dict[str, dict[int, float]],
    log_y: bool,
    hline: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5.5))
    for label, points in sorted(series.items()):
        if not points:
            continue
        xs = sorted(points)
        ys = [points[x] for x in xs]
        plt.plot(xs, ys, marker="o", label=label)
    if hline is not None:
        plt.axhline(hline, linestyle="--", linewidth=1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    if log_y:
        plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def render_plots(rows: Sequence[dict[str, Any]], *, output_dir: Path, log_y: bool) -> list[Path]:
    """Render per-scenario runtime plots and return the list of written PNG paths."""
    medians = _median_by_cell(rows)
    scenarios = sorted({cell.scenario for cell in medians})
    plot_dir = output_dir / "plots"
    created: list[Path] = []

    for scenario in scenarios:
        series: dict[str, dict[int, float]] = defaultdict(dict)
        for cell, values in medians.items():
            if cell.scenario == scenario:
                series[cell.algorithm][cell.n] = values["median_ms"]

        current_ns = sorted({cell.n for cell in medians if cell.scenario == scenario})
        theory_const = 1
        for n in current_ns:
            if n > 0:
                val_theoretical = theory_const * n * math.log2(n)
                series["cn log n"][n] = val_theoretical

        path = plot_dir / f"time_ms_vs_n__{_safe_name(scenario)}.png"
        _plot_series(
            path,
            title=f"Runtime vs cn log n: {scenario}",
            xlabel="n points",
            ylabel="median total time, ms",
            series=dict(series),
            log_y=log_y,
        )
        created.append(path)

    return created


def _format_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    if not rows:
        return ["_No data._"]
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *["| " + " | ".join(row) + " |" for row in rows],
    ]


def _status_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get("status"))] += 1
    return dict(sorted(counts.items()))


def _top_errors(rows: Sequence[dict[str, Any]], *, limit: int) -> list[tuple[str, int]]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        if row.get("status") != "ok":
            message = str(row.get("error") or row.get("status"))
            if len(message) > 180:
                message = message[:177] + "..."
            counts[message] += 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]


def build_report(
    rows: Sequence[dict[str, Any]],
    *,
    output_dir: Path,
    raw_jsonl: Path,
    summary_jsonl: Path,
    plot_paths: Sequence[Path],
    args: argparse.Namespace,
) -> str:
    """Build and return a markdown performance report from raw benchmark rows."""
    medians = _median_by_cell(rows)
    scenarios = sorted({cell.scenario for cell in medians})
    algorithms = sorted({cell.algorithm for cell in medians})
    status_counts = _status_counts(rows)

    lines: list[str] = []
    lines.append("# chan.py vs SciPy performance snapshot")
    lines.append("")
    lines.append(f"Generated at: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append(f"Repository root: `{REPO_ROOT}`")
    lines.append("")
    lines.append("## Configuration")
    lines.extend(
        _markdown_table(
            ["field", "value"],
            [
                ["algorithms", ", ".join(_tokens(args.algorithms, DEFAULT_ALGORITHMS))],
                ["scenarios", ", ".join(_tokens(args.scenarios, DEFAULT_SCENARIOS))],
                ["sizes", ", ".join(str(x) for x in _int_tokens(args.sizes, DEFAULT_SIZES))],
                ["seeds", ", ".join(str(x) for x in _int_tokens(args.seeds, DEFAULT_SEEDS))],
                ["repeats", str(args.repeats)],
                ["verify", str(args.verify)],
                ["oracle_max_n", str(args.oracle_max_n)],
            ],
        )
    )
    lines.append("")
    lines.append("## Files")
    lines.extend(
        _markdown_table(
            ["artifact", "path"],
            [
                ["raw JSONL", f"`{raw_jsonl}`"],
                ["summary JSONL", f"`{summary_jsonl}`"],
                ["report", f"`{output_dir / 'report.md'}`"],
                ["plots dir", f"`{output_dir / 'plots'}`"],
            ],
        )
    )
    lines.append("")
    lines.append("## Run status")
    lines.extend(
        _markdown_table(["status", "count"], [[k, str(v)] for k, v in status_counts.items()])
    )
    lines.append("")

    if plot_paths:
        lines.append("## Plots")
        for path in plot_paths:
            try:
                rel_path = path.relative_to(output_dir)
            except ValueError:
                rel_path = path
            lines.append(f"- `{rel_path}`")
        lines.append("")

    lines.append("## Largest-n comparison")
    largest_rows: list[list[str]] = []
    for scenario in scenarios:
        ns = sorted({cell.n for cell in medians if cell.scenario == scenario})
        if not ns:
            continue
        n = ns[-1]
        chan = medians.get(Cell("chan", scenario, n))
        scipy = medians.get(Cell("scipy", scenario, n))
        h_values = [
            values["h_median"]
            for cell, values in medians.items()
            if cell.scenario == scenario and cell.n == n
        ]
        h_median = _bench_median(h_values)
        ratio = None
        if chan and scipy and chan["median_ms"] > 0.0:
            ratio = scipy["median_ms"] / chan["median_ms"]
        candidates = [
            (cell.algorithm, values["median_ms"])
            for cell, values in medians.items()
            if cell.scenario == scenario and cell.n == n
        ]
        fastest = min(candidates, key=lambda item: item[1])[0] if candidates else "n/a"
        largest_rows.append(
            [
                scenario,
                str(n),
                _format_float(h_median, 0),
                _format_float(chan["median_ms"] if chan else None, 3),
                _format_float(scipy["median_ms"] if scipy else None, 3),
                _format_float(ratio, 3),
                fastest,
            ]
        )
    lines.extend(
        _markdown_table(
            ["scenario", "n", "h~", "chan ms", "scipy ms", "scipy/chan", "fastest"],
            largest_rows,
        )
    )
    lines.append("")

    lines.append("## Scaling signatures")
    scaling_rows: list[list[str]] = []
    automated_notes: list[str] = []
    for scenario in scenarios:
        for algorithm in algorithms:
            series = {
                cell.n: values["median_ms"]
                for cell, values in medians.items()
                if cell.scenario == scenario and cell.algorithm == algorithm
            }
            if len(series) < 2:
                continue
            slope = _slope_log_time_vs_log_n(series)
            norm_series = {
                cell.n: values["time_per_n_log2_h_us"]
                for cell, values in medians.items()
                if cell.scenario == scenario and cell.algorithm == algorithm
            }
            norm_growth = None
            if len(norm_series) >= 2:
                first_n, last_n = min(norm_series), max(norm_series)
                first_v = norm_series[first_n]
                last_v = norm_series[last_n]
                if first_v > 0.0:
                    norm_growth = last_v / first_v
            scaling_rows.append(
                [
                    scenario,
                    algorithm,
                    _format_float(slope, 3),
                    _format_float(norm_growth, 2),
                    _format_float(series[max(series)], 3),
                    _format_float(norm_series[max(norm_series)] if norm_series else None, 3),
                ]
            )
            if algorithm == "chan" and slope is not None:
                if slope > 1.45:
                    automated_notes.append(
                        f"`{scenario}`: chan has steep empirical slope `{slope:.2f}`; "
                        "this is a likely DK/wrapping-heavy regime."
                    )
                elif slope < 1.20:
                    automated_notes.append(
                        f"`{scenario}`: chan is close to linear in this snapshot "
                        f"(`slope={slope:.2f}`)."
                    )
            if algorithm == "chan" and norm_growth is not None and norm_growth > 1.8:
                automated_notes.append(
                    f"`{scenario}`: chan normalized time per `n log h` grew "
                    f"by `{norm_growth:.2f}x`; constant factors still grow with n."
                )
    lines.extend(
        _markdown_table(
            ["scenario", "algorithm", "slope T~n^a", "norm growth", "largest ms",
             "largest us/(nlogh)"],
            scaling_rows,
        )
    )
    lines.append("")

    lines.append("## Automated notes")
    if automated_notes:
        for note in automated_notes:
            lines.append(f"- {note}")
    else:
        lines.append("- No strong scaling warnings from the selected snapshot matrix.")
    lines.append("")

    errors = _top_errors(rows, limit=args.max_report_errors)
    if errors:
        lines.append("## Top failures/errors")
        lines.extend(
            _markdown_table(
                ["count", "error"], [[str(count), f"`{error}`"] for error, count in errors]
            )
        )
        lines.append("")

    lines.append("## Reproduce")
    lines.append("```bash")
    lines.append(
        "poetry run python scripts/bench_snapshot_chan_scipy.py "
        f"--output-dir {output_dir} "
        f"--algorithms {' '.join(_tokens(args.algorithms, DEFAULT_ALGORITHMS))} "
        f"--scenarios {' '.join(_tokens(args.scenarios, DEFAULT_SCENARIOS))} "
        f"--sizes {' '.join(str(x) for x in _int_tokens(args.sizes, DEFAULT_SIZES))} "
        f"--seeds {' '.join(str(x) for x in _int_tokens(args.seeds, DEFAULT_SEEDS))} "
        f"--repeats {args.repeats} --verify {args.verify}"
    )
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _bench_namespace(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        algorithms=list(_tokens(args.algorithms, DEFAULT_ALGORITHMS)),
        scenarios=list(_tokens(args.scenarios, DEFAULT_SCENARIOS)),
        sizes=[str(x) for x in _int_tokens(args.sizes, DEFAULT_SIZES)],
        seeds=[str(x) for x in _int_tokens(args.seeds, DEFAULT_SEEDS)],
        repeats=args.repeats,
        verify=args.verify,
        oracle_max_n=args.oracle_max_n,
        show_algorithm_stdout=args.show_algorithm_stdout,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a compact chan.py vs SciPy benchmark snapshot and plot it."
    )
    parser.add_argument("--algorithms", nargs="+", default=list(DEFAULT_ALGORITHMS))
    parser.add_argument("--scenarios", nargs="+", default=list(DEFAULT_SCENARIOS))
    parser.add_argument("--sizes", nargs="+", default=[str(x) for x in DEFAULT_SIZES])
    parser.add_argument("--seeds", nargs="+", default=[str(x) for x in DEFAULT_SEEDS])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--verify",
        choices=("none", "scipy-small-only", "scipy-all"),
        default="scipy-small-only",
    )
    parser.add_argument("--oracle-max-n", type=int, default=512)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Snapshot directory. Default: benchmarks/snapshot_chan_scipy_<timestamp>",
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=None,
        help="Reuse an existing raw JSONL instead of running benchmarks.",
    )
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument(
        "--linear-y", action="store_true", help="Do not use log scale on runtime plots."
    )
    parser.add_argument("--show-algorithm-stdout", action="store_true", default=False)
    parser.add_argument("--max-report-errors", type=int, default=12)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the full benchmark-snapshot pipeline: measure, write JSONL, render plots, report."""
    args = _parse_args(argv)
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = Path("benchmarks") / f"snapshot_chan_scipy_{timestamp}"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_jsonl = output_dir / "raw.jsonl"
    summary_jsonl = output_dir / "summary.jsonl"

    if args.input_jsonl is not None:
        rows = _read_jsonl(args.input_jsonl)
        raw_jsonl = args.input_jsonl
        print(f"Loaded existing raw JSONL: {raw_jsonl}")
    else:
        bench_args = _bench_namespace(args)
        rows = run_benchmark(bench_args)
        write_jsonl(raw_jsonl, rows)
        print(f"Wrote raw JSONL: {raw_jsonl}")

    summary_rows = summarize_rows(rows)
    write_jsonl(summary_jsonl, summary_rows)
    print(f"Wrote summary JSONL: {summary_jsonl}")

    plot_paths: list[Path] = []
    if not args.no_plots:
        plot_paths = render_plots(rows, output_dir=output_dir, log_y=not args.linear_y)
        print(f"Wrote {len(plot_paths)} plot(s) to: {output_dir / 'plots'}")

    report = build_report(
        rows,
        output_dir=output_dir,
        raw_jsonl=raw_jsonl,
        summary_jsonl=summary_jsonl,
        plot_paths=plot_paths,
        args=args,
    )
    report_path = output_dir / "report.md"
    _write_text(report_path, report)
    print(f"Wrote report: {report_path}")
    print("\n" + report)


if __name__ == "__main__":
    main()
