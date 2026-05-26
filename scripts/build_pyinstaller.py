"""Build standalone GUI folder with PyInstaller.

Usage:
    poetry run python scripts/build_pyinstaller.py
"""

from __future__ import annotations

import importlib.util
import os
import shlex
import shutil
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BUILD_ROOT = PROJECT_ROOT / "build"
OUT_ROOT = PROJECT_ROOT / "out"
PYINSTALLER_ROOT = BUILD_ROOT / ".pyinstaller"

ENTRYPOINT = PYINSTALLER_ROOT / "entrypoint.py"
NO_MYPY_RUNTIME_HOOK = PYINSTALLER_ROOT / "runtime_no_mypy.py"

TARGET_DIR = BUILD_ROOT / "convex_hull"


class TeeLog:
    """Write build output both to console and to a log file."""

    def __init__(self, path: Path) -> None:
        """Open the log file at *path* for line-buffered writing."""
        self.path = path
        self._file = path.open("w", encoding="utf-8", buffering=1)

    def close(self) -> None:
        """Flush and close the underlying log file."""
        self._file.close()

    def __enter__(self) -> TeeLog:
        """Return self for use as a context manager."""
        return self

    def __exit__(self, exc_type, exc_value, exc_tb) -> None:  # type: ignore[no-untyped-def]
        """Close the log file on context exit."""
        self.close()

    def write(self, text: str, *, stream=None) -> None:  # type: ignore[no-untyped-def]
        """Write *text* to both the console stream and the log file."""
        if stream is None:
            stream = sys.stdout

        stream.write(text)
        stream.flush()

        self._file.write(text)
        self._file.flush()

    def line(self, text: str = "", *, stream=None) -> None:  # type: ignore[no-untyped-def]
        """Write *text* followed by a newline."""
        self.write(f"{text}\n", stream=stream)


def _git_version() -> str:
    """Return the exact git tag for HEAD, or the short commit hash."""
    for cmd in (
        ["git", "describe", "--tags", "--exact-match", "HEAD"],
        ["git", "rev-parse", "--short", "HEAD"],
    ):
        try:
            out = subprocess.check_output(
                cmd, cwd=str(PROJECT_ROOT), stderr=subprocess.DEVNULL, text=True
            ).strip()
            if out:
                return out
        except subprocess.CalledProcessError:
            pass
    return "unknown"


def _rmtree_force(path: Path) -> None:
    """Remove a directory tree, fixing read-only files on Windows before retrying."""

    def _on_error(func, p, exc):  # type: ignore[no-untyped-def]
        try:
            os.chmod(p, 0o777)
            func(p)
        except Exception:
            pass

    shutil.rmtree(path, onexc=_on_error)


def _add_data_arg(source: Path, dest: str) -> str:
    return f"{source}{os.pathsep}{dest}"


def _format_command(command: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(command)

    return " ".join(shlex.quote(part) for part in command)


def _find_installed_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _run_streamed(command: list[str], log: TeeLog) -> int:
    """Run a child process and tee its combined stdout/stderr to console and log."""
    env = os.environ.copy()

    # Make PyInstaller child process output predictable in redirected mode too.
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    creationflags = 0
    if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
        # Does not hide this script's console; only prevents extra child consoles.
        creationflags = subprocess.CREATE_NO_WINDOW

    log.line()
    log.line("Running PyInstaller:")
    log.line(_format_command(command))
    log.line()

    process = subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
        creationflags=creationflags,
    )

    try:
        assert process.stdout is not None

        for line in process.stdout:
            log.write(line)

        return process.wait()

    except KeyboardInterrupt:
        log.line("\nBuild interrupted. Terminating PyInstaller...", stream=sys.stderr)

        process.terminate()

        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

        return 130


def _write_entrypoint() -> None:
    """Write a PyInstaller entrypoint that can run both GUI and the hull worker.

    In the source tree the GUI starts the worker as:

        sys.executable -X utf8 -m app.hull_worker ...

    After freezing, sys.executable points to convex_hull.exe, not to python.exe.
    Without this dispatcher, the worker launch starts a second GUI window.
    """
    ENTRYPOINT.write_text(
        "from __future__ import annotations\n\n"
        "import multiprocessing\n"
        "import sys\n\n"
        "def _run_hull_worker(worker_args: list[str]) -> None:\n"
        "    sys.argv = [sys.argv[0], *worker_args]\n"
        "    from app.hull_worker import main as worker_main\n"
        "    worker_main()\n\n"
        "def _maybe_run_worker() -> bool:\n"
        "    argv = sys.argv[1:]\n"
        "    if len(argv) >= 4 and argv[:4] == ['-X', 'utf8', '-m', 'app.hull_worker']:\n"
        "        _run_hull_worker(argv[4:])\n"
        "        return True\n"
        "    if len(argv) >= 2 and argv[:2] == ['-m', 'app.hull_worker']:\n"
        "        _run_hull_worker(argv[2:])\n"
        "        return True\n"
        "    if len(argv) >= 1 and argv[0] == '--hull-worker':\n"
        "        _run_hull_worker(argv[1:])\n"
        "        return True\n"
        "    return False\n\n"
        "if __name__ == '__main__':\n"
        "    multiprocessing.freeze_support()\n"
        "    if not _maybe_run_worker():\n"
        "        from app.startup_profile import mark\n"
        "        mark('entrypoint_gui_start')\n"
        "        from gui.main import main\n"
        "        mark('after_gui_import')\n"
        "        main()\n",
        encoding="utf-8",
    )


def _write_no_mypy_runtime_hook() -> None:
    """Hide mypy/mypyc from runtime imports inside the frozen application.

    PyVista has a mypy plugin module. If mypy is present in the build environment,
    PyInstaller can accidentally freeze an incomplete mypy/mypyc set and then the
    packaged application may fail at startup with:

        ModuleNotFoundError: No module named '<hash>__mypyc'

    The GUI does not need mypy at runtime, so the safest behavior is to make mypy
    invisible inside the packaged application.
    """
    NO_MYPY_RUNTIME_HOOK.write_text(
        "import importlib.util\n\n"
        "_real_find_spec = importlib.util.find_spec\n\n"
        "def _find_spec_without_mypy(name, *args, **kwargs):\n"
        "    if name == 'mypy' or name.startswith('mypy.'):\n"
        "        return None\n"
        "    if name == 'mypyc' or name.startswith('mypyc.'):\n"
        "        return None\n"
        "    return _real_find_spec(name, *args, **kwargs)\n\n"
        "importlib.util.find_spec = _find_spec_without_mypy\n",
        encoding="utf-8",
    )

def _copy_external_algorithm_sources(log: TeeLog) -> None:
    runtime_src_dir = TARGET_DIR / "convex_hull" / "src"
    runtime_src_dir.mkdir(parents=True, exist_ok=True)

    files = [
        PROJECT_ROOT / "convex_hull" / "src" / "chan.py",
        PROJECT_ROOT / "convex_hull" / "src" / "bruteforce_degenerate.py",
    ]

    log.line()
    log.line("Copying external algorithm sources for runtime adapters...")

    for source in files:
        if not source.exists():
            raise FileNotFoundError(f"Required source file does not exist: {source}")
        target = runtime_src_dir / source.name
        shutil.copy2(source, target)
        log.line(f"Copied: {source} -> {target}")


def _build(log: TeeLog) -> int:
    if _find_installed_module("PyInstaller") is False:
        log.line(
            "PyInstaller is not installed. Run: poetry install",
            stream=sys.stderr,
        )
        return 1

    log.line(f"Project root: {PROJECT_ROOT}")
    log.line(f"Build root:   {BUILD_ROOT}")
    log.line(f"Log file:     {log.path}")

    installed_dev_modules = [
        name
        for name in ("mypy", "mypyc", "pytest", "hypothesis", "ruff", "scalene")
        if _find_installed_module(name)
    ]

    if installed_dev_modules:
        log.line()
        log.line(
            "Warning: dev modules are installed in this Poetry environment: "
            + ", ".join(installed_dev_modules),
            stream=sys.stderr,
        )
        log.line(
            "The build script will exclude them from the packaged app, "
            "but a cleaner build environment is still recommended.",
            stream=sys.stderr,
        )

    log.line()
    log.line("Cleaning previous build artifacts...")

    _rmtree_force(TARGET_DIR)
    _rmtree_force(PYINSTALLER_ROOT)
    for old_zip in BUILD_ROOT.glob("convex_hull-*.zip"):
        old_zip.unlink(missing_ok=True)

    PYINSTALLER_ROOT.mkdir(parents=True, exist_ok=True)

    _write_entrypoint()
    _write_no_mypy_runtime_hook()

    pyinstaller_args = [
        "--noconfirm",
        "--clean",
        "--onedir",
        "--windowed",
        "--name",
        "convex_hull",
        "--distpath",
        str(BUILD_ROOT),
        "--workpath",
        str(PYINSTALLER_ROOT / "work"),
        "--specpath",
        str(PYINSTALLER_ROOT / "spec"),
        "--paths",
        str(PROJECT_ROOT / "src"),
        "--add-data",
        _add_data_arg(
            PROJECT_ROOT / "convex_hull" / "src" / "chan.py",
            "convex_hull/src",
        ),
        "--add-data",
        _add_data_arg(
            PROJECT_ROOT / "convex_hull" / "src" / "bruteforce_degenerate.py",
            "convex_hull/src",
        ),
        "--hidden-import",
        "app.hull_worker",
        "--hidden-import",
        "app.startup_profile",
        # Minimum VTK rendering modules not always auto-traced by PyInstaller hooks.
        "--hidden-import",
        "vtkmodules.vtkRenderingOpenGL2",
        "--hidden-import",
        "vtkmodules.vtkInteractionStyle",
        "--hidden-import",
        "vtkmodules.vtkRenderingFreeType",
        "--runtime-hook",
        str(NO_MYPY_RUNTIME_HOOK),
        # Dev / build tools — never needed at runtime.
        "--exclude-module",
        "mypy",
        "--exclude-module",
        "mypyc",
        "--exclude-module",
        "pytest",
        "--exclude-module",
        "hypothesis",
        "--exclude-module",
        "ruff",
        "--exclude-module",
        "scalene",
        "--exclude-module",
        "pre_commit",
        # scipy is bench-only (used only by SciPyOracle3D, not by the GUI).
        "--exclude-module",
        "scipy",
        # VTK test / VR modules — not needed for desktop GUI.
        "--exclude-module",
        "vtkmodules.test",
        "--exclude-module",
        "vtkmodules.vtkTestingCore",
        "--exclude-module",
        "vtkmodules.vtkTestingRendering",
        "--exclude-module",
        "vtkmodules.vtkRenderingVR",
        "--exclude-module",
        "vtkmodules.vtkRenderingVRModels",
        "--exclude-module",
        "vtkmodules.vtkWebCore",
        str(ENTRYPOINT),
    ]

    return_code = _run_streamed(
        [sys.executable, "-m", "PyInstaller", *pyinstaller_args],
        log,
    )

    if return_code != 0:
        log.line(
            f"PyInstaller failed with exit code {return_code}",
            stream=sys.stderr,
        )
        return return_code

    exe_name = "convex_hull.exe" if os.name == "nt" else "convex_hull"
    exe_path = TARGET_DIR / exe_name

    if not exe_path.exists():
        log.line(
            f"Build finished, but expected executable was not found: {exe_path}",
            stream=sys.stderr,
        )
        return 1

    _copy_external_algorithm_sources(log)

    version = _git_version()
    zip_stem = f"convex_hull-{version}"
    zip_path = BUILD_ROOT / f"{zip_stem}.zip"

    log.line()
    log.line(f"Creating zip archive: {zip_path.name}")

    shutil.make_archive(
        base_name=str(BUILD_ROOT / zip_stem),
        format="zip",
        root_dir=BUILD_ROOT,
        base_dir="convex_hull",
    )

    if not zip_path.exists():
        log.line(
            f"Archive was not created: {zip_path}",
            stream=sys.stderr,
        )
        return 1

    log.line()
    log.line(f"Build finished: {exe_path}")
    log.line(f"Archive created: {zip_path}")
    log.line(f"Full log:        {log.path}")

    return 0


def main() -> int:
    """Build the PyInstaller bundle and return an exit code."""
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = OUT_ROOT / f"build_{timestamp}.log"

    with TeeLog(log_path) as log:
        try:
            return _build(log)
        except Exception:
            log.line("\nUnhandled build script error:", stream=sys.stderr)
            log.write(traceback.format_exc(), stream=sys.stderr)
            log.line(f"Full log: {log.path}", stream=sys.stderr)
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
