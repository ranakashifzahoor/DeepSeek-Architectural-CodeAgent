"""
build_exe.py - Package the CodeAgent_DeepSeek desktop app with PyInstaller.

Bundles ``main_gui.py`` together with ``parser.py``, ``agent_core.py`` and all
third-party dependencies (CustomTkinter themes/fonts, requests, the Python
runtime, Tcl/Tk libraries) into a single double-clickable executable:

    dist/DeepSeek_CodeAgent.exe

The app is built windowed (``--noconsole``), so no terminal window flashes up
when the executable is launched.

Usage::

    python build_exe.py             # windowed onefile build (normal use)
    python build_exe.py --console   # keep a console window (debugging)
    python build_exe.py --check     # print the PyInstaller command only
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
ENTRY_SCRIPT = PROJECT_DIR / "main_gui.py"
APP_NAME = "DeepSeek_CodeAgent"
DIST_DIR = PROJECT_DIR / "dist"
EXE_PATH = DIST_DIR / f"{APP_NAME}.exe"


def build_pyinstaller_args(*, windowed: bool) -> list[str]:
    """Return the PyInstaller argument list for the build."""
    args = [
        str(ENTRY_SCRIPT),
        "--name", APP_NAME,
        "--onefile",
        "--noconfirm",
        "--clean",
        # Keep the local modules resolvable no matter where the build runs from.
        "--paths", str(PROJECT_DIR),
        "--hidden-import", "parser",
        "--hidden-import", "agent_core",
        # CustomTkinter ships JSON themes + fonts that must be bundled.
        "--collect-all", "customtkinter",
    ]
    args.append("--windowed" if windowed else "--console")
    return args


def preflight_problems() -> list[str]:
    """Check that the current interpreter has everything the build needs."""
    problems: list[str] = []
    for module in ("tkinter", "customtkinter", "requests", "PyInstaller"):
        try:
            __import__(module)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{module}: {exc}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"Build {APP_NAME}.exe")
    parser.add_argument(
        "--console",
        action="store_true",
        help="build with a console window (useful for debugging)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="print the command without running it",
    )
    args = parser.parse_args(argv)

    if not ENTRY_SCRIPT.is_file():
        print(f"[build_exe] ERROR: entry script not found: {ENTRY_SCRIPT}")
        return 2

    problems = preflight_problems()
    if problems:
        print("[build_exe] ERROR: this Python is missing build dependencies:")
        for problem in problems:
            print(f"  - {problem}")
        print("[build_exe] Install them first:  python -m pip install -r requirements.txt")
        return 2

    cmd = [
        sys.executable,
        "-m", "PyInstaller",
        *build_pyinstaller_args(windowed=not args.console),
    ]
    print("[build_exe] command:")
    print("  " + " ".join(f'"{part}"' if " " in part else part for part in cmd))

    if args.check:
        return 0

    os.chdir(PROJECT_DIR)  # PyInstaller's default build/dist paths are CWD-relative
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[build_exe] ERROR: PyInstaller exited with code {result.returncode}")
        return 1

    if not EXE_PATH.is_file():
        print(f"[build_exe] ERROR: build finished but {EXE_PATH} was not created")
        return 1

    size_mb = EXE_PATH.stat().st_size / (1024 * 1024)
    print(f"[build_exe] SUCCESS: {EXE_PATH} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
