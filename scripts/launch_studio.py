#!/usr/bin/env python3
"""Shared bootstrap for desktop launchers."""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

REQUIRED_MODULES = ("PySide6", "numpy", "pandas", "matplotlib", "sklearn")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def applescript_quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def show_error(title: str, message: str) -> None:
    if sys.platform == "darwin" and shutil.which("osascript"):
        script = (
            f"display dialog {applescript_quote(message)} "
            f'buttons {{"OK"}} default button "OK" '
            f"with title {applescript_quote(title)} with icon stop"
        )
        subprocess.run(["osascript", "-e", script], check=False)
        return

    if os.name == "nt":
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(0, message, title, 0x10)
            return
        except Exception:
            pass

    for command in (
        ["zenity", "--error", "--title", title, "--width", "420", "--text", message],
        ["kdialog", "--error", message, "--title", title],
    ):
        if shutil.which(command[0]):
            subprocess.run(command, check=False)
            return

    print(f"{title}: {message}", file=sys.stderr)


def missing_modules() -> list[str]:
    missing: list[str] = []
    for module_name in REQUIRED_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception:
            missing.append(module_name)
    return missing


def dependency_message(root: Path, missing: list[str]) -> str:
    joined = ", ".join(missing)
    return (
        "ML-Teaching Studio could not start because some Python packages are missing.\n\n"
        f"Missing modules: {joined}\n\n"
        "Install the project dependencies first, ideally inside a local virtual environment.\n"
        f"Project root: {root}\n"
        "Suggested commands:\n"
        "python3 -m venv .venv\n"
        "source .venv/bin/activate\n"
        "pip install -r requirements.txt"
    )


def bootstrap_src(root: Path) -> None:
    src_dir = root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "ml_teaching_studio_mpl_cache"),
    )
    os.chdir(root)


def run(check_only: bool = False) -> int:
    root = project_root()
    bootstrap_src(root)

    missing = missing_modules()
    if missing:
        message = dependency_message(root, missing)
        if check_only:
            print(message)
        else:
            show_error("ML-Teaching Studio", message)
        return 1

    try:
        from ml_teaching_studio.main import main as app_main
    except Exception as exc:
        message = (
            "ML-Teaching Studio failed during startup import.\n\n"
            f"{exc}\n\n"
            f"Project root: {root}"
        )
        if check_only:
            print(message)
            traceback.print_exc()
        else:
            show_error("ML-Teaching Studio", message)
            traceback.print_exc()
        return 1

    if check_only:
        print("Launcher check passed.")
        print(f"Project root: {root}")
        print(f"Interpreter: {sys.executable}")
        return 0

    try:
        return int(app_main())
    except SystemExit as exc:
        return int(exc.code or 0)
    except Exception as exc:
        message = (
            "ML-Teaching Studio started but then exited with an error.\n\n"
            f"{exc}\n\n"
            f"Interpreter: {sys.executable}"
        )
        show_error("ML-Teaching Studio", message)
        traceback.print_exc()
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Launcher bootstrap for ML-Teaching Studio.")
    parser.add_argument("--check", action="store_true", help="Validate the launcher environment without opening the GUI.")
    args = parser.parse_args()
    return run(check_only=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
