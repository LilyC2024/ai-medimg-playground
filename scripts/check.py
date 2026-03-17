from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str], *, required: bool = True) -> int:
    print(f"$ {' '.join(command)}")
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if required and completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return completed.returncode


def main() -> int:
    formatter = shutil.which("ruff")
    if formatter:
        _run([formatter, "format", "--check", "."])
    else:
        print("ruff not installed; skipping formatting check.")

    python_executable = sys.executable
    _run([python_executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
