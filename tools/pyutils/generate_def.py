#!/usr/bin/env python3
"""Delegate NNTrainer's Windows DEF generation from a nested subproject.

NNTrainer's Windows Meson build expects this helper below the top-level
source root. Quick.AI keeps NNTrainer as a submodule, so this wrapper
preserves that expectation without changing the pinned submodule.
"""

from pathlib import Path
import runpy
import sys


def main() -> None:
    nntrainer_script = (
        Path(__file__).resolve().parents[2]
        / "subprojects"
        / "nntrainer"
        / "tools"
        / "pyutils"
        / "generate_def.py"
    )
    sys.argv[0] = str(nntrainer_script)
    runpy.run_path(str(nntrainer_script), run_name="__main__")


if __name__ == "__main__":
    main()
