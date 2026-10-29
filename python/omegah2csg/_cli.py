"""Console scripts: run the bundled native executable (one CLI implementation)."""

import os
import sys
from pathlib import Path


def _find(name):
    from . import _core  # bin/ is installed next to the compiled module

    bundled = Path(_core.__file__).with_name("bin") / name
    if bundled.is_file():
        return bundled
    me = Path(sys.argv[0]).resolve()
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        exe = Path(directory) / name
        if exe.is_file() and os.access(exe, os.X_OK) and exe.resolve() != me:
            return exe
    sys.exit(
        f"{name}: native executable not found; reinstall omegah2csg or build it with CMake"
    )


def _run(name):
    exe = _find(name)
    os.execv(exe, [str(exe), *sys.argv[1:]])


def openmc_main():
    _run("convert2openmc")


def degas2_main():
    _run("convert2degas2")
