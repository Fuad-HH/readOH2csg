import importlib.resources
import sys
from ctypes import CDLL, RTLD_GLOBAL
from pathlib import Path

assert sys.platform == "linux"  # only works with linux

_lib_dir = Path(str(importlib.resources.files(__package__) / "lib"))

# Pre-load bundled dependency shared libraries with RTLD_GLOBAL so their
# symbols and SONAMEs are registered before libomegah2csg.so is loaded.
# Load order matters: kokkoscore -> kokkoscontainers -> omega_h
_dep_patterns = ["libkokkoscore.so*", "libkokkoscontainers.so*", "libomega_h.so*"]
_loaded_deps = []
for _pattern in _dep_patterns:
    for _lib_path in sorted(_lib_dir.glob(_pattern)):
        _loaded_deps.append(CDLL(str(_lib_path), mode=RTLD_GLOBAL))
        break

_filename = _lib_dir / "libomegah2csg.so"
_dll = CDLL(str(_filename))
