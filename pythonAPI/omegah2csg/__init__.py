from importlib import metadata
from pathlib import Path
import re

from .config import KokkosRuntime, kokkos_runtime
from .OmegaHMesh import (
    OmegaHMesh,
    read_edge_coefficients_from_file,
    read_face_connectivity_from_file,
)
from .convert2openmc import (
    convert2openmcXML,
)
from .convert2degas2 import convert2degas2


def _read_cmake_version() -> str | None:
    cmake_path = Path(__file__).resolve().parents[2] / "CMakeLists.txt"
    if not cmake_path.is_file():
        return None
    content = cmake_path.read_text(encoding="utf-8")
    match = re.search(
        r"project\([^)]*?\bVERSION\s+([0-9]+\.[0-9]+\.[0-9]+)[^)]*\)",
        content,
        re.DOTALL,
    )
    return match.group(1) if match else None


def _get_version() -> str:
    try:
        return metadata.version("omegah2csg")
    except metadata.PackageNotFoundError:
        return _read_cmake_version() or "0.0.0"


__version__ = _get_version()

__all__ = [
    # runtime
    "KokkosRuntime",
    "kokkos_runtime",
    # mesh
    "OmegaHMesh",
    "read_edge_coefficients_from_file",
    "read_face_connectivity_from_file",
    # openmc
    "convert2openmcXML",
    # degas2
    "convert2degas2",
    # version
    "__version__",
]
