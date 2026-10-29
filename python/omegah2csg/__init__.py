"""Convert Omega_h tokamak meshes to CSG geometry for OpenMC and DEGAS2.

All computation runs in C++ (Kokkos, OpenMP or CUDA); this package exposes the
results as NumPy arrays that can be edited before building OpenMC objects or
writing files.
"""

from ._core import (
    BoundaryType,
    CsgGeometry,
    Degas2Geometry,
    Degas2Options,
    GeometryError,
    InputError,
    NcFormat,
    OpenMCOptions,
    SurfaceType,
    __version__,
    compute_csg,
    compute_degas2,
    convert_to_degas2,
    convert_to_openmc,
    initialize,
    show_config,
    write_degas2,
    write_openmc_xml,
)


def to_openmc_universe(geometry, **kwargs):
    """Build an ``openmc.Universe`` from a CsgGeometry (needs ``openmc``).

    See :func:`omegah2csg.openmc_builder.to_openmc_universe`.
    """
    from .openmc_builder import to_openmc_universe as build

    return build(geometry, **kwargs)


__all__ = [
    "BoundaryType",
    "CsgGeometry",
    "Degas2Geometry",
    "Degas2Options",
    "GeometryError",
    "InputError",
    "NcFormat",
    "OpenMCOptions",
    "SurfaceType",
    "__version__",
    "compute_csg",
    "compute_degas2",
    "convert_to_degas2",
    "convert_to_openmc",
    "initialize",
    "show_config",
    "to_openmc_universe",
    "write_degas2",
    "write_openmc_xml",
]
