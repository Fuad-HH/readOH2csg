"""Build OpenMC objects from a CsgGeometry.

Uses the same surface/cell rules and ids as the C++ ``geometry.xml`` writer:
surface of edge ``e`` has id ``e + 1 + surface_id_offset`` (a cone's
disambiguation plane ``nedges + e + 1 + surface_id_offset``) and the cell of
face ``f`` has id ``f + 1 + cell_id_offset``. Building Python objects costs
time per cell; for very large meshes write ``geometry.xml`` from C++ instead.
"""

import numpy as np

from ._core import SurfaceType

_BOUNDARY = ("transmission", "vacuum", "reflective", "white")


def _surfaces(geom, boundary, offset):
    import openmc

    nedges = geom.nedges
    out = []
    for e, (kind, c) in enumerate(
        zip(geom.surface_type, geom.surface_coeffs, strict=True)
    ):
        sid, bc = e + 1 + offset, _BOUNDARY[int(boundary[e])]
        if kind == int(SurfaceType.ZPlane):
            s = openmc.ZPlane(z0=-c[3], surface_id=sid, boundary_type=bc)
        elif kind == int(SurfaceType.ZCylinder):
            s = openmc.ZCylinder(r=np.sqrt(-c[3]), surface_id=sid, boundary_type=bc)
        else:  # one-sided cone with explicit ids; the plane gets no boundary
            s = openmc.model.ZConeOneSided.__new__(openmc.model.ZConeOneSided)
            s.cone = openmc.ZCone(
                z0=c[2] / 2, r2=1 / abs(c[0]), surface_id=sid, boundary_type=bc
            )
            s.plane = openmc.ZPlane(z0=c[2] / 2, surface_id=nedges + e + 1 + offset)
            s.up = bool(c[4] > 0)
        out.append(s)
    return out


def _fill(geom, materials):
    """Material (or None for void) per cell."""
    if materials is None or not isinstance(materials, (dict, list, tuple, np.ndarray)):
        return [materials] * geom.nfaces
    if isinstance(materials, dict):
        return [materials.get(int(tag)) for tag in geom.face_region_tag]
    if len(materials) != geom.nfaces:
        raise ValueError(f"materials has {len(materials)} entries, need {geom.nfaces}")
    return list(materials)


def to_openmc_universe(
    geom,
    *,
    materials=None,
    boundary_type=None,
    cell_id_offset=0,
    surface_id_offset=0,
    universe_id=None,
):
    """Build an ``openmc.Universe`` with one cell per mesh face.

    Parameters
    ----------
    geom : CsgGeometry
        Result of ``compute_csg`` (arrays may have been edited).
    materials : None, openmc.Material, dict or sequence
        ``None``: void cells. A Material: every cell. A dict: offset_face tag
        value -> Material. A sequence: one Material (or None) per cell.
    boundary_type : str, optional
        Override for the mesh-boundary surfaces; default ``geom.surface_boundary``.
    cell_id_offset, surface_id_offset : int
        Added to every cell / surface id so the universe can be embedded.
    universe_id : int, optional
    """
    try:
        import openmc
    except ImportError as err:
        raise ImportError(
            "to_openmc_universe needs openmc: pip install openmc"
        ) from err

    boundary = np.array(geom.surface_boundary, copy=True)
    if boundary_type is not None:
        boundary[:] = 0
        boundary[geom.boundary_edges] = _BOUNDARY.index(boundary_type)
    surfaces = _surfaces(geom, boundary, surface_id_offset)
    fills = _fill(geom, materials)
    cells = []
    for f, (edges, sides) in enumerate(
        zip(geom.face_edges, geom.face_sides, strict=True)
    ):
        halves = [
            +surfaces[e] if s > 0 else -surfaces[e]
            for e, s in zip(edges, sides, strict=True)
        ]
        region = halves[0] & halves[1] & halves[2]
        cells.append(
            openmc.Cell(cell_id=f + 1 + cell_id_offset, region=region, fill=fills[f])
        )
    return openmc.Universe(universe_id=universe_id, cells=cells)
