"""
This module provides functions for converting Omega_h meshes to OpenMC geometry.

.. code-block:: python

    convert2openmc.create_openmc_geometry(mesh)
"""

import numpy as np
import openmc
from typing import Tuple

from ..OmegaHMesh import EdgeType, OmegaHMesh
from ..config import kokkos_runtime


Coord = Tuple[float, float]


def is_horizontal(p1: Coord, p2: Coord, tol=1e-6) -> bool:
    return np.abs(p1[1] - p2[1]) < tol


def is_vertical(p1: Coord, p2: Coord, tol=1e-6) -> bool:
    return np.abs(p1[0] - p2[0]) < tol


def get_slope(p1: Coord, p2: Coord, tol=1e-6):
    if is_horizontal(p1, p2, tol):
        return 0
    if is_vertical(p1, p2, tol):
        return np.nan

    m = (p2[1] - p1[1]) / (p2[0] - p1[0])
    return m


def get_vertical_intersection(p1: Coord, p2: Coord, tol=1e-6):
    if is_horizontal(p1, p2, tol):
        return p1[1]
    if is_vertical(p1, p2, tol):
        return np.nan

    m = get_slope(p1, p2, tol)
    c = p1[1] - m * p1[0]
    return c


def get_line_equation(p1: Coord, p2: Coord, tol=1e-6) -> Tuple:
    """
    Returns m,c for a line equation
    :param p1: Point 1 Coordinates
    :param p2: Point 2 Coordinates
    :param tol: Tolerance
    :return: m, c, up: Slope, Intersection, Up Flag for cone
    """
    if is_horizontal(p1, p2, tol):
        return 0, p1[1], np.nan
    if is_vertical(p1, p2, tol):
        return np.inf, np.nan, np.nan

    m = get_slope(p1, p2, tol)
    c = p1[1] - m * p1[0]

    # see doc for figure
    up = p1[1] > c and p2[1] > c
    # both should work since both points are on the right of z
    assert up == (p1[1] > c or p2[1] > c)
    return m, c, up


def is_cone(p1: Coord, p2: Coord, tol=1e-6) -> bool:
    return not is_horizontal(p1, p2, tol) and not is_vertical(p1, p2, tol)


def create_z_cone(p1: Coord, p2: Coord, tol=1e-6) -> openmc.model.ZConeOneSided:
    assert not is_horizontal(p1, p2, tol), (
        "Horizonal line. Use create_z_plane instead of create_z_cone."
    )
    assert not is_vertical(p1, p2, tol), (
        "Vertical line. Use create_z_cylinder instead of create_z_cone."
    )

    m, c, up = get_line_equation(p1, p2, tol)
    assert not np.isinf(m), "Slope m found to be infinity."
    assert not np.isnan(c), "Slope c found to be nan."

    cone = openmc.model.ZConeOneSided(x0=0, y0=0, z0=c, r2=m * m, up=up)
    return cone


def create_z_plane(p1: Coord, p2: Coord, tol=1e-6) -> openmc.ZPlane:
    assert is_horizontal(p1, p2, tol), "Not a horizontal line"
    return openmc.ZPlane(z0=p1[1])


def create_z_cylinder(p1: Coord, p2: Coord, tol=1e-6) -> openmc.ZCylinder:
    assert is_vertical(p1, p2, tol), "Not a vertical line"
    return openmc.ZCylinder(r=p1[0])


def create_openmc_surface(p1: Coord, p2: Coord, tol=1e-6):
    if is_horizontal(p1, p2, tol):
        return create_z_plane(p1, p2, tol)
    if is_vertical(p1, p2, tol):
        return create_z_cylinder(p1, p2, tol)
    if is_cone(p1, p2, tol):
        return create_z_cone(p1, p2, tol)

    raise RuntimeError(f"Error creating surface: {p1} and {p2}")


def create_openmc_universe(
    mesh: OmegaHMesh, materials=None, print_debug=False, tol=1e-6
):
    """Create OpenMC geometry (OpenMC.Universe) from OmegaHMesh by rorating along Z-axis

    Parameters
    ----------
    mesh: OmegaHMesh
        OmegaHMesh object with isOnWall and offset_face tags
    materials: Any interable of size of number of cells, optional
        Optional array of OpenMC materials to fill the cells. Size should match number of faces in the mesh.
        If None, dummy materials will be used to avoid OpenMC errors.
    print_debug: boo, optional
        If True, prints edge coefficients and face connectivity for debugging.
    tol: float, optional
        Tolerance for determining edge types and connectivity. Should be a small positive number, e.g. 1e-6.

    Returns
    -------
    OpenMC.Universe
        OpenMC Universe object representing the geometry

    """
    if not kokkos_runtime.is_running():
        raise RuntimeError("Kokkos not running...")

    [edge_coefficients, edge_types, boundary_edge_ids, face_connctivity] = (
        mesh.get_all_geometry_info(print_debug, tol)
    )
    n_edges = mesh.num_entities(1)
    n_faces = mesh.num_entities(2)

    top_bottom_flag = edge_coefficients[:, 4]
    edges = np.empty(shape=n_edges, dtype=object)
    intersections = edge_coefficients[:, 2] / 2.0
    m2 = edge_coefficients[:, 0]
    z2 = edge_coefficients[:, 1]
    neg_c = edge_coefficients[:, 3]

    for i in range(len(intersections)):
        if edge_types[i] == EdgeType.Z_PLANE:
            edges[i] = openmc.ZPlane(z0=-neg_c[i])
        elif edge_types[i] == EdgeType.Z_CYLINDER:
            edges[i] = openmc.ZCylinder(r=np.sqrt(-neg_c[i]))
        elif edge_types[i] == EdgeType.Z_CONE:
            edges[i] = openmc.model.ZConeOneSided(
                z0=intersections[i],
                r2=1.0 / abs(m2[i]),
                up=True if top_bottom_flag[i] == 1 else False,
            )
        else:
            raise RuntimeError(
                f"Z2 value {z2[i]} not recognized. Coefficients: {edge_coefficients[i, :]}"
            )

    for edge_id in boundary_edge_ids:
        edges[edge_id].boundary_type = "reflective"

    if materials is None:
        # to stop openmc errors
        dummy = openmc.Material(name="dummy")
        dummy.add_nuclide("H1", 1.0)
        dummy.set_density(units="g/cc", density=1.0)
        materials = np.array([dummy] * n_faces)

    assert materials.ndim == 1
    assert materials.shape[0] == n_faces

    cells = np.empty(shape=n_faces, dtype=object)
    for i in range(face_connctivity.shape[0]):
        vol1 = (
            +edges[int(face_connctivity[i][0])]
            if int(face_connctivity[i][1]) == 1
            else -edges[int(face_connctivity[i][0])]
        )
        vol2 = (
            +edges[int(face_connctivity[i][2])]
            if int(face_connctivity[i][3]) == 1
            else -edges[int(face_connctivity[i][2])]
        )
        vol3 = (
            +edges[int(face_connctivity[i][4])]
            if int(face_connctivity[i][5]) == 1
            else -edges[int(face_connctivity[i][4])]
        )

        cells[i] = openmc.Cell(
            region=vol1 & vol2 & vol3,
            fill=materials[i],
            name="cell" + str(i),
            cell_id=i,
        )

    universe = openmc.Universe(cells=cells)
    return universe


def convert2openmcXML(filename, tol):
    assert filename.endswith(".osh")
    assert (tol < 1e-6) and (tol > 0.0)

    with OmegaHMesh(filename) as mesh:
        universe = create_openmc_universe(mesh=mesh, tol=tol)
        geom = openmc.Geometry(universe)
        geom.export_to_xml()
