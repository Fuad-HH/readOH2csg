"""
This module provides bindings to C/C++ functions defined in the C++ library.

.. code-block:: python

    openmcGeometry.readCoefficients()
"""

import numpy as np
from numpy.ctypeslib import ndpointer
import openmc

from .OmegaHMesh import OmegaHMeshPointer, OmegaHMesh
from ctypes import c_int, c_double, c_bool
from . import _dll, kokkos_runtime

_dll.capi_compute_edge_coefficients.restype = None
_dll.capi_compute_edge_coefficients.argtypes = [OmegaHMeshPointer, c_int, ndpointer(c_double), c_bool]

_dll.capi_get_boundary_edge_ids.restype = None
_dll.capi_get_boundary_edge_ids.argtypes = [OmegaHMeshPointer, c_int, ndpointer(c_int)]

_dll.capi_get_number_of_boundary_edges.restype = c_int
_dll.capi_get_number_of_boundary_edges.argtypes = [OmegaHMeshPointer]

_dll.capi_get_face_connectivity.restype = None
_dll.capi_get_face_connectivity.argtypes = [OmegaHMeshPointer, c_int, ndpointer(c_double), c_int, ndpointer(c_int), c_bool]

_dll.capi_get_all_geometry_info.restype = None
_dll.capi_get_all_geometry_info.argtypes = [OmegaHMeshPointer, c_int, c_int, ndpointer(c_double), ndpointer(c_int), ndpointer(c_int), c_bool]


def get_edge_coefficients(mesh: OmegaHMesh, print_debug=False):
    if not kokkos_runtime.is_running():
        raise RuntimeError("Kokkos not running...")

    num_edges = mesh.num_entities(1)
    coefficients = np.zeros(num_edges * 6, dtype=np.float64)
    size = coefficients.size

    try:
        _dll.capi_compute_edge_coefficients(mesh.mesh, c_int(size), coefficients, print_debug)

    except Exception as exception:
        raise RuntimeError(f"Error computing edge coefficients: {exception}")

    return coefficients.reshape(num_edges, 6)

def get_num_of_boundary_edges(mesh: OmegaHMesh) -> int :
    if not kokkos_runtime.is_running():
        raise RuntimeError("Kokkos not running...")

    try:
        return _dll.capi_get_number_of_boundary_edges(mesh.mesh)

    except Exception as exception:
        raise RuntimeError(f"Error computing number of boundary edges: {exception}")

def get_boundary_edge_ids(mesh: OmegaHMesh, size=0):
    if not kokkos_runtime.is_running():
        raise RuntimeError("Kokkos not running...")

    if size == 0:
        size = get_num_of_boundary_edges(mesh)

    boundary_edge_ids = np.zeros(size, dtype=np.int32)
    try:
        _dll.capi_get_boundary_edge_ids(mesh.mesh, size, boundary_edge_ids)
    except Exception as exception:
        raise RuntimeError(f"Error computing boundary edge ids: {exception}")

    return boundary_edge_ids

def get_face_connectivity(mesh: OmegaHMesh, edge_coefficients= None, print_debug=False):
    if not kokkos_runtime.is_running():
        raise RuntimeError("Kokkos not running...")

    n_faces = mesh.num_entities(2)
    n_edges = mesh.num_entities(1)


    if edge_coefficients is None:
        edge_coefficients = get_edge_coefficients(mesh)

    assert edge_coefficients.shape[0] == n_edges
    assert edge_coefficients.shape[1] == 6
    assert edge_coefficients.dtype == np.float64

    edge_coefficients_shaped = edge_coefficients.reshape(n_edges*6)

    face_connectivity = np.zeros(n_faces * 6, dtype=np.int32)


    # TODO add core-dump capture
    try:
        _dll.capi_get_face_connectivity(mesh.mesh, n_edges * 6, edge_coefficients_shaped, n_faces*6, face_connectivity, print_debug)
    except Exception as exception:
        raise RuntimeError(f"Error computing face connectivity: {exception}")

    return face_connectivity.reshape(n_faces, 6)


def get_all_geometry_info(mesh: OmegaHMesh, print_debug=False):
    if not kokkos_runtime.is_running():
        raise RuntimeError("Kokkos not running...")

    n_faces = mesh.num_entities(2)
    n_edges = mesh.num_entities(1)

    edge_coefficients = np.zeros(n_edges * 6, dtype=np.float64)
    boundary_edge_ids_num = get_num_of_boundary_edges(mesh)
    boundary_edge_ids = np.zeros(boundary_edge_ids_num, dtype=np.int32)
    face_connctivity = np.zeros(n_faces * 6, dtype=np.int32)

    try:
        _dll.capi_get_all_geometry_info(mesh.mesh, n_edges, n_faces, edge_coefficients, boundary_edge_ids, face_connctivity, print_debug)
    except Exception as exception:
        raise RuntimeError(f"Error computing all geometry info: {exception}")

    return edge_coefficients.reshape(n_edges, 6), boundary_edge_ids, face_connctivity.reshape(n_faces, 6)

def create_openmc_geometry(mesh: OmegaHMesh, print_debug=False):
    if not kokkos_runtime.is_running():
        raise RuntimeError("Kokkos not running...")

    [edge_coefficients, boundary_edge_ids, face_connctivity] = get_all_geometry_info(mesh, print_debug)

    top_bottom_flag = edge_coefficients[:, 5]
    edges = []
    intersections = edge_coefficients[:, 3]/2.0
    m2 = edge_coefficients[:, 1]
    z2 = edge_coefficients[:, 2]
    neg_c = edge_coefficients[:, 4]

    for i in range(len(intersections)):

        if abs(m2[i]) < 1e-10:  # zplane
            #print("Zplane - id ", int(data[i][0]))
            edges.append(openmc.ZPlane(z0=-neg_c[i]))
        elif abs(z2[i] + 1) > 1e-10:  # not a cone
            #print("Quad - id ", int(data[i][0]))
            edges.append(openmc.Quadric(a=m2[i], b=m2[i], c=z2[i], j=intersections[i], k=neg_c[i]))
        else:  # cone
            #print("Cone - id ", int(data[i][0]), " flag ", top_bottom_flag[i])
            edges.append(openmc.model.ZConeOneSided(z0=intersections[i], r2=1.0 / abs(m2[i]),
                                                    up=True if top_bottom_flag[i] == 1 else False))

    # this is incorrect as the above is appending
    for edge_id in boundary_edge_ids:
        edges[edge_id].boundary_type = 'reflective'

    fuel = openmc.Material(name='plasma')

    cells = []
    for i in range(face_connctivity.shape[0]):
        vol1 = +edges[int(face_connctivity[i][0])] if int(face_connctivity[i][1]) == 1 else -edges[int(face_connctivity[i][0])]
        vol2 = +edges[int(face_connctivity[i][2])] if int(face_connctivity[i][3]) == 1 else -edges[int(face_connctivity[i][2])]
        vol3 = +edges[int(face_connctivity[i][4])] if int(face_connctivity[i][5]) == 1 else -edges[int(face_connctivity[i][4])]

        cells.append(openmc.Cell(region=vol1 & vol2 & vol3, fill=fuel, name='cell' + str(i), cell_id=i))

    universe = openmc.Universe(cells=cells)
    return universe


def read_edge_coefficients_from_file(filename):
    with open(filename, 'r') as f:
        data = f.readlines()
        n_edges = len(data) - 2

        # this file has first 5 coefficients
        edge_coefficients = np.zeros((n_edges, 6), dtype=np.float64)

        for i in range(n_edges):
            values = data[i].split()
            assert len(values) == 6, f"Length of edge_coefficients is not 6: {values}"

            edge_id = int(values[0])
            assert edge_id == i, f"Edge id is not {i}, edge id {edge_id}"

            # fill the 5 coefficients
            edge_coefficients[edge_id, 0] = np.float64(values[1])
            edge_coefficients[edge_id, 1] = np.float64(values[2])
            edge_coefficients[edge_id, 2] = np.float64(values[3])
            edge_coefficients[edge_id, 3] = np.float64(values[4])
            edge_coefficients[edge_id, 4] = np.float64(values[5])

        boundary_edges_line = data[-2].split()
        assert len(boundary_edges_line) == 3
        num_boundary_edges = int(boundary_edges_line[-1])
        boundary_edge_ids = np.zeros(num_boundary_edges, dtype=np.int32)

        boundary_edge_ids_line = data[-1].split()
        assert len(boundary_edge_ids_line) == num_boundary_edges
        for i in range(num_boundary_edges):
            boundary_edge_ids[i] = int(boundary_edge_ids_line[i])


    return edge_coefficients, boundary_edge_ids

def read_face_connectivity_from_file(filename):
    with open(filename, 'r') as f:
        data = f.readlines()
        n_faces = len(data)
        connectivity = np.zeros((n_faces, 6), dtype=np.int32)

        for i in range(n_faces):
            values = data[i].split()
            assert len(values) == 7
            face_id = int(values[0])
            assert face_id == i

            for j in range(6):
                connectivity[i, j] = np.int32(values[j+1])

    return connectivity

