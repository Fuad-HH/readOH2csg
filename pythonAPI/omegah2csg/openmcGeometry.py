"""
This module provides bindings to C/C++ functions defined in the C++ library.

.. code-block:: python

    openmcGeometry.readCoefficients()
"""

import numpy as np
from numpy.ctypeslib import ndpointer

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


