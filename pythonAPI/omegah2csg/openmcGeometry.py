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

