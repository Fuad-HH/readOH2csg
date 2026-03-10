from ctypes import c_void_p, c_char_p, Structure, c_int, c_bool, c_double

import numpy as np
from numpy.ctypeslib import ndpointer

from ._lib import _dll
from .config import kokkos_runtime
import os
from enum import IntEnum


# Define C structures
class OmegaHLibraryPointer(Structure):
    _fields_ = [("pointer", c_void_p)]


class OmegaHMeshPointer(Structure):
    _fields_ = [("pointer", c_void_p)]


class EdgeType(IntEnum):
    Z_PLANE = 1
    Z_CYLINDER = 2
    Z_CONE = 3


# Define function prototypes
_dll.create_omegah_library.restype = OmegaHLibraryPointer
_dll.create_omegah_library.argtypes = []

_dll.destroy_omegah_library.argtypes = [OmegaHLibraryPointer]

_dll.create_omegah_mesh.restype = OmegaHMeshPointer
_dll.create_omegah_mesh.argtypes = [OmegaHLibraryPointer, c_char_p]

_dll.destroy_omegah_mesh.argtypes = [OmegaHMeshPointer]

_dll.print_mesh_info.argtypes = [OmegaHMeshPointer]

_dll.get_num_entities.restype = c_int
_dll.get_num_entities.argtypes = [OmegaHMeshPointer, c_int]

_dll.get_dim.restype = c_int
_dll.get_dim.argtypes = [OmegaHMeshPointer]

_dll.capi_is_mesh_bounded_by_box.restype = c_bool
_dll.capi_is_mesh_bounded_by_box.argtypes = [OmegaHMeshPointer]

_dll.capi_has_boundary_layer.restype = c_bool
_dll.capi_has_boundary_layer.argtypes = [OmegaHMeshPointer]

_dll.capi_get_mesh_int_tag_array.restype = c_bool
_dll.capi_get_mesh_int_tag_array.argtypes = [
    OmegaHMeshPointer,
    c_int,
    c_char_p,
    ndpointer(c_int),
    c_int,
]

_dll.capi_get_cell_bounding_boxes.argtypes = [
    OmegaHMeshPointer,
    ndpointer(c_double),
    c_int,
]

_dll.capi_get_cell_volumes.argtypes = [OmegaHMeshPointer, ndpointer(c_double), c_int]

_dll.capi_get_cell_centroids.argtypes = [OmegaHMeshPointer, ndpointer(c_double), c_int]

_dll.capi_get_edge_coordinates.argtypes = [
    OmegaHMeshPointer,
    ndpointer(c_double),
    c_int,
]

_dll.capi_get_number_of_edges_inside_wall.argtypes = [OmegaHMeshPointer]
_dll.capi_get_number_of_edges_inside_wall.restype = c_int

_dll.capi_get_edge_to_face_connectivity.argtypes = [
    OmegaHMeshPointer,
    ndpointer(c_int),
    c_int,
]

_dll.capi_get_wall_adjacent_triangles.argtypes = [
    OmegaHMeshPointer,
    ndpointer(c_int),
    c_int,
]

_dll.capi_get_wall_edge_ids.argtypes = [OmegaHMeshPointer, ndpointer(c_int), c_int]

_dll.capi_get_edge_to_face_connectivity.argtypes = [
    OmegaHMeshPointer,
    ndpointer(c_int),
    c_int,
]

_dll.capi_get_node_coordinates.argtypes = [
    OmegaHMeshPointer,
    ndpointer(c_double),
    c_int,
]

_dll.capi_compute_edge_coefficients.restype = None
_dll.capi_compute_edge_coefficients.argtypes = [
    OmegaHMeshPointer,
    c_int,
    ndpointer(c_double),
    ndpointer(c_int),
    c_bool,
    c_double,
]

_dll.capi_get_boundary_edge_ids.restype = None
_dll.capi_get_boundary_edge_ids.argtypes = [OmegaHMeshPointer, c_int, ndpointer(c_int)]

_dll.capi_get_number_of_boundary_edges.restype = c_int
_dll.capi_get_number_of_boundary_edges.argtypes = [OmegaHMeshPointer]

_dll.capi_get_face_connectivity.restype = None
_dll.capi_get_face_connectivity.argtypes = [
    OmegaHMeshPointer,
    c_int,
    ndpointer(c_double),
    c_int,
    ndpointer(c_int),
    c_bool,
    c_double,
]

_dll.capi_get_all_geometry_info.restype = None
_dll.capi_get_all_geometry_info.argtypes = [
    OmegaHMeshPointer,
    c_int,
    c_int,
    ndpointer(c_double),
    ndpointer(c_int),
    ndpointer(c_int),
    ndpointer(c_int),
    c_bool,
    c_double,
]


class OmegaHMesh:
    """
    A context manager for loading and working with OmegaH meshes.

    Example usage:
     with OmegaHMesh(filename) as mesh:
         mesh = OmegaHMesh()
    """

    def __init__(self, filename):
        self.filename = str(filename)
        self.lib = None
        self.mesh = None
        if not kokkos_runtime.kokkos_initialized:
            kokkos_runtime.kokkos_initialize()

        # check if given directory exits
        if not os.path.isdir(filename):
            raise RuntimeError(f"OmegaHMesh filename {filename} does not exist.")

    def __enter__(self):
        # Create library
        self.lib = _dll.create_omegah_library()

        # Create mesh in try catch to capture error
        try:
            self.mesh = _dll.create_omegah_mesh(self.lib, self.filename.encode("utf-8"))
        except Exception as exception:
            raise RuntimeError(f"Error creating OmegaH mesh: {exception}")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clean up in reverse order
        if self.mesh:
            _dll.destroy_omegah_mesh(self.mesh)
        if self.lib:
            _dll.destroy_omegah_library(self.lib)

    def print_info(self):
        """Print information about the loaded mesh."""
        if self.mesh is None:
            raise RuntimeError("No mesh loaded. Use as context manager.")
        print(f"Mesh information for: {self.filename}")
        _dll.print_mesh_info(self.mesh)

    def num_entities(self, dim):
        """Number of entities of given dimension."""
        if self.mesh is None:
            raise RuntimeError("No mesh loaded. Use as context manage.")
        return _dll.get_num_entities(self.mesh, dim)

    @property
    def dim(self):
        """Dimension of given mesh."""
        if self.mesh is None:
            raise RuntimeError("No mesh loaded. Use as context manager.")
        return _dll.get_dim(self.mesh)

    def get_node_coordinates(self) -> np.ndarray:
        if not kokkos_runtime.is_running():
            raise RuntimeError("Kokkos not running...")

        try:
            num_nodes = self.num_entities(0)
            size = 2 * num_nodes
            coordinates = np.empty(size, dtype=np.float64)
            _dll.capi_get_node_coordinates(self.mesh, coordinates, size)
            return coordinates
        except Exception as exception:
            raise RuntimeError(f"Error getting node coordinates: {exception}")

    def is_bounded_by_box(self) -> bool:
        if not kokkos_runtime.is_running():
            raise RuntimeError("Kokkos not running...")

        try:
            return _dll.capi_is_mesh_bounded_by_box(self.mesh)
        except Exception as exception:
            raise RuntimeError(f"Error finding box: {exception}")

    def has_boundary_layer(self) -> bool:
        """
        Checks if the mesh has a single boundary layer after the wall
        :return: If the input mesh has a single boundary layer after the wall
        """
        if not kokkos_runtime.is_running():
            raise RuntimeError("Kokkos not running...")

        try:
            return _dll.capi_has_boundary_layer(self.mesh)
        except Exception as exception:
            raise RuntimeError(f"Error finding layer: {exception}")

    def get_integer_tag_array(self, dim, name) -> np.ndarray:
        if not kokkos_runtime.is_running():
            raise RuntimeError("Kokkos not running...")

        try:
            array_size = self.num_entities(dim)
            tag_array = np.empty(array_size, dtype=np.int32)
            success = _dll.capi_get_mesh_int_tag_array(
                self.mesh, dim, str(name).encode("utf-8"), tag_array, array_size
            )
            assert success, "Internal error"
            return tag_array
        except Exception as exception:
            raise RuntimeError(f"Error getting integer tag: {exception}")

    def get_boundary_face_flag(self) -> np.ndarray:
        return self.get_integer_tag_array(2, "offset_face")

    def get_cell_bounding_boxes(self) -> np.ndarray:
        if not kokkos_runtime.is_running():
            raise RuntimeError("Kokkos not running...")
        try:
            n_cells = self.num_entities(2)
            cell_bounding_boxes = np.empty(n_cells * 4, dtype=np.float64)
            _dll.capi_get_cell_bounding_boxes(
                self.mesh, cell_bounding_boxes, n_cells * 4
            )
            return cell_bounding_boxes
        except Exception as exception:
            raise RuntimeError(f"Error getting cell bounding boxes: {exception}")

    def get_cell_volumes(self) -> np.ndarray:
        if not kokkos_runtime.is_running():
            raise RuntimeError("Kokkos not running...")

        try:
            n_cells = self.num_entities(2)
            cell_volumes = np.empty(n_cells, dtype=np.float64)
            _dll.capi_get_cell_volumes(self.mesh, cell_volumes, n_cells)
            return cell_volumes
        except Exception as exception:
            raise RuntimeError(f"Error getting cell volumes: {exception}")

    def get_cell_centroids(self) -> np.ndarray:
        if not kokkos_runtime.is_running():
            raise RuntimeError("Kokkos not running...")

        try:
            n_cells = self.num_entities(2)
            cell_centroids = np.empty(2 * n_cells, dtype=np.float64)
            _dll.capi_get_cell_centroids(self.mesh, cell_centroids, 2 * n_cells)
            return cell_centroids
        except Exception as exception:
            raise RuntimeError(f"Error getting cell centroids: {exception}")

    def get_edge_coordinates(self) -> np.ndarray:
        if not kokkos_runtime.is_running():
            raise RuntimeError("Kokkos not running...")

        try:
            n_edges = self.num_entities(1)
            edge_coordinates = np.empty(n_edges * 4, dtype=np.float64)
            _dll.capi_get_edge_coordinates(self.mesh, edge_coordinates, n_edges * 4)
            return edge_coordinates
        except Exception as exception:
            raise RuntimeError(f"Error getting edge coordinates: {exception}")

    def get_number_of_edges_inside_wall(self) -> int:
        if not kokkos_runtime.is_running():
            raise RuntimeError("Kokkos not running...")

        try:
            return _dll.capi_get_number_of_edges_inside_wall(self.mesh)
        except Exception as exception:
            raise RuntimeError(f"Error getting number of edges: {exception}")

    def get_edge_to_face_map(self) -> np.ndarray:
        if not kokkos_runtime.is_running():
            raise RuntimeError("Kokkos not running...")

        try:
            nedges = self.num_entities(1)
            edge_map_size = nedges * 2
            edge_to_face_map = np.empty(edge_map_size, dtype=np.int32)
            _dll.capi_get_edge_to_face_connectivity(
                self.mesh, edge_to_face_map, edge_map_size
            )
            return edge_to_face_map.reshape((nedges, 2))

        except Exception as exception:
            raise RuntimeError(f"Error getting edge to face map: {exception}")

    def get_wall_edge_ids(self) -> np.ndarray:
        if not kokkos_runtime.is_running():
            raise RuntimeError("Kokkos not running...")

        num_first_wall_points = np.sum(self.get_integer_tag_array(0, "isOnWall"))
        wall_edge_ids = np.empty(num_first_wall_points, dtype=np.int32)

        try:
            _dll.capi_get_wall_edge_ids(
                self.mesh, wall_edge_ids, wall_edge_ids.shape[0]
            )
            return wall_edge_ids
        except Exception as exception:
            raise RuntimeError(f"Error getting wall edge ids: {exception}")

    def get_wall_adjacent_triangles(self) -> np.ndarray:
        if not kokkos_runtime.is_running():
            raise RuntimeError("Kokkos not running...")

        num_first_wall_points = np.sum(self.get_integer_tag_array(0, "isOnWall"))
        wall_adjacent_triangles = np.empty(2 * num_first_wall_points, dtype=np.int32)

        try:
            _dll.capi_get_wall_adjacent_triangles(
                self.mesh, wall_adjacent_triangles, 2 * num_first_wall_points
            )
            return wall_adjacent_triangles
        except Exception as exception:
            raise RuntimeError(f"Error getting wall adjacent triangles: {exception}")

    def get_edge_coefficients(self, print_debug=False, tol=1e-6):
        if not kokkos_runtime.is_running():
            raise RuntimeError("Kokkos not running...")

        num_edges = self.num_entities(1)
        coefficients = np.zeros(num_edges * 6, dtype=np.float64)
        edge_types = np.empty(num_edges, dtype=np.int32)
        size = coefficients.size

        try:
            _dll.capi_compute_edge_coefficients(
                self.mesh, c_int(size), coefficients, edge_types, print_debug, tol
            )

        except Exception as exception:
            raise RuntimeError(f"Error computing edge coefficients: {exception}")

        return coefficients.reshape(num_edges, 6), edge_types

    def get_num_of_boundary_edges(self) -> int:
        if not kokkos_runtime.is_running():
            raise RuntimeError("Kokkos not running...")

        try:
            return _dll.capi_get_number_of_boundary_edges(self.mesh)

        except Exception as exception:
            raise RuntimeError(f"Error computing number of boundary edges: {exception}")

    def get_boundary_edge_ids(self, size=0):
        if not kokkos_runtime.is_running():
            raise RuntimeError("Kokkos not running...")

        if size == 0:
            size = self.get_num_of_boundary_edges()

        boundary_edge_ids = np.zeros(size, dtype=np.int32)
        try:
            _dll.capi_get_boundary_edge_ids(self.mesh, size, boundary_edge_ids)
        except Exception as exception:
            raise RuntimeError(f"Error computing boundary edge ids: {exception}")

        return boundary_edge_ids

    def get_face_connectivity(
        self, edge_coefficients=None, print_debug=False, tol=1e-6
    ):
        if not kokkos_runtime.is_running():
            raise RuntimeError("Kokkos not running...")

        n_faces = self.num_entities(2)
        n_edges = self.num_entities(1)

        if edge_coefficients is None:
            edge_coefficients, edge_types_ = self.get_edge_coefficients()

        assert edge_coefficients.shape[0] == n_edges
        assert edge_coefficients.shape[1] == 6
        assert edge_coefficients.dtype == np.float64

        edge_coefficients_shaped = edge_coefficients.reshape(n_edges * 6)

        face_connectivity = np.zeros(n_faces * 6, dtype=np.int32)

        # TODO add core-dump capture
        try:
            _dll.capi_get_face_connectivity(
                self.mesh,
                n_edges * 6,
                edge_coefficients_shaped,
                n_faces * 6,
                face_connectivity,
                print_debug,
                tol,
            )
        except Exception as exception:
            raise RuntimeError(f"Error computing face connectivity: {exception}")

        return face_connectivity.reshape(n_faces, 6)

    def get_all_geometry_info(self, print_debug=False, tol=1e-6):
        if not kokkos_runtime.is_running():
            raise RuntimeError("Kokkos not running...")

        n_faces = self.num_entities(2)
        n_edges = self.num_entities(1)

        edge_coefficients = np.zeros(n_edges * 6, dtype=np.float64)
        edge_types = np.empty(n_edges, dtype=np.int32)
        boundary_edge_ids_num = self.get_num_of_boundary_edges()
        boundary_edge_ids = np.zeros(boundary_edge_ids_num, dtype=np.int32)
        face_connctivity = np.zeros(n_faces * 6, dtype=np.int32)

        try:
            _dll.capi_get_all_geometry_info(
                self.mesh,
                n_edges,
                n_faces,
                edge_coefficients,
                edge_types,
                boundary_edge_ids,
                face_connctivity,
                print_debug,
                tol,
            )
        except Exception as exception:
            raise RuntimeError(f"Error computing all geometry info: {exception}")

        return (
            edge_coefficients.reshape(n_edges, 6),
            edge_types,
            boundary_edge_ids,
            face_connctivity.reshape(n_faces, 6),
        )


def read_edge_coefficients_from_file(filename):
    with open(filename, "r") as f:
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
    with open(filename, "r") as f:
        data = f.readlines()
        n_faces = len(data)
        connectivity = np.zeros((n_faces, 6), dtype=np.int32)

        for i in range(n_faces):
            values = data[i].split()
            assert len(values) == 7
            face_id = int(values[0])
            assert face_id == i

            for j in range(6):
                connectivity[i, j] = np.int32(values[j + 1])

    return connectivity
