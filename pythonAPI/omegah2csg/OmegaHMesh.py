from ctypes import c_void_p, c_char_p, Structure, c_int
from . import _dll, kokkos_runtime
import os


# Define C structures
class OmegaHLibraryPointer(Structure):
    _fields_ = [("pointer", c_void_p)]


class OmegaHMeshPointer(Structure):
    _fields_ = [("pointer", c_void_p)]


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
        if not kokkos_runtime.kokkos_initialized :
            kokkos_runtime.kokkos_initialize()

        # check if given directory exits
        if not os.path.isdir(filename) :
            raise RuntimeError(f"OmegaHMesh filename {filename} does not exist.")



    def __enter__(self):
        # Create library
        self.lib = _dll.create_omegah_library()

        # Create mesh in try catch to capture error
        try:
            self.mesh = _dll.create_omegah_mesh(self.lib, self.filename.encode('utf-8'))
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


