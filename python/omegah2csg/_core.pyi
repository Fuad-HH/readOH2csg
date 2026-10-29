"""
omegah2csg C++/Kokkos pipeline
"""
from __future__ import annotations
import collections.abc
import numpy
import numpy.typing
import os
import typing
__all__: list[str] = ['BoundaryType', 'CommonOptions', 'CsgGeometry', 'Degas2Geometry', 'Degas2Options', 'GeometryError', 'InputError', 'NcFormat', 'OpenMCOptions', 'SurfaceType', 'compute_csg', 'compute_degas2', 'convert_to_degas2', 'convert_to_openmc', 'initialize', 'show_config', 'write_degas2', 'write_openmc_xml']
class GeometryError(RuntimeError):
    pass
class InputError(ValueError):
    pass
class SurfaceType:
    """
    Members:
    
      ZPlane
    
      ZCylinder
    
      ZCone
    """
    ZCone: typing.ClassVar[SurfaceType]
    ZCylinder: typing.ClassVar[SurfaceType]
    ZPlane: typing.ClassVar[SurfaceType]
    __members__: typing.ClassVar[dict[str, SurfaceType]]
    @typing.overload
    def __eq__(self, other: SurfaceType) -> bool:
        ...
    @typing.overload
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    @typing.overload
    def __ne__(self, other: SurfaceType) -> bool:
        ...
    @typing.overload
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class BoundaryType:
    """
    Members:
    
      Transmission
    
      Vacuum
    
      Reflective
    
      White
    """
    Reflective: typing.ClassVar[BoundaryType]
    Transmission: typing.ClassVar[BoundaryType]
    Vacuum: typing.ClassVar[BoundaryType]
    White: typing.ClassVar[BoundaryType]
    __members__: typing.ClassVar[dict[str, BoundaryType]]
    @typing.overload
    def __eq__(self, other: BoundaryType) -> bool:
        ...
    @typing.overload
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    @typing.overload
    def __ne__(self, other: BoundaryType) -> bool:
        ...
    @typing.overload
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class NcFormat:
    """
    Members:
    
      Auto
    
      Cdf2
    
      Cdf5
    
      NetCdf4
    """
    Auto: typing.ClassVar[NcFormat]
    Cdf2: typing.ClassVar[NcFormat]
    Cdf5: typing.ClassVar[NcFormat]
    NetCdf4: typing.ClassVar[NcFormat]
    __members__: typing.ClassVar[dict[str, NcFormat]]
    @typing.overload
    def __eq__(self, other: NcFormat) -> bool:
        ...
    @typing.overload
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    @typing.overload
    def __ne__(self, other: NcFormat) -> bool:
        ...
    @typing.overload
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class CommonOptions:
    force: bool
    output: str
    timing: bool
    @property
    def debug_entities(self) -> int:
        ...
    @debug_entities.setter
    def debug_entities(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def tol(self) -> float:
        ...
    @tol.setter
    def tol(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def verbosity(self) -> int:
        ...
    @verbosity.setter
    def verbosity(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
class OpenMCOptions(CommonOptions):
    boundary_type: str
    material_ids_file: str
    def __init__(self) -> None:
        ...
    def validate(self) -> None:
        ...
    @property
    def cell_id_offset(self) -> int:
        ...
    @cell_id_offset.setter
    def cell_id_offset(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def surface_id_offset(self) -> int:
        ...
    @surface_id_offset.setter
    def surface_id_offset(self, arg0: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
class Degas2Options(CommonOptions):
    aux_dir: str
    aux_files: bool
    nc_format: NcFormat
    def __init__(self) -> None:
        ...
    def validate(self) -> None:
        ...
    @property
    def target_recyc_coef(self) -> float:
        ...
    @target_recyc_coef.setter
    def target_recyc_coef(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    @property
    def target_temperature_K(self) -> float:
        ...
    @target_temperature_K.setter
    def target_temperature_K(self, arg0: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class CsgGeometry:
    """
    Result of compute_csg(): one surface per mesh edge, one cell per mesh face.
    Array properties are NumPy copies made on first access; edit them in place
    or assign new arrays. surface_coeffs columns: A, B, C, D, up flag, slope of
    F(r, z) = A r^2 + B z^2 + C z + D.
    """
    @property
    def boundary_edges(self) -> numpy.ndarray:
        ...
    @boundary_edges.setter
    def boundary_edges(self, arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.int32]) -> None:
        ...
    @property
    def boundary_sides(self) -> numpy.ndarray:
        ...
    @boundary_sides.setter
    def boundary_sides(self, arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.int8]) -> None:
        ...
    @property
    def cell_bbox(self) -> numpy.ndarray:
        ...
    @cell_bbox.setter
    def cell_bbox(self, arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None:
        ...
    @property
    def cell_centroid(self) -> numpy.ndarray:
        ...
    @cell_centroid.setter
    def cell_centroid(self, arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None:
        ...
    @property
    def cell_material(self) -> numpy.typing.NDArray[numpy.int32]:
        """
        material id per cell (-1 = void)
        """
    @cell_material.setter
    def cell_material(self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.int32]) -> None:
        ...
    @property
    def cell_volume(self) -> numpy.ndarray:
        ...
    @cell_volume.setter
    def cell_volume(self, arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None:
        ...
    @property
    def edge_faces_sorted(self) -> numpy.ndarray:
        ...
    @edge_faces_sorted.setter
    def edge_faces_sorted(self, arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.int32]) -> None:
        ...
    @property
    def face_edges(self) -> numpy.ndarray:
        ...
    @face_edges.setter
    def face_edges(self, arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.int32]) -> None:
        ...
    @property
    def face_region_tag(self) -> numpy.ndarray:
        ...
    @face_region_tag.setter
    def face_region_tag(self, arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.int32]) -> None:
        ...
    @property
    def face_sides(self) -> numpy.ndarray:
        ...
    @face_sides.setter
    def face_sides(self, arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.int8]) -> None:
        ...
    @property
    def interior_edge_count(self) -> int:
        ...
    @property
    def nedges(self) -> int:
        ...
    @property
    def nfaces(self) -> int:
        ...
    @property
    def node_coordinates(self) -> numpy.typing.NDArray[numpy.float64]:
        ...
    @property
    def nverts(self) -> int:
        ...
    @property
    def surface_boundary(self) -> numpy.typing.NDArray[numpy.int8]:
        """
        BoundaryType code per surface
        """
    @surface_boundary.setter
    def surface_boundary(self, arg0: typing.Annotated[numpy.typing.ArrayLike, numpy.int8]) -> None:
        ...
    @property
    def surface_coeffs(self) -> numpy.ndarray:
        ...
    @surface_coeffs.setter
    def surface_coeffs(self, arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.float64]) -> None:
        ...
    @property
    def surface_type(self) -> numpy.ndarray:
        ...
    @surface_type.setter
    def surface_type(self, arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.int8]) -> None:
        ...
    @property
    def wall_edges(self) -> numpy.ndarray:
        ...
    @wall_edges.setter
    def wall_edges(self, arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.int32]) -> None:
        ...
    @property
    def wall_faces(self) -> numpy.ndarray:
        ...
    @wall_faces.setter
    def wall_faces(self, arg1: typing.Annotated[numpy.typing.ArrayLike, numpy.int32]) -> None:
        ...
class Degas2Geometry:
    """
    Result of compute_degas2(): one array per DEGAS2 geometry.nc variable,
    accessed by name (geom["zone_volume"]). Arrays are NumPy copies made on
    first access; edit them in place or assign new arrays (shapes are checked
    when written).
    """
    def __getitem__(self, arg0: str) -> numpy.ndarray:
        ...
    def __setitem__(self, arg0: str, arg1: typing.Any) -> None:
        ...
    def keys(self) -> list[str]:
        ...
@typing.overload
def compute_csg(source: os.PathLike | str | bytes, options: OpenMCOptions = ...) -> CsgGeometry:
    """
    Compute CSG surfaces and cells from an .osh path.
    """
@typing.overload
def compute_csg(source: ..., options: OpenMCOptions = ...) -> CsgGeometry:
    """
    Compute CSG surfaces and cells from a PyOmega_h mesh.
    """
@typing.overload
def compute_degas2(source: os.PathLike | str | bytes, options: Degas2Options = ...) -> Degas2Geometry:
    """
    Compute every DEGAS2 geometry variable from an .osh path.
    """
@typing.overload
def compute_degas2(source: ..., options: Degas2Options = ...) -> Degas2Geometry:
    """
    Compute every DEGAS2 geometry variable from a PyOmega_h mesh.
    """
def convert_to_degas2(source: os.PathLike | str | bytes, options: Degas2Options = ...) -> None:
    """
    Mesh path -> options.output (geometry.nc) and aux files.
    """
def convert_to_openmc(source: os.PathLike | str | bytes, options: OpenMCOptions = ...) -> None:
    """
    Mesh path -> options.output (geometry.xml), without Python objects.
    """
def initialize(kokkos_args: collections.abc.Sequence[str]) -> None:
    """
    Start Kokkos/MPI with arguments (before any other call).
    """
def show_config() -> dict[str, str]:
    """
    Versions, Kokkos execution space and library paths.
    """
def write_degas2(geometry: Degas2Geometry, path: os.PathLike | str | bytes, options: Degas2Options = ...) -> None:
    """
    Write geometry.nc (+ aux files) from a (possibly edited) Degas2Geometry.
    """
def write_openmc_xml(geometry: CsgGeometry, path: os.PathLike | str | bytes, options: OpenMCOptions = ...) -> None:
    """
    Write geometry.xml from a (possibly edited) CsgGeometry.
    """
__version__: str = '0.1.0'
