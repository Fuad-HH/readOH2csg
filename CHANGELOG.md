# Changelog

All notable changes are listed here ([Keep a Changelog](https://keepachangelog.com/)).

## [0.1.0] - unreleased

### Changed
- All computation moved to a C++/Kokkos pipeline (OpenMP or CUDA). Python is
  a thin pybind11 layer (`omegah2csg._core`) with editable NumPy results.
- `convert2openmc` and `convert2degas2` are native executables (also shipped
  in the wheel).
- DEGAS2 files are written as CDF-2 by netcdf-c (`--nc-format` to change).
- See [docs/migration-0.1.md](docs/migration-0.1.md) for the API and CLI changes.

### Removed
- ctypes C API (`capi.*`), `OmegaHMesh`, `kokkos_runtime`, Python converters.

### Fixed
- OpenMC reflective boundary set on the wrong surfaces (signed ids used as indices).
- DEGAS2 boundary-layer assertion never ran (`has_boundary_layer` not called).
- Default tolerance rejected by `convert2openmcXML`.
- DEGAS2 last `zone_volume` computed from an uninitialized array.
- Kernels allocated in Kokkos default spaces instead of Omega_h's.
- Bad geometry aborted the process; it now raises `GeometryError`.
