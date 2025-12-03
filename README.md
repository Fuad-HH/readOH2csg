# Convert `Omega_h` Mesh to CSG Format

Efficiently convert Tokamak mesh from `Omega_h` format
to Constructive Solid Geometry (CSG). It supports creating
CSG representation in
1. `Degas2`'s `geometry.nc` format
2. `OpenMC`'s geometry XML or a Python object


## Installation
It depends on `Omega_h`. `Omega_h` has to be built
with `Kokkos` and supports any `Kokkos` backend.

Configuration example with `CMake`:
```bash
cmake -S . -B build \
  -DOmega_h_ROOT=<Omega_h_install_dir> \
  -DKokkos_ROOT=<Kokkos_install_dir> \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build
```