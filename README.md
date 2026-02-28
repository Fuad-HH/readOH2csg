# Convert `Omega_h` Mesh to CSG Format
[![License](https://img.shields.io/badge/license-BSD--3--Clause-02B36C)](https://github.com/Fuad-HH/readOH2csg/blob/parallel/LICENSE)
[![GitHub Actions build status (Linux)](https://github.com/Fuad-HH/readOH2csg/actions/workflows/ci.yml/badge.svg?branch=parallel)](https://github.com/Fuad-HH/readOH2csg/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/Fuad-HH/readOH2csg?include_prereleases
)](https://github.com/Fuad-HH/readOH2csg/releases)

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
# you may not need to set PKG_CONFIG_PATH manually if installed in standard locations
# or using a package managers or modules
export PKG_CONFIG_PATH=<netCDF-cxx_package_config_path>:$PKG_CONFIG_PATH
export PKG_CONFIG_PATH=<netCDF-c_package_config_path>:$PKG_CONFIG_PATH

# configure
cmake -S . -B build \
  -DOmega_h_ROOT=<Omega_h_install_dir> \
  -DKokkos_ROOT=<Kokkos_install_dir> \
  -DCMAKE_BUILD_TYPE=Release

# build
cmake --build build
```
>[!NOTE]
> `netCDF-cxx` and `netCDF-c` are sometimes installed as
> package config instead of cmake modules. If they are cmake
> modules, you have to pass `-DnetCDF_CXX_ROOT=<netCDF-cxx_install_dir>`
> and `-DnetCDF_C_ROOT=<netCDF-c_install_dir>`.
> If the netCDF libraries are installed as package configs,
> you may need to set `PKG_CONFIG_PATH` if they are not set by
> the package manager or module system.

## Documentation
More details on strategies, logic, and math are included in the `doc/` directory.
