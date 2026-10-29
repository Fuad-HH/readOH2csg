# Convert `Omega_h` Mesh to CSG Format
[![License](https://img.shields.io/badge/license-BSD--3--Clause-02B36C)](https://github.com/Fuad-HH/readOH2csg/blob/main/LICENSE)
[![GitHub Actions build status (Linux)](https://github.com/Fuad-HH/readOH2csg/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Fuad-HH/readOH2csg/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/Fuad-HH/readOH2csg?include_prereleases)](https://github.com/Fuad-HH/readOH2csg/releases)

Convert tokamak meshes in `Omega_h` (`.osh`) format to Constructive Solid
Geometry (CSG) for
1. `DEGAS2`: `geometry.nc` (plus `wallfile.txt`, `plasmafile.txt`, `sourcefile.txt`),
   or the DEGAS2 arrays themselves, editable from Python;
2. `OpenMC`: `geometry.xml`, or an [`openmc.Universe`](https://docs.openmc.org/en/stable/pythonapi/generated/openmc.Universe.html)
   built from editable arrays.

Every computation runs in C++ with Kokkos (OpenMP or CUDA). Each mesh edge
becomes a plane, cylinder or one-sided cone around the z axis, and each
triangle becomes one cell.

## Installation

### 1. Pre-built wheel (Linux x86_64, OpenMP)
```bash
python -m venv omegah2csg-env && source omegah2csg-env/bin/activate
pip install omegah2csg            # add [openmc] for to_openmc_universe()
convert2degas2 --help
```
The wheel bundles `Omega_h`, `Kokkos` (OpenMP) and `netcdf-c`, and ships the
native executables. GPU builds come from source or spack.

### 2. From source (any Kokkos backend, including CUDA)
Requirements: a C++17 compiler, CMake ≥ 3.24, Python ≥ 3.11, and
- [`Omega_h`](https://github.com/SCOREC/omega_h) built with Kokkos (any backend),
- [`netcdf-c`](https://github.com/Unidata/netcdf-c) (CMake config or pkg-config),
- [`pugixml`](https://pugixml.org) (CMake config),
- [`pybind11`](https://github.com/pybind/pybind11) (same version as `PyOmega_h`, see `ci/constraints.txt`).

With a spack environment that provides them:
```bash
pip install scikit-build-core numpy
pip install -e .[test] --no-build-isolation   # Python module + executables
pytest
```
Or the C++ library and executables only:
```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH="<omega_h>;<pugixml>;<netcdf-c>"
cmake --build build -j && cmake --install build --prefix <install_dir>
```

## Usage

### Command line
```bash
convert2degas2 mesh.osh -o geometry.nc -a --aux-dir case/   # DEGAS2 (+ aux files)
convert2openmc mesh.osh -o geometry.xml --material-ids ids.txt
convert2degas2 mesh.osh --check                               # validate tags only
convert2openmc --version -v                                   # build configuration
```
Run `--help` for every option (`--tol`, `--force`, `--timing`, `--nc-format`,
`--boundary-type`, id offsets, ...). DEGAS2 meshes need the `offset_face`
(faces: 0 plasma, 1 boundary layer) and `isOnWall` (vertices on the wall) tags.
Exit codes: 0 success, 1 usage error, 2 input/geometry error, 3 other error.

### Python
```python
import omegah2csg as o2c

g = o2c.compute_csg("mesh.osh")             # arrays copied to NumPy on access
g.cell_material[:] = 1                      # edit inputs or geometry arrays
o2c.write_openmc_xml(g, "geometry.xml")     # fast C++ writer
universe = o2c.to_openmc_universe(g, materials={0: fuel, 1: wall})

d = o2c.compute_degas2("mesh.osh")
d["target_temperature"][1:] = 1e-20         # any DEGAS2 variable by name
o2c.write_degas2(d, "geometry.nc")
```
A worked DEGAS2 case is in [`tests/assets/simple_degas2_case/degas2-case`](tests/assets/simple_degas2_case/degas2-case/).

### Performance
- `--timing` prints the time of each stage. `convert2openmc --version -v`
  (or `omegah2csg.show_config()`) shows the execution space (`OpenMP` or `Cuda`).
- OpenMP: `export OMP_PROC_BIND=spread OMP_PLACES=threads OMP_NUM_THREADS=<n>`.
  Kokkos options can also be given as `--kokkos-num-threads=<n>` or
  `--kokkos-device-id=<id>`.
- For very large meshes, use the C++ writers (`write_openmc_xml`,
  `convert2openmc`) instead of `to_openmc_universe`. Building Python
  OpenMC objects costs time for every cell.

## Documentation
- [`docs/DEVELOPERS_GUIDE.md`](docs/DEVELOPERS_GUIDE.md): code structure, building, testing.
- [`docs/design_choices.rst`](docs/design_choices.rst) and [`docs/references.rst`](docs/references.rst).
- [`docs/migration-0.1.md`](docs/migration-0.1.md): changes from 0.0.x.
- [`docs/designs/`](docs/designs/): design documents.
