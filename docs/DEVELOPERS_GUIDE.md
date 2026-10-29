# Developer's Guide

The design is in [`designs/cpp-pipeline-modernization-design.md`](designs/cpp-pipeline-modernization-design.md);
decisions are summarized in [`design_choices.rst`](design_choices.rst).

## Layout

| Path | Contents |
|---|---|
| `src/omegah2csg/` | C++ library. Host-only code (`options`, `degas2_schema`, `io/` writers) is in target `omegah2csg_host`; device code (stages, DEGAS2 assembly, `pipeline`) is in `omegah2csg_core` |
| `apps/` | native executables `convert2openmc`, `convert2degas2` (shared driver `cli.cpp`) |
| `bindings/` | pybind11 module `omegah2csg._core` (no kernels) |
| `python/omegah2csg/` | Python package: re-exports, `openmc_builder`, console scripts |
| `tests/python/` | pytest suite (compares with saved results in `tests/assets`) |
| `bench/` | mesh generator, Kokkos Tools copy counter, `run_bench.py` |
| `ci/` | pinned versions, constraints, DEGAS2 acceptance script |

### Layers (dependencies point down only)

| Layer | Modules |
|---|---|
| L0 | `geom.hpp`: per-entity math (`OMEGA_H_INLINE`, host and device) |
| L1 | `mesh_input` (checks before omega-h is called), `topology` (adjacency built once) |
| L2 | `csg`, `wall`, `cell_quantities`: kernels over `Topology` |
| L3 | `degas2_assemble` → `Degas2Geometry` |
| L4 | `io/openmc_xml`, `io/degas2_netcdf`, `io/degas2_aux`: host only, no Omega_h/Kokkos headers |
| L5 | `pipeline`: the only code that chains stages |

### Kernel rules
- Allocate with `Omega_h::Write<T>` and launch with `Omega_h::parallel_for`, or
  with `Kokkos::parallel_reduce/scan` using `Omega_h::policy(n)`. This keeps code
  in Omega_h's memory and execution spaces. CI rejects `Kokkos::View<` and
  `Kokkos::parallel_for` outside `transfer.hpp`.
- Kernels never abort. Report bad entities with `DeviceErrors::report()`, then
  call `throw_if_any()` on the host.
- Device→host copies of bulk arrays go through `transfer.hpp`.
- Initialize Omega_h arrays with `=` or `()`, never braces: `Write<T>{...}`
  selects the `initializer_list` constructor.
- Style: `#ifndef` include guards, no `using` declarations or aliases, fully
  qualified names (`Omega_h::LO`).

## Building
With a spack environment providing omega-h (Kokkos), netcdf-c, pugixml and pybind11:
```bash
pip install scikit-build-core numpy pytest netCDF4 openmc
pip install -e .[test] --no-build-isolation      # build dir: build/{wheel_tag}
pytest
```
C++ edits need the same `pip install` again. For quick C++ iterations:
```bash
cmake -S . -B build/dev -DOMEGAH2CSG_ENABLE_PYTHON=ON -DOMEGAH2CSG_ENABLE_BENCH=ON
cmake --build build/dev -j
```
CMake options: `OMEGAH2CSG_ENABLE_PYTHON`, `OMEGAH2CSG_ENABLE_APPS`,
`OMEGAH2CSG_ENABLE_TESTS` (registers pytest with CTest),
`OMEGAH2CSG_ENABLE_BENCH`, `OMEGAH2CSG_OPENMC_EXE_NAME`,
`OMEGAH2CSG_DEGAS2_EXE_NAME`. All dependencies are required; configuration
fails if one is missing.

## Testing
- `pytest` runs the suite in `tests/python`. It compares results with the
  saved 0.0.x files (`6elem_mesh_files/*.dat`) and the DEGAS2 gold files
  (`simple_degas2_case/*.nc`), and also covers editing arrays and the CLI.
- `ci/degas2_acceptance.sh` runs DEGAS2 on the converted test case (needs
  `DEGAS2_SRC` and `DEGAS2_BIN`).
- Benchmarks: see [`../bench/README.md`](../bench/README.md).

CI: `ci.yml` (pre-commit, Kokkos spaces guard, OpenMP tests at 1 and 4 threads,
CUDA compile-only); `gpu.yml` (self-hosted GPU: tests, DEGAS2 acceptance,
benchmarks); `build-wheels.yml` (manylinux wheels, sdist, PyPI on tags).

## Formatting
`pre-commit run --all-files` runs ruff, clang-format 18, cmake-format and
validate-pyproject. Install the hook once with `pre-commit install`.

## Releasing
Bump `project(VERSION ...)` in `CMakeLists.txt`, update `CHANGELOG.md`, and
push a `v*` tag. `build-wheels.yml` builds the wheels and the sdist and
publishes them with PyPI Trusted Publishing.
