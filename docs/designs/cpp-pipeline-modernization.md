# omegah2csg 0.1: a C++/Kokkos conversion pipeline with a thin, extensible Python layer

* **Author:** Fuad Hasan
* **Date:** 2026-10-28
* **Status:** Brainstorm brief, revision 2. Supersedes the rev-1 plan (`~/.claude/plans/previously-omega-h-didn-t-provide-floofy-flamingo.md`). Revision 2 adds the user's decisions after the Goldfish review: no vendored PyOmega_h, native executables, the DEGAS2 netCDF format, and partitioned meshes out of scope.
* **Reviewed by Goldfish lenses:** design, codebase-details, zoom-out, comprehension, implementation-readiness, structural. All six completed on revision 1. Revision 2 changes the packaging and runtime sections, which have not been re-reviewed yet.

## Overview

omegah2csg converts 2D Omega_h tokamak meshes (XGC/DEGAS2 meshes, generally multi-million triangles, not partitioned) into constructive solid geometry (CSG) for OpenMC and DEGAS2. Other tools already do this but are slow, so speed is the reason this project exists.

How the new design works:
- **Compute:** a C++ library reads the mesh once. It computes everything with Kokkos kernels: `parallel_for`, plus Omega_h's own reduce and scan helpers. The kernels run on OpenMP or CUDA, and the data stays on the device until output. "Everything" means:
  - edge surfaces, cell sides, boundary and wall data;
  - cell volumes, centroids and bounding boxes;
  - the complete OpenMC tables and DEGAS2 arrays.
- **Results:** the results are two data objects, `CsgGeometry` and `Degas2Geometry`. Their arrays can be read and modified from Python.
- **Front ends:**
  - **Native C++ executables** `convert2openmc` and `convert2degas2`, used from spack and source builds and on GPU nodes.
  - **A thin pybind11 module `omegah2csg._core`.** It is used from scripts, OpenMC workflows and the wheel's console scripts.
- **Output:**
  - C++ writers produce `geometry.xml` and the DEGAS2 netCDF and aux files directly.
  - From Python, users can also build `openmc` cells and universes from the arrays, or edit DEGAS2 arrays before writing. This lets them change materials, cells or boundaries, or extend the geometry.
- **Runtime:** omega-h's own `Omega_h::Library` owns Kokkos and MPI at runtime. PyOmega_h is not vendored and not built here.
- **PyOmega_h interop:** works today in spack or source builds. omega-h's own CI/CD will publish PyOmega_h to PyPI. Once it does, our wheel depends on that package and links against the libomega_h and Kokkos it ships, so interop works from pip too, with one copy of each library. Until then, wheels bundle their own libomega_h and accept file paths only.
- **omega-h itself is not changed.**

## Analysis

### 1. Why rev 1 is replaced (all six lenses agreed)

Rev 1 moved cell volumes, centroids, bounding boxes and edge coordinates to NumPy on top of PyOmega_h, and kept both converters in Python. Under the speed goal that is backwards:

- **The NumPy path is slow.** PyOmega_h converts each array by copying device→host and then copying again one element at a time in a single thread (`PyOmega_h_numpy_transform.hpp:36-47`). `mesh.coords()`, `ask_elem_verts()`, `ask_verts_of()` and `get_tag()` all go through this path, and the NumPy formulas add large temporaries on top.
- **The fast versions already exist.** These quantities are already device kernels in `src/capi.cpp:298-436`. Keep them in C++ and fix their two small bugs: the bounding box is seeded with `INT64_MAX`, and the centroid kernel carries the wrong label.
- **The Python converters are where the time goes.**
  - `convert2degas2.py` loops in Python over every triangle (`:517`) and every edge (`:15`, `:537`).
  - It also runs an `np.isin` inside the per-edge loop (`:689-698`), which is O(N_edges × N_wall).
  - `convert2openmc.py:152-204` creates one Python object per surface and per cell.

### 2. Data model: plain arrays, computed once, with named meaning

- **Build topology once.** A `Topology` struct of device arrays holds: coordinates, edge→vertex, face→vertex, face→edge, edge→face, `offset_face`, `isOnWall`. Kernels can then be unit-tested on hand-built arrays without loading a file. Three places in the current code recompute this:
  - Passing `Omega_h::Mesh` by value (`compute_surface.cpp:129`, `:670`) drops Omega_h's adjacency cache.
  - `get_wall_adjacent_triangles` recomputes the wall edges (`:97`).
  - `capi.cpp:449-450` calls `ask_up` twice.
- **One explicit stage chain:** topology → edge surfaces → face sides → boundary/wall → cell quantities → OpenMC tables / DEGAS2 arrays. A stage never calls another stage; only the pipeline chains them.
- **Columns and encodings get explicit meaning.**
  - Today the 6 coefficient columns are positional: index 4 is the up/down flag and index 5 is the slope. Code and tests already disagree on which column is which (`convert2openmc.py:145-150` vs `test_openmc_geometry_creation.py:144-148`).
  - Face connectivity is interleaved as `[edge, sign] × 3`.
  - Boundary edges are stored as `edge_id * sign`, which cannot carry the sign of edge 0.
  - The new data model uses separate `face_edges[n,3]` and `face_signs[n,3]` arrays, named accessors for coefficients, and an explicit `EdgeType` instead of testing `coeffs[1] ≈ -1`.
  - Signed or 1-based encodings exist only inside the writers. This fixes edge 0 without changing any output format.
- **Memory and execution spaces must match Omega_h's.** This is a correctness bug, verified in `Omega_h_kokkos.hpp:29-49`.
  - **Omega_h's spaces:** its `ExecSpace` is `Kokkos::OpenMP` whenever `OMEGA_H_USE_OpenMP` is set, even with CUDA on. Its `Space` can be `SharedSpace` or `HostPinned`.
  - **Our spaces:** the current code allocates `Kokkos::View<…*[6]>` in the default space (`compute_surface.cpp:128-141`, `capi.cpp:36-38`). It runs `Kokkos::parallel_scan`/`parallel_reduce` on the default execution space (`compute_surface.cpp:73`, `:262`; `capi.cpp:461`).
  - **Consequence:** where these differ, kernels touch memory they cannot reach.
  - **Fix:** allocate with `Omega_h::Write<T>`, or with `Kokkos::View<T*[N], LayoutRight, Omega_h::Space>`, and launch everything through `Omega_h::policy` / `Omega_h::parallel_for`.
- **Size limits.** Multi-million-triangle meshes are far below the int32 `Omega_h::LO` limit, which is about 358M edges for flat 6-wide arrays. They are also below DEGAS2's `i4` and OpenMC's 32-bit ID limits, which are hundreds of millions of triangles. The host still checks these limits before running kernels, so an oversized mesh fails with a message instead of overflowing silently.
  - Device memory is roughly 300–500 bytes per triangle. That is 1.5–2.5 GB at 5M triangles, which fits common GPUs.

### 3. Python access and extensibility

- **Result objects, bound in pybind11:**
  - `CsgGeometry`: surface table (type and coefficients), cell side tables, boundary flags, cell quantities.
  - `Degas2Geometry`: every array the netCDF file needs.
- **How arrays reach Python.** Arrays stay on the device until asked for. Each property then copies once, straight into a NumPy buffer, and caches the result. The copy is a Kokkos `deep_copy` into an unmanaged LayoutRight view of the NumPy memory.
- **Two ways to use the results:**
  - **Extend or modify, then build or write:**
    - `to_openmc_universe(geom, *, materials, boundary_type, cell_id_offset, surface_id_offset)` builds `openmc` objects from the arrays.
    - `materials` accepts one Material, a mapping from tag value to Material, or a per-cell array. The default fill is void, replacing today's silent 1 g/cc H-1 placeholder (`convert2openmc.py:171-176`).
    - `Degas2Geometry` arrays can be edited in NumPy, then passed to `write_netcdf(geom, path)`.
  - **Direct output:** the C++ writers produce `geometry.xml` and the netCDF file with no Python objects.
  - The writers take host arrays, so arrays modified in Python go through the same writer.
- **Cost on multi-million meshes.** Building `openmc` objects in Python costs time per object, and so does `openmc.Geometry.from_xml`. At a few million cells this takes minutes. The docs say so and point large-mesh users to the XML path.
- **Consistency between the two paths.** The C++ XML writer must reproduce how OpenMC expands `ZConeOneSided` (a cone plus a plane, combined in a region expression). A test keeps the XML and Python-object paths identical.
- **One classification.** One `CsgGeometry` feeds both paths, so surface classification exists in exactly one place. Today it exists twice and disagrees:
  - Python checks "horizontal" first (`convert2openmc.py:20-110`).
  - C++ checks "vertical" first (`compute_surface.cpp:221-233`).

### 4. Bugs the port must not copy (verified)

| Bug | Location | Fix |
|---|---|---|
| Reflective boundaries go to the wrong surfaces: signed boundary IDs are used as list indices, and negative IDs index from the end | `convert2openmc.py:168-169` | Use the unsigned ID; test `boundary_type` |
| `assert mesh.has_boundary_layer` is missing `()` | `convert2degas2.py:112` | Call it |
| `assert tol < 1e-6` rejects the documented default of 1e-6 | `convert2openmc.py:212` | Range check with `ValueError` |
| Dummy material is referenced in `geometry.xml` but never exported | `convert2openmc.py:171-176`, `:216-217` | Void default |
| `inorout`/`inoroutWline` silently return 0 | `compute_surface.h:156`, `:216` | Device error counter, then a host exception |
| Device `OMEGA_H_CHECK` and per-entity `printf` abort the process or truncate output | `compute_surface.cpp:66`, `:105`, `:683` | Same as above |
| Tolerance defaults disagree (1e-10 vs 1e-6) | header vs C API vs CLI | One options struct holds all defaults |
| Last zone volume computed as "total minus the sum of the others", which loses precision | `convert2degas2.py:624` | Compute it directly |

Order of work: **characterize, then fix, then freeze, then port.**
1. Capture goldens (reference outputs) from the current code.
2. Fix each bug in Python in its own PR, reviewing the golden diff each time.
3. Freeze the goldens.
4. Port to C++.

### 5. Errors and debugging on device

- **Kernels never abort.** They count violations with a reduction and record the first bad entity with `atomic_min`. The host then throws `std::runtime_error`, which reaches Python as `RuntimeError` with the entity ID.
- **No debug flags in kernel signatures.** `print_flag` branches cost registers even when the flag is off, and on large meshes they print billions of lines. Debug output moves to `template<bool Debug>` kernels or to a host dump of a few entities.
- **Visible configuration.** Every kernel is labelled `omegah2csg::<stage>`, and each stage is wrapped in `Kokkos::Profiling::pushRegion`. `--version --verbose` and `show_config()` report:
  - the Kokkos backend and thread count;
  - the Omega_h version;
  - the paths of libomega_h and `_core`.

  The native executables also accept omega-h's `--osh-time` profiling flag and Kokkos flags such as `--kokkos-num-threads`, because they construct the `Library` from `argc`/`argv`.

### 6. Runtime ownership without vendoring PyOmega_h

These rules satisfy "don't vendor, don't build PyOmega_h" and still use omega-h's own runtime manager.

- **One owner type: `Omega_h::Library`.** I checked how it handles initialization and shutdown (`Omega_h_library.cpp:82-90`, `:150-160`, `:189-218`):
  - Its constructor starts MPI and Kokkos only if they are not already running.
  - Its destructor finalizes only what it started. It destroys omega-h's memory pool before calling `Kokkos::finalize`.

  PyOmega_h's own import-time initialize/finalize follows the same "only finalize what you started" rule (`PyOmega_h_runtime.cpp:24-59`). The two can coexist in one process in either import order.
- **Native executables:** `int main(int argc, char** argv) { Omega_h::Library lib(&argc, &argv); … }`.
- **Python `_core`:**
  - On first use, `_core` creates one process-wide `Omega_h::Library`. If PyOmega_h was imported first, the Library sees that Kokkos and MPI are already running, owns nothing, and PyOmega_h stays the owner.
  - The Library is released by a `Py_AtExit` hook. That runs after Python objects are gone, so no device array outlives `Kokkos::finalize`. This is the same mechanism PyOmega_h uses.
  - An optional `omegah2csg.initialize(kokkos_args=[...])` lets scripts pass Kokkos options before first use.
- **Lifetime rules:**
  - No static Kokkos or Omega_h data in `_core`.
  - Result objects hold device arrays only while alive.
  - Call `ask_up` and the other adjacency builders before releasing the GIL, because they lazily modify the mesh cache.
- **MPI:**
  - Wheels build omega-h with MPI off.
  - Native and spack builds may have MPI on. The tool is serial: it runs on rank 0 and refuses `mpirun -n >1` with a clear message.

### 7. PyOmega_h interop (optional, source/spack builds only)

- **What `_core` binds.**
  - Primary API: path-based. `convert_to_openmc(path, …)`, `convert_to_degas2(path, …)`, `compute_csg(path, …)`.
  - Mesh overloads: the same functions are also bound to take `Omega_h::Mesh&`. `_core` never registers `Omega_h::Mesh` itself. pybind11 looks up the type when the function is called, so these overloads only match when PyOmega_h's `OmegaHMesh` is registered in the same pybind11 internals. In a wheel, or when the pybind11 version differs, passing a PyOmega_h mesh gives a normal `TypeError`. `_core` adds a message: "mesh interop needs omegah2csg and PyOmega_h built against the same pybind11 and libomega_h; pass a file path instead".
- **Identity guard against two copies of libomega_h.** A user might pip-install our wheel, which bundles its own libomega_h, and also have spack's PyOmega_h on `PYTHONPATH`. If the pybind11 versions happen to match, the mesh type would be accepted, but it would come from a different libomega_h and Kokkos runtime. The fix:
  - Before using a passed-in mesh, `_core` compares, via `dladdr`, the libomega_h that its own code comes from with the one that owns the mesh's code.
  - On mismatch it raises a `TypeError` naming both paths.
  - This check is cheap and runs once per call.
- **pybind11 version.** Pin to the version used to build the spack omega-h, which you control and will update. For source builds, CMake reads the `__pybind11_internals_v*` string from the PyOmega_h `.so` it finds and warns if it does not match our pybind11. It warns rather than fails, because the path API works regardless. Wheels have no PyOmega_h to match, so they use the same pinned version for consistency.
- **Session helper.** `omegah2csg.read_mesh(path)` returns a PyOmega_h mesh only when PyOmega_h is importable. It reuses PyOmega_h's `OmegaHLibrary` so there is only one Library per process. This follows the rule in omega-h's `pytest/conftest.py`.
- **CUDA build notes.**
  - `_core` needs `NO_EXTRAS` on CUDA, as omega-h's own module does (`src/CMakeLists.txt:900-905`), plus `kokkos_compilation(TARGET …)`.
  - Writer code that includes no Omega_h or Kokkos headers builds with the host compiler.

### 8. netCDF from C++ (DEGAS2 requirements checked in `~/degas2/src`)

**What DEGAS2 reads.**
- DEGAS2 opens `geometry.nc` with the netCDF v2 Fortran call `ncopen(file, NC_NOWRITE)` (`geometry.web:1389`). This detects the format automatically, so DEGAS2 accepts any format its netCDF library can read:
  - classic (CDF-1) and 64-bit offset (CDF-2): every netCDF ≥ 3.6;
  - CDF-5: netCDF-C ≥ 4.4;
  - netCDF-4/HDF5: only with an HDF5-enabled build.
- Your DEGAS2 build links netCDF 4.10.0 from the spack `xgc-dev` view, which supports all four.

**What DEGAS2 writes.**
- DEGAS2's own geometry writer uses classic CDF-1 (`geomsetup.web:1266`).
- `definegeometry2d` writes its polygon file as 64-bit offset (`definegeometry2d.web:1555`, `:2143`).
- HDF5 is an option in DEGAS2, not a requirement.

**Types and dimensions.**
- `NC_INT` maps to 4-byte `NC_LONG` and `NC_REAL` to `NC_DOUBLE` (`netcdf.hweb`).
- `dimen_inq` asserts that dimension sizes match exactly. Our writer must produce `int32` and `double` with DEGAS2's exact dimension names and sizes.

**Recommendation.**
- **Format:**
  - Default to CDF-2 (64-bit offset). Every DEGAS2 build can read it, it needs no HDF5, and it removes the 2 GiB file limit of CDF-1.
  - At multi-million-triangle sizes each variable stays well under CDF-2's per-variable limit of about 4 GiB. Example: `surface_coeffs` at 10M surfaces is about 0.8 GB.
  - The writer still checks sizes and switches to CDF-5 with a log message if a variable would exceed the limit.
  - Add a `--nc-format {cdf2,cdf5,netcdf4}` flag.
- **Current code:** it writes `NETCDF4_CLASSIC` (`convert2degas2.py:150`), which forces HDF5 for no benefit. Its "2GB" FIXME (`:149`) applies only to CDF-1.
- **Library:**
  - Use netcdf-c directly behind a small RAII wrapper. netcdf-cxx4 is a thin wrapper that is rarely updated.
  - The wheel's netcdf-c is built without HDF5, which also avoids loading a second HDF5 next to `h5py`, which `openmc` imports.
- **Memory:** write one variable at a time (copy to host, `nc_put_var`, free), so peak host memory is the largest single variable.
- **Dependencies:** `netCDF4` becomes a test-only dependency, and `openmc` an optional extra.
- **Acceptance test:** run DEGAS2's `defineback`/`flighttest` on the converted `tests/assets/simple_degas2_case`. It runs on demand locally and on a self-hosted CI job with the spack DEGAS2.

### 9. Native executables and the Python CLIs

- **Native executables.** CMake builds `convert2openmc` and `convert2degas2` as native programs linked to the `pipeline` target.
- **Python console scripts.** The wheel's console scripts have the same names and call `_core`.
- **Flags.** Both front ends build their flags from the same C++ `OpenMCOptions` and `Degas2Options` structs, which also hold every default. A test checks that both expose identical flags and defaults.
- **Name clashes.** In a spack environment, whichever comes first on `PATH` runs. `--version --verbose` reports which front end and which build is running.
- **CLI fixes:**
  - add `-o/--output` and `--output-dir`, with the aux files written next to the output;
  - write to a temporary file, then rename;
  - `--force` to overwrite;
  - reserve `-v` for verbose (today it means `--version` in `convert2degas2`);
  - add `--check` to list missing tags (`offset_face`, `isOnWall`) before any work.

### 10. Wheels

**Contents, decided.** The wheel ships `_core`, the Python console scripts, **and the native executables** under `omegah2csg/bin/`. Executable and CLI names are to be decided later. Python entry points can `os.execv` the bundled binary, or call `_core` directly. auditwheel repairs the executables as well. A wheel test runs each executable's `--version` from a clean virtualenv.

**Stage A: now, before omega-h publishes PyOmega_h on PyPI.**
- libomega_h, Kokkos (OpenMP) and netcdf-c (no HDF5) are built once in `before-all` and bundled by auditwheel.
- The wheel contains no PyOmega_h. Input is file paths, and mesh-object interop is a spack/source feature.

**Stage B: once the PyOmega_h wheel exists.** This is the target design: `omegah2csg` depends on `PyOmega_h` (pinned to a compatible range) and stops bundling omega-h.
- **Build.** Our build adds the PyOmega_h wheel to `build-system.requires` and finds omega-h from it. For example, `CMAKE_PREFIX_PATH` comes from a helper such as `python -c "import PyOmega_h; print(PyOmega_h.cmake_dir())"`. This is the same approach PyTorch extensions take with torch.
- **Repair.** `auditwheel repair --exclude 'libomega_h*' --exclude 'libkokkos*'` keeps those libraries out of our wheel. The RPATH of `_core` and the executables points into the installed PyOmega_h package, for example `$ORIGIN/../PyOmega_h/lib`. The process then holds exactly one libomega_h, one Kokkos and one pybind11 type registry, so mesh-object interop works from pip as well.
- **Version match.** The pybind11 version and the compiler and manylinux image must match the PyOmega_h wheel's. CI reads them from the installed PyOmega_h wheel metadata and fails on a mismatch.

**What we need the PyOmega_h wheel to ship.** This is input for omega-h's own packaging work, not a change this project makes:
- the shared `libomega_h` and Kokkos libraries;
- the headers;
- `Omega_hConfig.cmake` (and Kokkos' config), relocatable;
- a way to find them from Python;
- the pybind11 version it was built with.

If it ships only the extension module, Stage B is not possible and Stage A continues.

**GPU.** Wheels are OpenMP-only in both stages. GPU users build from source or spack.

### 11. Testing, CI and migration

- **CI does not protect `main` today.** `ci.yml` and `format-check.yml` run only on `parallel` and `release/*`, and the wheel workflow runs only by hand. Fixing this is phase 0.
- **The goldens are too weak to catch a port regression.** `gold-geometry.nc` is checked for only 4 of about 60 variables, and `test_convert2degas2` checks nothing. Add, all captured from a frozen baseline commit:
  - full netCDF and aux-file goldens;
  - a semantic `geometry.xml` comparison, plus a point check: the cell found at each triangle centroid must be that face;
  - `.npz` array goldens at several tolerances.
- **Tolerances across backends.** Integers compare exactly. Floats use a per-variable tolerance table, because CUDA fused multiply-add and different reduction order change the last bits. A test checks that no golden edge sits within 100·eps of `tol`.
- **GPU coverage, in three tiers:**
  - hosted Serial/OpenMP runs at 1 and 4 threads, plus a bounds-check debug build;
  - a CUDA compile-only container job;
  - GPU runtime tests on a self-hosted SCOREC runner.
- **Performance.**
  - Benchmark synthetic tagged meshes from 1e4 to 1e7 triangles, plus real XGC/DEGAS2 meshes stored outside git with checksums.
  - Measure per-stage wall time, peak memory, and the number of host↔device copies (counted with a Kokkos Tools callback).
  - Hosted CI runs a scaling check, t(1e5)/t(1e4) < 15, which catches O(n²) code. The self-hosted runner fails on a regression of more than 15%.
- **Rollback.**
  - Keep a `release/0.0.x` branch.
  - Add a hidden per-converter `OMEGAH2CSG_BACKEND={cpp,python}` switch: `python` by default for one release, then `cpp`.
  - The Python converters then move to `tests/reference/` as the oracle.

### 12. Module layout and tooling

| Layer | Module | Notes |
|---|---|---|
| L0 | `geom` (header-only) | `OMEGA_H_INLINE` surface math and sidedness; testable on the host |
| L1 | `mesh_input`, `topology` | File reading (rejects `nparts > 1` with a message); `Topology` built once |
| L2 | `csg`, `wall`, `cell_quantities` | Device kernels over `Topology` |
| L3 | `openmc` and `degas2` assemblers | Produce `CsgGeometry` / `Degas2Geometry` |
| L4 | `openmc_xml`, `degas2_netcdf`, `degas2_aux` writers | Host only; no Omega_h/Kokkos headers; optional CMake components |
| L5 | `pipeline` | `convert_to_openmc`, `convert_to_degas2`, `compute_csg` |
| apps | `convert2openmc`, `convert2degas2` | Native executables |
| — | `_core` | pybind11: pipeline, result objects, optional mesh overloads. No kernels |

**Tooling:** ruff and clang-format through pre-commit, one CI workflow and one wheel workflow (sdist plus Trusted Publishing), CTest/Catch2 plus pytest, and committed `_core.pyi` stubs. Defer the Sphinx site. Write `design_choices.rst` and `references.rst` now, as you asked.

**Plan rewrite (rev 2):**
- Order: status → goals/constraints/targets → decision log → architecture (data-flow diagram showing host and device) → modules → Python layer and migration table → layout → build and packaging → tests → CI → phases with exit criteria → risks.
- Remove local paths and the "upstream omega-h contributions" section.
- Add a glossary.

## Goals vs. Non-Goals

* **In-scope goals:**
  - All computation in C++/Kokkos on OpenMP and CUDA, with data on the device until output.
  - Speed on multi-million-triangle meshes as the main metric.
  - Result arrays readable and editable from Python, plus Python builders for `openmc` objects.
  - C++ writers for `geometry.xml` and DEGAS2 netCDF/aux files.
  - Native executables and Python CLIs that share the same options.
  - Optional PyOmega_h mesh interop in spack/source builds.
  - Full-output goldens and per-backend testing.
  - `design_choices.rst` and `references.rst`.
* **Out-of-scope non-goals:**
  - Partitioned meshes. They are rejected with a clear message; XGC/DEGAS2 meshes are generally not partitioned.
  - Distributed computation.
  - Any change to omega-h.
  - Vendoring or building PyOmega_h.
  - GPU wheels.
  - Zero-copy NumPy views of device memory.
  - A full Sphinx site in 0.1.
  - Backward compatibility with the 0.0.x `OmegaHMesh`/`kokkos_runtime` API (clean break with a migration table).

## Alternatives Weighed

| Alternative | Why rejected |
|---|---|
| Cell quantities in NumPy on PyOmega_h (rev 1) | Element-by-element host copies; would move existing device kernels to the host |
| Keep the converters in Python (rev 1) | Per-triangle and per-edge Python loops, including O(N_edges × N_wall) |
| ctypes + `extern "C"` (current) | Exceptions cross the C ABI (undefined behaviour); hand-written prototypes; manual marshalling |
| nanobind | Cannot share types with pybind11, so no PyOmega_h interop |
| Vendor PyOmega_h in wheels | Your decision: it shadows or duplicates a user's PyOmega_h and its libomega_h/Kokkos, and needs per-Python omega-h builds. It is replaced by depending on omega-h's own PyPI wheel (Stage B) |
| Depend on the PyOmega_h wheel but also bundle our own libomega_h | Two copies of libomega_h and Kokkos in one process; a mesh from one would be handled by code from the other |
| Build PyOmega_h's sources in our CMake | Your decision: it couples us to omega-h's binding source list and build internals |
| Require a PyOmega_h mesh object as the only input | Ties every user to pybind11 ABI matching. Path input works everywhere; mesh input is an optional extra |
| Our own Kokkos/MPI init and finalize code | `Omega_h::Library` already starts and finalizes only what it owns, and coexists with PyOmega_h |
| One Library per mesh (current `OmegaHMesh`) | Repeated init and finalize; omega-h's tests show that multiple Libraries cause MPI-finalize errors |
| OpenMC output only as `geometry.xml` | You require Python access to cells and arrays |
| OpenMC output only as Python objects | Per-object Python cost is minutes at multi-million cells |
| netCDF-4/HDF5 output by default | DEGAS2 does not need it (its own geometry writer uses classic); a second HDF5 copy can crash next to `h5py` |
| netcdf-cxx4 | Thin, rarely updated wrapper; netcdf-c is enough |
| Unpinned `pybind11>=3.0` | Breaks type sharing with PyOmega_h when versions drift |
| Upstream omega-h binding additions | Conflicts with the no-omega-h-changes constraint; adapt later if omega-h adds them |
| "Only `parallel_for`" as a strict rule | Compaction and counting need scan and reduce; use Omega_h's tested primitives |

## Key Decisions & Staging for Next Steps

**Decided:**
1. **C++ pipeline.** The full pipeline runs in C++. Result objects expose arrays that copy to the host on request and can be edited. There are two output paths: Python builders, and C++ writers. Both are fed from one `CsgGeometry`.
2. **Spaces and stages.** Memory and execution always use `Omega_h::Space` and `Omega_h::policy`. Topology is built once, and the stage chain is explicit.
3. **Errors.** Kernels never abort: device error counters lead to host exceptions.
4. **Runtime ownership.** `Omega_h::Library` owns Kokkos and MPI, in both the native executables and `_core`. In `_core` it is created lazily and released with `Py_AtExit`, and it defers to PyOmega_h when PyOmega_h was imported first.
5. **No PyOmega_h in wheels or in our CMake.**
   - The main input is a file path.
   - PyOmega_h mesh overloads are optional, matched by pybind11 at call time, with a `dladdr` identity guard against two copies of libomega_h.
   - pybind11 is pinned to the spack omega-h's version, with a configure-time warning on mismatch.
6. **Native executables** `convert2openmc`/`convert2degas2`, plus Python console scripts that share the same options structs.
7. **DEGAS2 netCDF format.** CDF-2 by default, CDF-5 automatically when needed, netCDF-4 on request. Variables are `int32`/`double`, with DEGAS2's exact dimensions.
8. **Partitioned meshes** are out of scope and rejected with a clear message.
9. **Order of work.** Characterize, then fix, then freeze, then port. The Python converters stay as the test oracle until C++ matches.
10. **Phase 0.** CI on `main`, and commit the current `fix-build-error` diff.

11. **Wheels ship the native executables** as well as the Python console scripts. Names for both will be chosen later.
12. **Benchmarks** go up to about 1e7 triangles, and include real XGC/DEGAS2 meshes.
13. **Two-stage wheel packaging.**
    - Stage A (now): bundle libomega_h, Kokkos and netcdf-c; file-path input only.
    - Stage B (after omega-h's CI/CD publishes PyOmega_h to PyPI): depend on the PyOmega_h wheel, link to its libomega_h and Kokkos, and keep them out of our wheel with `auditwheel --exclude`. This enables mesh-object interop from pip.

**Open items for `/egm-draft`:**
1. Names for the executables and CLI tools.
2. Whether the PyOmega_h wheel will ship shared libraries, headers and a relocatable CMake config. This decides whether Stage B is possible; coordinate with omega-h's packaging work.
