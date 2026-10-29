# omegah2csg 0.1: C++/Kokkos Conversion Pipeline with a Thin, Extensible Python Layer

*   **Author:** Fuad Hasan
*   **Status:** Proposed
*   **Date:** 2026-10-28
*   **Predecessor:** [cpp-pipeline-modernization.md](./cpp-pipeline-modernization.md) (brainstorm brief, rev 2)

--------------------------------------------------------------------------------

## Overview

omegah2csg converts 2D Omega_h meshes of tokamak cross-sections into constructive solid geometry (CSG) for two Monte Carlo codes:

*   **OpenMC:** the result is available either as a `geometry.xml` file or as in-memory `openmc` objects.
*   **DEGAS2:** the result is available either as a `geometry.nc` netCDF file plus auxiliary text files, or as the individual DEGAS2 geometry arrays.

In both cases users can get the underlying arrays (surfaces, cells, boundaries, zones, sectors and the rest), modify or extend them in Python, and then build the OpenMC objects or write the files from the modified arrays.

The input meshes come from XGC/DEGAS2 workflows. They are usually not partitioned and contain several million triangles. Other tools already do this conversion but are slow, so conversion speed at these mesh sizes is the reason this project exists.

Version 0.1 restructures the project around one rule: **all computation happens in C++ on the device, and Python is a thin layer on top.**

*   **C++ library.** It reads the mesh once and computes every result with Kokkos kernels in Omega_h's memory and execution spaces. It runs on OpenMP (CPU) or CUDA (GPU). Data stays on the device until a result is written or requested. Results come back as two data objects, `CsgGeometry` and `Degas2Geometry`.
*   **Two front ends share that library:** native executables (one per converter) and a pybind11 module, `omegah2csg._core`.
*   **Python layer.** Users call the converters from scripts and OpenMC workflows, read and edit the result arrays, and either build `openmc` objects or write the files.
*   **Runtime.** Omega-h's own `Omega_h::Library` starts and stops Kokkos and MPI.
*   **Removed and unchanged.** The ctypes/C-API layer from 0.0.x is removed. Omega-h itself is not changed.

## The Problem

**The current implementation is slow where it matters, and fragile where it crosses language boundaries.**

1.  **Most of the work runs in Python, one element at a time.**
    *   Only edge coefficients, face connectivity and a few helpers run as C++ kernels.
    *   The DEGAS2 converter (`convert2degas2.py`, 915 lines) loops in Python over every triangle and every edge. One loop calls `np.isin` against the wall-edge list for each edge (`convert2degas2.py:689-698`), so its cost grows with edges × wall edges.
    *   The OpenMC converter creates one Python object per surface and per cell (`convert2openmc.py:152-204`).
    *   At several million triangles, these loops dominate the run time.

2.  **The existing C++ code makes extra copies and recomputes work.**
    *   Every C-API function copies device results to host buffers one element at a time (`capi.cpp`).
    *   `Omega_h::Mesh` is passed by value (`compute_surface.cpp:129`, `:670`), so Omega_h's cached adjacency is discarded and rebuilt.
    *   Wall edges are computed twice (`compute_surface.cpp:97`).

3.  **Some kernels allocate and launch in the wrong Kokkos spaces.** Omega_h runs on `Kokkos::OpenMP` whenever OpenMP is enabled, even in CUDA builds, and may use shared or pinned memory (`Omega_h_kokkos.hpp:29-49`). Our code allocates views and launches scans and reductions in Kokkos's *default* spaces (`compute_surface.cpp:73`, `:128-141`, `:262`; `capi.cpp:36-38`, `:461`). In builds where these differ, kernels touch memory they cannot reach.

4.  **Failures crash the process instead of reporting an error.** For a tool that runs inside OpenMC Python sessions, a bad mesh must raise an error, not kill the interpreter. Today:
    *   C++ exceptions cross the `extern "C"` boundary into ctypes, which is undefined behaviour.
    *   Device-side `OMEGA_H_CHECK` aborts the process.
    *   Two sidedness helpers silently return 0 on bad input (`compute_surface.h:156`, `:216`).

5.  **The Python wrapper duplicates runtime management that omega-h now provides.**
    *   The project manages Kokkos start-up and shutdown itself (`config.py`), and its finalize calls `Kokkos::Tools::finalize`, not `Kokkos::finalize`.
    *   It creates a new `Omega_h::Library` for every mesh.
    *   Omega-h's `Library` and its Python bindings (PyOmega_h) now handle this, and they tolerate other owners.

6.  **There are correctness bugs that no test catches.**
    *   OpenMC marks the wrong surfaces as reflective. Signed boundary IDs are used as list indices, so negative IDs count from the end (`convert2openmc.py:168-169`).
    *   The DEGAS2 boundary-layer assertion never fires, because the method is not called (`convert2degas2.py:112`).
    *   The documented default tolerance fails the converter's own assertion (`convert2openmc.py:212`).
    *   Test coverage of outputs is thin:
        *   The DEGAS2 converter's output is never compared to its golden file (`tests/test_degas2_mesh_functions.py:155-156`).
        *   Only 4 of about 60 netCDF variables are checked anywhere.
        *   No test checks the OpenMC output.

7.  **CI does not protect the main branch.**
    *   CI and format checks run only on the `parallel` and `release/*` branches.
    *   Wheels are built only on manual dispatch.

**Proposed solution:**
*   **Pipeline.** Move the whole conversion into one C++ pipeline of small, separately testable stages. Every stage runs on device data in Omega_h's spaces.
*   **Results.** Expose the results as array-backed objects that Python users can inspect, modify and extend.
*   **Outputs.** Write both output formats from C++, and also let users build OpenMC objects or DEGAS2 files from (possibly modified) arrays in Python.
*   **Migration.** Lock the current behaviour down with full-output golden tests before porting, fix the known bugs one at a time, then port.

## Goals

*   **G1: All computation runs in C++ on the device.**
    *   No per-element Python loops remain in either converter.
    *   Data moves from device to host only at output or when Python asks for an array, with one copy per array.
    *   Verified with a Kokkos Tools counter of device↔host copies on the GPU CI runner.
*   **G2: Run time scales linearly with mesh size.**
    *   Every stage scales linearly with mesh size on OpenMP and CUDA. CI checks that t(1e5 triangles) / t(1e4 triangles) < 15 for each converter.
    *   Benchmarks cover synthetic meshes up to 1e7 triangles plus real XGC/DEGAS2 meshes. They report per-stage time, peak host and device memory, and the copy count.
*   **G3: No speed regressions.** The self-hosted benchmark job fails when any stage is more than 15% slower than the stored `main` baseline.
*   **G4: Output matches the frozen goldens.** After the bug fixes, C++ output matches the frozen golden outputs on all test meshes:
    *   every DEGAS2 netCDF variable and aux file;
    *   `geometry.xml`, compared by meaning rather than by bytes;
    *   intermediate arrays.

    Integers match exactly. Floating-point values match within a documented per-variable tolerance.
*   **G5: Python access to the result arrays.**
    *   Every array in `CsgGeometry` and `Degas2Geometry` is readable and writable from Python as NumPy.
    *   Users can build an `openmc.Universe` from these arrays. Materials can be given as one material, a tag-to-material map, or a per-cell array.
    *   Users can write a DEGAS2 file from edited arrays.
*   **G6: The two OpenMC paths agree.** The C++ `geometry.xml` writer and the Python `openmc` builder produce the same geometry for the same input. Verified by a test.
*   **G7: Errors become exceptions, not crashes.**
    *   Mesh and geometry errors raise exceptions that name the failing entity. In Python this is `RuntimeError` or `ValueError`; the executables exit non-zero with a message.
    *   No kernel aborts the process.
*   **G8: Two front ends with identical options.**
    *   Native executables and Python console scripts read their flags and defaults from the same C++ options structs. Verified by a test.
    *   The wheel ships both.
*   **G9: One runtime owner per process.**
    *   `Omega_h::Library` is the only runtime owner in our code.
    *   It works alone or alongside PyOmega_h, in either import order, without initializing or finalizing anything twice.
*   **G10: DEGAS2 can read our files.**
    *   DEGAS2 files are written in a format every DEGAS2 build can read: CDF-2 by default.
    *   They use DEGAS2's exact dimension names, sizes and types (`int32`, `double`).
    *   Verified by running DEGAS2's own tools on the converted test case.
*   **G11: CI runs on every change to `main`.**
    *   CI runs on every push and pull request to `main`, covering Serial and OpenMP execution plus a CUDA compile-only build.
    *   A self-hosted runner covers GPU runtime tests and benchmarks.

## Non-Goals

*   **Partitioned meshes.** Reading `.osh` meshes with more than one part is out of scope. They are detected before omega-h is called and rejected with a clear message.
*   **Distributed (multi-rank) computation.** The tool is serial. Under MPI with more than one rank, it refuses to run.
*   **Changes to omega-h.** This project adapts to omega-h as it is.
*   **Vendoring PyOmega_h or building its sources here.** Whether pip users can pass PyOmega_h mesh objects depends on omega-h publishing its own PyPI wheel. See the Technical Plan.
*   **GPU wheels on PyPI.** Wheels are OpenMP-only. GPU users build from source or spack.
*   **Zero-copy NumPy views of device memory.** Each array access from Python is one explicit copy.
*   **Fast Python `openmc` object construction.** It is limited by OpenMC's Python API. Large meshes should use the C++ `geometry.xml` writer.
*   **Backward compatibility with the 0.0.x Python API.** `OmegaHMesh`, `kokkos_runtime` and the `get_*` methods are removed. A migration table maps old names to new ones.
*   **A full Sphinx documentation site in 0.1.** Only `design_choices.rst` and `references.rst` are written now.

## Open Questions

*   **Naming.** What will the native executables and the CLI tools be called? The owner has deferred this.
*   **PyOmega_h wheel contents.** Wheel Stage B links against PyOmega_h's libomega_h and Kokkos instead of bundling them. It is only possible if the PyOmega_h PyPI wheel ships:
    *   the shared libomega_h and Kokkos libraries;
    *   headers;
    *   a relocatable `Omega_hConfig.cmake`;
    *   the pybind11 version it was built with.
*   **Absolute performance target.** G2 and G3 measure scaling and regressions, but there is no absolute time target, such as "convert a 1e7-triangle mesh in under N seconds on one A100 / one 64-core node". Setting one needs a baseline of the current 0.0.2 code on the same meshes.
*   **Editable arrays and consistency.** Users can modify arrays before building or writing, so which checks run on modified arrays?
    *   Examples: index ranges, sign values, and agreement between related arrays (cell → surface references, zone counts).
    *   Writers must not produce a file that DEGAS2 or OpenMC rejects with a cryptic error.

## Critical User Journeys (CUJs)

### CUJ 1: Convert a large mesh for DEGAS2 from the command line

*   **Given**:
    *   A user on a GPU node with a spack build of omegah2csg (CUDA).
    *   A non-partitioned XGC/DEGAS2 `.osh` mesh with several million triangles, tagged with `offset_face` and `isOnWall`.
*   **When**: The user runs the native DEGAS2 converter with an output path and, optionally, Kokkos flags.
*   **Then**:
    *   The tool reads the mesh and runs all stages on the GPU.
    *   It writes a CDF-2 `geometry.nc`, using a temporary file followed by a rename. When `--aux-files` is given, it writes the aux files to `--aux-dir`, which defaults to the directory of the `.nc` file.
    *   It prints per-stage timings when asked.
    *   DEGAS2's `defineback` and `flighttest` read the file without errors.
    *   On a mesh with missing tags, it exits non-zero and names the missing tag before any kernel runs.

### CUJ 2: Modify DEGAS2 geometry arrays from Python before writing

*   **Given**: A user with the omegah2csg wheel and a tagged mesh.
*   **When**:
    *   The user calls the Python API on the mesh path and gets a `Degas2Geometry`.
    *   They change some arrays in NumPy, for example wall-zone properties, sector or diagnostic definitions, or surface assignments.
    *   They write the file.
*   **Then**:
    *   The written `geometry.nc` contains the modified values.
    *   The arrays the user did not change are identical to a direct C++ conversion.
    *   If an edit breaks a consistency rule (for example, an index out of range), the write fails with a `ValueError` that names the array and the index.

### CUJ 3: Build an OpenMC model from Python with custom materials

*   **Given**: A user with `pip install omegah2csg[openmc]` (OpenMP wheel) and an OpenMC Python script that defines materials.
*   **When**:
    *   The user calls the Python API on a mesh path and gets a `CsgGeometry`.
    *   They edit the boundary types or material assignments in NumPy.
    *   They build an `openmc.Universe`, passing a tag-to-material map and ID offsets.
*   **Then**:
    *   The `openmc.Universe` has one cell per mesh face.
    *   Boundary surfaces have the requested boundary type.
    *   The universe can be placed inside a larger model without ID collisions.
    *   Its geometry matches what the C++ `geometry.xml` writer produces from the same arrays.

### CUJ 4: Write OpenMC geometry for a very large mesh without Python objects

*   **Given**: A multi-million-triangle mesh and a per-cell material-ID array.
*   **When**: The user runs the native OpenMC converter, or the Python API's direct-write function, with that array.
*   **Then**:
    *   `geometry.xml` is written by C++ without creating Python `openmc` objects.
    *   Writing time grows linearly with mesh size.
    *   The file loads in OpenMC together with the user's own `materials.xml` and `settings.xml`.

### CUJ 5: Pass an in-memory PyOmega_h mesh (spack/source build)

*   **Given**: omegah2csg and PyOmega_h are built in the same spack environment, against the same libomega_h and pybind11.
*   **When**: The user reads or adapts a mesh with PyOmega_h and passes the mesh object to an omegah2csg function.
*   **Then**:
    *   omegah2csg uses the mesh directly, with no file round trip, and reuses PyOmega_h's runtime instead of starting its own.
    *   If omegah2csg and PyOmega_h link different libomega_h copies or pybind11 versions, the call raises a `TypeError` that names both paths and suggests passing a file path instead.

### CUJ 6: A developer changes a stage safely

*   **Given**: The golden outputs are frozen and CI runs on `main`.
*   **When**: A developer changes one stage, for example the face-side kernel.
*   **Then**:
    *   Unit tests for that stage run without loading a mesh file.
    *   Golden comparisons for both outputs run on Serial and OpenMP at 1 and 4 threads.
    *   The CUDA compile-only job catches host/device mistakes.
    *   A difference beyond tolerance fails the pull request, naming the variable and the first differing index.

## Technical Plan (Architecture)

### 1. Data flow and where data lives

```text
                      DEVICE (Omega_h::Space: CUDA, or host memory under OpenMP)
 .osh file ──read──▶ Omega_h::Mesh ──extract──▶ Topology ──▶ CSG stages ──▶ CsgGeometry ──┐
   (host)            (nparts==1,                (built once)  edge surfaces              │
                      checked before read)                    face sides                 │
                                                              boundary / wall            │
                                                              cell quantities            │
                                                                     │                   │
                                                                     ▼                   │
                                                              DEGAS2 assembler ─▶ Degas2Geometry
                                                                                         │
 ───────────────────────────── one device→host copy per array, on demand ────────────────┼─────
                      HOST                                                               ▼
   C++ writers (no Omega_h/Kokkos headers)            Python (pybind11 _core)
   ├─ openmc_xml (pugixml) ─▶ geometry.xml            ├─ NumPy views of each array (copied once, cached, editable)
   ├─ degas2_netcdf (netcdf-c) ─▶ geometry.nc         ├─ openmc builder (pure Python) ─▶ openmc.Universe
   └─ degas2_aux ─▶ wallfile/plasmafile/sourcefile    └─ write_*(geom, path) ─▶ the same C++ writers
```

*   **Device-resident results.** Every array in `Topology`, `CsgGeometry` and `Degas2Geometry` is a flat `Omega_h::Write<T>` or `Read<T>` with a documented component count. They live in Omega_h's memory space, so they also use omega-h's memory pool.
*   **One copy at output.** Data crosses to the host once, at output. For each array, either a writer requests it, or Python reads it for the first time.
*   **Host-only writers.** The writers take plain host arrays: a pointer, a length and a component count. They never see `Omega_h` or Kokkos types, so they compile with the ordinary host compiler on CUDA builds. Arrays that a user edited in Python go through the same writers.

### 2. Layers and dependency rules

Dependencies point downward only. Each row is its own CMake target, so it can be tested alone.

| Layer | Module / target | Responsibility | May depend on |
|---|---|---|---|
| L0 | `geom` (header-only) | Per-entity math as `OMEGA_H_INLINE` functions: edge → surface coefficients, surface type, sidedness of a point, triangle area, centroid, toroidal volume, bounding box | `Omega_h_vector.hpp` only |
| L1 | `mesh_input` | Validate path, `nparts == 1` and format version, then read with `binary::read`. Check that the required tags exist | omega-h |
| L1 | `topology` | Build a `Topology` from a `Mesh&` once: coords, edge→vertex, face→vertex, face→edge, edge→face (CSR), `offset_face`, `isOnWall` | omega-h |
| L2 | `csg` | Edge surface type and coefficients, face sides, boundary edges | L0, L1 |
| L2 | `wall` | Wall edges, wall-adjacent faces, edge→face sorted by side | L1 |
| L2 | `cell_quantities` | Cell volumes, centroids, bounding boxes, universal-cell bounds and volume | L0, L1 |
| L3 | `degas2_assemble` | Build every `Degas2Geometry` array from L2 results | L2 |
| L4 | `openmc_xml` | `CsgGeometry` host arrays → `geometry.xml` via pugixml | pugixml only |
| L4 | `degas2_netcdf` | `Degas2Geometry` host arrays → netCDF via netcdf-c, driven by the schema table | netcdf-c only |
| L4 | `degas2_aux` | wallfile / plasmafile / sourcefile text writers | none |
| L5 | `pipeline` | `compute_csg`, `compute_degas2`, `convert_to_openmc`, `convert_to_degas2`. Chains the stages and runs the device→host copy and the writers | all of the above |
| apps | native executables | Parse flags into the options structs, construct `Omega_h::Library(&argc, &argv)`, call `pipeline` | L5 |
| py | `_core` (pybind11) | Bind the options structs, result objects, pipeline functions and writers. Contains no kernels | L5 |

*   **No stage calls another stage.** Only `pipeline` chains stages, so every stage's inputs are explicit. This rules out hidden recomputation, such as computing wall edges twice.
*   **L0 functions run on the host as well.** Catch2 can test them on hand-written points without a mesh or a device.
*   **L2 stages take a `Topology`, not a `Mesh`.** Unit tests can build a `Topology` by hand for small cases: a single triangle, a vertical edge, a cone edge.

### 3. Data model

**`Topology`** is built once and never copied by value. Arrays are flat and use `LO` indices.

| Field | Type | Size |
|---|---|---|
| `coords` | `Reals` | 2·nverts |
| `edge_verts` | `LOs` | 2·nedges |
| `face_verts` | `LOs` | 3·nfaces |
| `face_edges` | `LOs` | 3·nfaces |
| `edge_faces_offsets` | `LOs` | nedges+1 |
| `edge_faces` | `LOs` | — |
| `offset_face` | `LOs` | nfaces |
| `is_on_wall` | `LOs` | nverts |

**`CsgGeometry`** has named fields. Column meaning is defined once in `geom`, with named index constants instead of bare numbers.

| Field | Contents |
|---|---|
| `surface_type` | `I8[nedges]`; values `ZPlane`, `ZCylinder`, `ZCone` |
| `surface_coeffs` | `Reals[6·nedges]`. Column constants: `kA`, `kB`, `kC`, `kD`, `kUpFlag`, `kSlope` |
| `face_edges` | `LOs[3·nfaces]` |
| `face_sides` | `I8[3·nfaces]`; ±1 |
| `boundary_edges` | `LOs[nbdry]`, unsigned |
| `boundary_sides` | `I8[nbdry]` |
| `cell_volume` | `Reals[nfaces]` |
| `cell_centroid` | `Reals[2·nfaces]` |
| `cell_bbox` | `Reals[4·nfaces]` |
| `wall_edges` | Wall data, filled only when the wall tags exist |
| `wall_faces` | Wall data, filled only when the wall tags exist |

*   **Signs stay out of IDs.** Signs and IDs are kept in separate arrays. The writers create DEGAS2's encodings (`edge_id * sign`, 1-based, signed) and OpenMC's halfspace signs. This removes the "edge 0 has no sign" limitation inside our code. File formats are unchanged.

**`Degas2Geometry`** holds one array per DEGAS2 netCDF variable, with the exact DEGAS2 names: `cells`, `boundaries`, `surfaces`, `neighbors`, `surface_coeffs`, `surface_points`, `surface_sectors`, `zone_*`, `sectors`, `sector_*`, `strata`, `diagnostic_*`, `de_*` and scalars such as `ncells` and `nsectors`.

*   **A schema table defines every variable.** `degas2_schema.hpp` lists each variable once: name, element type (`int32` / `double` / `char`), dimension names in the same order as today's writer, and a rule for each dimension size. One table drives:
    *   the netCDF writer (dimension and variable definitions);
    *   the Python binding (one property per variable);
    *   validation of edited arrays.
*   **Large arrays come from device kernels.** These are the arrays that scale with the mesh: `cells`, `boundaries`, `surfaces`, `neighbors`, `surface_coeffs`, `surface_points`, `surface_sectors`, `zone_*`, `sectors`, `sector_*`, `strata`, `diagnostic_sector_tab`.
*   **Small arrays are filled on the host from options.** These are fixed-size constants and per-run settings, such as detector placeholders, diagnostic group definitions, and target temperature and recycling coefficient.
*   **Today's Python loops become device kernels:**
    *   per-face `boundaries`: `parallel_for` over faces;
    *   per-edge `surfaces`: `parallel_for` over edges, plus an error count;
    *   the `surface_sectors` running counter: a mark kernel, then an exclusive scan, then a fill kernel. This replaces the O(N_edges × N_wall) `np.isin` loop;
    *   `sectors`, `sector_surface`, `sector_points` and `sector_type_pointer`: `parallel_for` over wall edges;
    *   the edge→face sort: `parallel_for` over edges.
*   **Two precision and ordering fixes.**
    *   The last zone volume is computed directly as a sum over its cells, not as "total minus the rest".
    *   Two of today's assumptions are checked once with a reduction: plasma faces numbered first, and a boundary face last. If either check fails, an error is raised instead of producing a silently wrong file.

### 4. Kernel rules

*   **Spaces.** Allocate only through `Omega_h::Write<T>`. Launch only through `Omega_h::parallel_for`, or `Kokkos::parallel_reduce` / `parallel_scan` with `Omega_h::policy(n)`. Use Omega_h's array helpers (`get_sum`, `offset_scan`, marking and compaction helpers) where they fit. A CI grep check rejects `Kokkos::View<` and policy-less `Kokkos::parallel_*` calls in `cpp/`.
*   **Errors.** Kernels never call `OMEGA_H_CHECK` or `printf` on error paths. Instead, each stage that can fail does two things:
    *   it counts violations with a `parallel_reduce`;
    *   it records the smallest failing entity with an atomic min.

    The host then throws `omegah2csg::GeometryError`, which names the stage, the entity and the reason. pybind11 maps it to `RuntimeError`. The executables print it and exit with code 2.
*   **Debug output.** Kernels take no debug flag. `--debug N` dumps the first N entities of each stage on the host, after the stage finishes.
*   **Profiling.** Every kernel is named `omegah2csg::<stage>::<kernel>`. Every stage runs inside `Kokkos::Profiling::pushRegion`. With `--timing`, the pipeline prints per-stage wall time using `Kokkos::Timer` with a fence.
*   **Size checks.** Before allocating, the host checks:
    *   counts fit in `LO`;
    *   DEGAS2 `i4` indices (`nwall + 3·ntri`) fit in int32;
    *   OpenMC IDs plus offsets fit in int32.

### 5. OpenMC output

*   **Shared tables.** Both outputs read the same `CsgGeometry` host arrays, plus three per-cell / per-surface inputs: material IDs, boundary types, and ID offsets.
*   **C++ writer (`openmc_xml`).**
    *   Builds a pugixml document with one `<surface>` per surface and one `<cell>` per face, then saves it.
    *   Cone surfaces are expanded the same way `openmc.model.ZConeOneSided` expands them: a cone surface plus a plane surface, with a region of the form `(cone_side plane_side)` or `(cone_side | plane_side)`. Both sides of this expansion are pinned by a golden generated with a fixed OpenMC version.
    *   Doubles are written with `std::to_chars`, the shortest form that reads back to the same value.
*   **Python builder (`omegah2csg.openmc_builder`, pure Python).**
    *   Builds `openmc.ZPlane`, `ZCylinder` and `model.ZConeOneSided` objects, plus `Cell`s, from the arrays.
    *   Accepts `materials` (a Material, a dict mapping tag value to Material, or a per-cell sequence), `boundary_type`, `cell_id_offset` and `surface_id_offset`.
    *   The default fill is void.
*   **Equivalence test.**
    *   Write XML with C++. Build the universe with Python and export it.
    *   Parse both, normalise IDs, and compare surfaces by type and rounded coefficients, and cells by their region.
    *   Classify each triangle centroid with `openmc.Geometry.find` and check the hit is that face's cell.
*   **Memory guard for pugixml.** The DOM can use several times the file size in memory. The benchmark records peak RSS for the XML stage. If at 1e7 triangles it exceeds 4× the file size or is the slowest stage, a hand-rolled writer replaces pugixml behind the same `openmc_xml` interface. No other code changes.

### 6. DEGAS2 output

*   **`degas2_netcdf`.**
    *   A small RAII wrapper over netcdf-c, driven by the schema table.
    *   Default format is CDF-2 (`NC_64BIT_OFFSET`).
    *   If any fixed-size variable would exceed CDF-2's limit, the writer switches to CDF-5 and logs why.
    *   `--nc-format cdf2|cdf5|netcdf4` overrides the choice.
    *   Variables are written one at a time: copy to host, `nc_put_var`, free. Peak host memory is therefore the largest single variable.
    *   The file is written to `<output>.tmp` and renamed into place when complete.
*   **`degas2_aux`.** Writes `wallfile.txt`, `plasmafile.txt` and `sourcefile.txt` into `--aux-dir`, which defaults to the output file's directory. The formats are the same as today.
*   **Validation of edited arrays.** Before writing a `Degas2Geometry` that Python modified, the writer checks:
    1.  every array's shape matches its schema dimensions;
    2.  dimension sizes agree across variables that share them;
    3.  index arrays marked in the schema (e.g. `cells[:,0]` pointing into `boundaries`, `neighbors` pointing into cells, `sectors` pointing into strata) stay in range.

    On failure it raises `ValueError` naming the variable, the index and the value. This answers part of the open question on edited arrays. Deeper physics consistency is not checked.

### 7. Runtime ownership

*   **Native executables.** `main` constructs `Omega_h::Library lib(&argc, &argv)` first. That gives Kokkos flags (`--kokkos-num-threads`, `--kokkos-device-id`) and omega-h flags (`--osh-time`). If the MPI communicator has more than one rank, the program prints a message and exits.
*   **Python `_core`.** A process-wide `std::shared_ptr<Omega_h::Library>` is created on the first call that needs it.
    *   If PyOmega_h was imported earlier, the Library sees that Kokkos and MPI are already running and takes ownership of neither.
    *   `Py_AtExit` releases the Library after Python objects are gone, so no device array outlives `Kokkos::finalize`. This is verified in `Omega_h_library.cpp:150-160`, `:189-218`.
    *   `omegah2csg.initialize(kokkos_args=[...])` lets a script pass Kokkos options before first use. If called after the runtime is already up, it raises an error.
*   **Shutdown safety.** `_core` holds no static Kokkos or Omega_h data. Result objects release their device arrays when garbage-collected.
*   **GIL.** Pipeline calls release the GIL only after all adjacency is built, because `ask_up` modifies the mesh cache.

### 8. Python layer

*   **`omegah2csg._core` (pybind11) binds:**
    *   `OpenMCOptions` and `Degas2Options`, holding every default;
    *   `CsgGeometry` and `Degas2Geometry`;
    *   `compute_csg(source, options)` and `compute_degas2(source, options)`, where `source` is a path or a PyOmega_h mesh;
    *   `write_openmc_xml(geom, path, ...)` and `write_degas2(geom, path, ...)`;
    *   `show_config()`.
*   **How arrays reach Python.**
    *   **First access.** Each result-object array property copies device → host once, into a NumPy array with the documented shape (e.g. `(nedges, 6)`), and caches it.
    *   **Editing.** The cached array is writable.
    *   **Writing.** Writers use the cached host array if it exists, otherwise they copy from the device. The device copy is never updated from Python edits. After results are returned, only writers and the Python builder read the data, so this is safe.
*   **No mesh reader of our own.** Users who want an in-memory mesh read it with PyOmega_h's own functions, for example `read_mesh_binary`, and pass the mesh in. Everyone else passes a path.
*   **Pure-Python package:**
    *   `omegah2csg.openmc_builder`;
    *   console-script entry points that run the bundled native executable;
    *   `__init__` exporting the public API.

    Nothing else lives in the Python package.

### 9. PyOmega_h interop

*   **Optional `Omega_h::Mesh&` overloads.** `compute_*` also accepts `Omega_h::Mesh&`. `_core` never registers `Omega_h::Mesh` itself. pybind11 matches the overload only when PyOmega_h's `OmegaHMesh` type is registered with the same pybind11 internals.
*   **Same-library guard.** Before using a passed-in mesh, `_core` checks with `dladdr` that the mesh's library code comes from the same `libomega_h` file as `_core`'s own. If not, it raises a `TypeError` naming both paths and suggesting a file path.
*   **pybind11 version pin.**
    *   It is pinned to the spack omega-h's pybind11 version.
    *   At configure time, CMake reads the `__pybind11_internals_v*` string from a found PyOmega_h `.so` and warns if it differs from ours.
*   **Wheel stages.**
    *   **Stage A (now):** wheels bundle libomega_h, Kokkos (OpenMP) and netcdf-c (no HDF5). They take file paths only.
    *   **Stage B (once omega-h publishes PyOmega_h with shared libraries, headers and CMake config):** the wheel depends on it, links to its libraries, and excludes them from auditwheel repair. The process then has one libomega_h and one Kokkos, and mesh interop works from pip.

### 10. Front ends

*   **Executables and console scripts.** Two native executables and two Python console scripts. Names are not chosen yet.
*   **One source of flags.** Both front ends take their flags and defaults from `OpenMCOptions` / `Degas2Options`. The console scripts run the bundled native executable with `os.execv`, so there is one CLI implementation. A test compares the `--help` option lists of both front ends.
*   **Flags:**

    | Flag | Purpose |
    |---|---|
    | `-o/--output` | Output file (`geometry.xml` or `geometry.nc`) |
    | `-a/--aux-files` | DEGAS2: also write `wallfile.txt`, `plasmafile.txt` and `sourcefile.txt`. Renamed from `--auxfiles` |
    | `--aux-dir DIR` | DEGAS2: directory for the aux files. Default: the directory of `-o` |
    | `-v/--verbose` | More progress output; repeatable (`-vv`) |
    | `-q/--quiet` | Errors only |
    | `--tol` | Single default, documented as an absolute length in mesh units |
    | `--force` | Overwrite existing output |
    | `--check` | Validate tags and preconditions, then exit |
    | `--timing` | Print per-stage wall time |
    | `--debug N` | Dump the first N entities of each stage |
    | `--nc-format` | DEGAS2 netCDF format |
    | `--target-temperature`, `--target-recyc-coef` | DEGAS2 target settings |
    | `--material-ids FILE` | Per-cell material IDs (OpenMC) |
    | `--version` | Version. With `-v`, also the build configuration. `--version` has no short form, because `-v` now means verbose; today's `convert2degas2 -v` changes meaning, and the migration guide records this |
    | Kokkos / omega-h flags | Passed through |

### 11. Build and packaging

*   **CMake options.** `OMEGAH2CSG_ENABLE_PYTHON`, `OMEGAH2CSG_ENABLE_OPENMC_IO` (pugixml), `OMEGAH2CSG_ENABLE_DEGAS2_IO` (netcdf-c), `OMEGAH2CSG_ENABLE_APPS`, `OMEGAH2CSG_ENABLE_TESTS`.
*   **Dependencies.**
    *   pugixml: `find_package`, falling back to `FetchContent` at a pinned tag.
    *   netcdf-c: `find_package(netCDF)`, falling back to pkg-config.
*   **CUDA builds.** Kokkos-compiled targets (`topology` through `pipeline`, apps, `_core`) get `kokkos_compilation`. `_core` uses `NO_EXTRAS`, as omega-h does. The writers compile with the host compiler.
*   **Exports.** Installed targets are exported as `omegah2csg::core` with optional `COMPONENTS openmc_io degas2_io`.
*   **Wheel (scikit-build-core + cibuildwheel).**
    *   `before-all` builds Kokkos (Serial+OpenMP), omega-h without pybind11, and netcdf-c without HDF5/DAP/NCZarr, once.
    *   The wheel contains `_core`, the executables in `omegah2csg/bin/`, and the libraries bundled by auditwheel.
    *   A wheel test runs each executable's `--version` and a small conversion in a clean virtualenv.

### 12. Testing architecture

| Level | What | Where it runs |
|---|---|---|
| Unit (C++) | L0 math on hand-written points; L2 stages on hand-built `Topology` | Catch2 via CTest, every PR |
| Golden (C++) | Pipeline output vs frozen goldens: every netCDF variable, aux files, `.npz` arrays, `geometry.xml` by meaning | CTest, Serial/OpenMP at 1 and 4 threads |
| Python | Result-object properties, edit-then-write, validation errors, openmc builder, C++/Python XML equivalence, PyOmega_h interop (when available) | pytest, every PR |
| CLI | Both front ends: `--help` parity, exit codes, `--check` | pytest, every PR |
| CUDA | Compile-only build in an `nvidia/cuda` container | Hosted CI, every PR |
| GPU runtime + benchmarks | Golden suite on the GPU; copy counter; per-stage time and memory up to 1e7 triangles; 15% regression gate | Self-hosted SCOREC runner: `main`, nightly, labelled PRs |
| DEGAS2 acceptance | `defineback` / `flighttest` on the converted test case | Self-hosted runner with the spack DEGAS2 |

**Goldens.**
*   Captured from a frozen baseline commit after the bug fixes, with per-variable tolerances.
*   `tests/golden/README.md` records the commit, tool versions and the command to regenerate them.
*   Synthetic benchmark meshes are generated by a script from PyOmega_h `build_box` with tags. Real meshes are stored outside git with checksums.

### 13. Migration order

Each phase is one mergeable pull request (or a small group), with exit criteria.

1.  **P0 – Safety net.**
    *   Commit the pending build fix.
    *   CI on `main`, with branch protection.
    *   Pin omega-h and Kokkos versions for CI.
    *   Create a `release/0.0.x` branch.
2.  **P1 – Characterize.** Generate full-output goldens from the current code, and add the comparators.
3.  **P2 – Fix, then freeze.**
    *   Fix each known bug in the current Python/C++ code in its own PR, with the golden diff reviewed.
    *   Freeze the goldens.
4.  **P3 – Core stages.** L0–L2 plus `pipeline::compute_csg`, with unit tests. The old ctypes path stays as a reference, and its arrays are compared with the new ones.
5.  **P4 – OpenMC.** `openmc_xml`, the Python builder and the equivalence test. A hidden `OMEGAH2CSG_BACKEND=python|cpp` switch defaults to `python` for one release.
6.  **P5 – DEGAS2.** Assembler, schema, netCDF and aux writers, and validation. Same switch.
7.  **P6 – Front ends and Python.** Executables, `_core`, console scripts, PyOmega_h interop. Remove ctypes, `capi.*`, `config.py` and `OmegaHMesh.py`. The Python converters move to `tests/reference/`.
8.  **P7 – Wheels Stage A.** Then benchmarks and the regression gate on the self-hosted runner.
9.  **P8 – Release 0.1.0.** Flip the backend defaults to `cpp`. Publish `design_choices.rst`, `references.rst` and the migration table.

### 14. Main risks

| Risk | Mitigation |
|---|---|
| pugixml memory or time at 1e7 cells | Measured in the benchmark; replaceable behind the `openmc_xml` interface |
| `ZConeOneSided` expansion drifts with OpenMC versions | Golden pinned to an OpenMC version; the equivalence test runs against the installed OpenMC |
| No GPU in hosted CI | CUDA compile-only job, plus the self-hosted runner; release checklist requires a GPU run log |
| Results differ slightly between OpenMP and CUDA | Per-variable tolerances; a test that no golden edge sits within 100·eps of `tol` |
| Stage B depends on the PyOmega_h wheel's contents | Stage A works without it; Stage B is optional |
| pybind11 drift between PyOmega_h and `_core` | Configure-time warning, the call-time `TypeError` with a message, and the same-library guard |

## Alternatives Considered

### Alternative A: Compute cell quantities in NumPy on top of PyOmega_h (rev-1 plan)

*   **Pros**:
    *   Less C++ code.
    *   Uses omega-h's Python bindings directly.
    *   NumPy formulas are easy to read.
*   **Cons**:
    *   PyOmega_h converts arrays with a device→host copy followed by a single-threaded copy loop, one element at a time (`PyOmega_h_numpy_transform.hpp:36-47`).
    *   NumPy indexing such as `coords[elem_verts]` creates large temporary arrays.
    *   It would move kernels that already run on the device (`capi.cpp:298-436`) back to the host.
*   **Reason for Rejection**: The project exists for speed on multi-million-triangle meshes. This alternative makes the slowest part slower and breaks G1.

### Alternative B: Keep the converters in Python and only speed up hot loops

*   **Pros**:
    *   Smaller change.
    *   The netCDF writing stays in the familiar `netCDF4` package.
*   **Cons**:
    *   Each converter still runs many Python passes over the data.
    *   Each array still crosses device→host separately.
    *   The converters stay tied to Python, so a native executable or a GPU spack build cannot use them.
*   **Reason for Rejection**: It contradicts the owner's direction to move everything into C++ with a thin Python layer, and it leaves G1 and G8 unmet.

### Alternative C: Keep ctypes with `extern "C"` functions (current design)

*   **Pros**:
    *   No compiled-binding dependency.
    *   Already works.
*   **Cons**:
    *   C++ exceptions crossing the C boundary are undefined behaviour.
    *   Every function needs a hand-written prototype and size checks.
    *   Arrays are marshalled by hand.
    *   It cannot exchange objects with PyOmega_h.
*   **Reason for Rejection**: It cannot meet G7 (errors become exceptions) or support PyOmega_h interop.

### Alternative D: nanobind instead of pybind11

*   **Pros**:
    *   Smaller binaries.
    *   Faster compiles and calls.
    *   Modern API.
*   **Cons**: nanobind types cannot be shared with pybind11 types, and omega-h's bindings use pybind11.
*   **Reason for Rejection**: It would make PyOmega_h mesh interop impossible.

### Alternative E: Vendor PyOmega_h in our wheels, or build PyOmega_h's sources in our CMake

*   **Pros**: Mesh-object interop from pip immediately.
*   **Cons**:
    *   It needs an omega-h build for each Python version.
    *   A top-level `PyOmega_h` module in our wheel shadows, or is shadowed by, a user's own copy.
    *   Two libomega_h and Kokkos copies can end up in one process.
    *   Building PyOmega_h's sources here couples us to omega-h's internal source list.
*   **Reason for Rejection**: Rejected by the owner. omega-h's own CI/CD will publish PyOmega_h to PyPI, and Stage B links against that instead.

### Alternative F: Depend on the PyOmega_h wheel but still bundle our own libomega_h

*   **Pros**: Simple to build, with no need to locate PyOmega_h's libraries at build time.
*   **Cons**: Two libomega_h and Kokkos copies would be loaded in one process. A mesh created by one copy would be processed by code from the other, and only one Kokkos would be initialized.
*   **Reason for Rejection**: It causes crashes or silently wrong results. Stage B must exclude omega-h and Kokkos from our wheel.

### Alternative G: Require a PyOmega_h mesh object as the only input

*   **Pros**: One input type, and PyOmega_h always owns the runtime.
*   **Cons**: Every user would need matching pybind11 builds and the same libomega_h. Command-line and plain-file workflows gain nothing.
*   **Reason for Rejection**: File paths work everywhere. Mesh objects are kept as an optional, guarded overload.

### Alternative H: Our own Kokkos/MPI start and stop code, or one Library per mesh

*   **Pros**: Full control over start-up options.
*   **Cons**:
    *   It duplicates logic omega-h already has.
    *   It risks double initialization or double finalization next to PyOmega_h.
    *   One Library per mesh is known to cause MPI-finalize errors, according to omega-h's own pytest notes.
*   **Reason for Rejection**: `Omega_h::Library` already starts only what is not running and finalizes only what it started (`Omega_h_library.cpp:150-160`, `:189-218`).

### Alternative I: Typed Kokkos 2D views, or one named 1D array per column, for internal data

*   **Typed `Kokkos::View<T*[N], LayoutRight, Omega_h::Device>`**:
    *   **Pros**: The column count is part of the type, sizes are `size_t`, and the layout is explicit.
    *   **Cons**:
        *   It is easy to declare a view without `Omega_h::Device`, which is the bug this design removes.
        *   It needs conversion to use Omega_h helpers.
        *   It does not use omega-h's memory pool.
*   **One named array per column**:
    *   **Pros**: The names are explicit, and each column maps one-to-one to a NumPy array.
    *   **Cons**:
        *   The face-side kernel would read six separate locations per randomly accessed edge instead of one contiguous row.
        *   Many more arrays to manage.
*   **Reason for Rejection**: The owner chose flat `Omega_h::Write/Read` with documented component counts and named column constants. These arrays always live in Omega_h's space, use its pool, and work with its helpers. The `LO` (int32) size limit is far above multi-million-triangle meshes, and the host checks it.

### Alternative J: Hand-rolled `geometry.xml` writers (parallel two-pass, or serial streaming)

*   **Pros**:
    *   No XML dependency.
    *   Memory close to the file size.
    *   The parallel version uses all cores.
*   **Cons**:
    *   Escaping and schema details are all our responsibility.
    *   The parallel version has two passes that must agree exactly.
    *   The serial version may become the slowest stage.
*   **Reason for Rejection**: The owner chose pugixml, which handles escaping and builds the XML structure for us. Both hand-rolled writers stay as the fallback behind the same interface if the benchmark memory or time gate fails.

### Alternative K: OpenMC output only as a C++ `geometry.xml`, or only as Python objects

*   **Pros**: One code path to maintain.
*   **Cons**:
    *   XML-only output removes Python access to cells and arrays, which the owner requires (G5).
    *   Python-only output costs time per object and takes minutes at multi-million cells.
*   **Reason for Rejection**: Both paths are kept, fed from one `CsgGeometry`, and a test keeps them equal (G6).

### Alternative L: Console scripts that call `_core` directly instead of running the native executable

*   **Pros**: No separate process image. Python-level exceptions are visible to callers who import the CLI module.
*   **Cons**:
    *   There would be two CLI implementations, in C++ and in Python, that must stay identical.
    *   Python start-up and import cost on every run.
*   **Reason for Rejection**: Running the bundled executable gives one CLI implementation. Python users who want exceptions and objects call the Python API, not the CLI.

### Alternative M: Push Python edits back to the device

*   **Pros**: C++ stages could run again on edited arrays.
*   **Cons**:
    *   Extra host→device copies.
    *   It must track which copy (host or device) is current.
    *   Nothing in the current workflows re-runs device stages after results are returned.
*   **Reason for Rejection**: Not needed for G5. Writers and the Python builder read the edited host copies. If a future feature needs device re-computation on edited data, this can be added later.

### Alternative N: netCDF-4/HDF5 output by default, or the netcdf-cxx4 library

*   **Pros**: It matches today's `NETCDF4_CLASSIC` output. netcdf-cxx4 offers a C++ API.
*   **Cons**:
    *   DEGAS2 does not need HDF5: its own geometry writer uses classic format (`geomsetup.web:1266`).
    *   HDF5 adds a large dependency to the wheel, and a second HDF5 copy can crash next to `h5py`, which `openmc` imports.
    *   netcdf-cxx4 is a thin, rarely updated wrapper.
*   **Reason for Rejection**: CDF-2 via netcdf-c meets G10 with fewer dependencies. netCDF-4 output stays available through `--nc-format`.

### Alternative O: Add the missing bindings upstream in omega-h (`ask_up`, zero-copy arrays)

*   **Pros**: More could be done in Python on top of PyOmega_h.
*   **Cons**: It requires omega-h changes, and the computation would still need to run in C++ for speed.
*   **Reason for Rejection**: Out of scope, since omega-h is not changed. This project adapts if omega-h adds such bindings on its own.

--------------------------------------------------------------------------------

## Non-Functional Requirements

*   **Performance**
    *   Run time scales linearly with mesh size (G2), with no regression above 15% (G3).
    *   Device↔host copies happen only at output (G1).
    *   Peak host memory during DEGAS2 output stays at about the size of the largest single variable, because variables are written one at a time.
    *   Peak memory of the pugixml stage is measured and capped by the fallback rule in Technical Plan §5.
*   **Reliability**
    *   No process aborts on bad input (G7).
    *   Output files are written to a temporary file and then renamed, so a failed run never leaves a partial file under the real name.
*   **Portability**
    *   Linux x86_64. OpenMP wheels on PyPI; CUDA through source or spack builds.
    *   Python 3.11–3.13, C++17, CMake ≥ 3.24, Kokkos 4.x, omega-h pinned by commit.
*   **Rollout and rollback**
    *   For one release, a hidden `OMEGAH2CSG_BACKEND=python|cpp` setting chooses per converter between the old Python converters and the new C++ path. It defaults to `python`.
    *   The default flips to `cpp` at 0.1.0.
    *   Rollback is a patch release that flips the default back.
    *   The `release/0.0.x` branch receives fixes for users who stay on the old API.
*   **Security and privacy**: none. The tool reads local mesh files and writes local outputs.

### Additional Resources

*   Predecessor brief: [cpp-pipeline-modernization.md](./cpp-pipeline-modernization.md).
*   omega-h sources consulted (commit `7647c8a`):
    *   `src/Omega_h_library.cpp` (runtime ownership)
    *   `src/Omega_h_kokkos.hpp` (memory and execution spaces)
    *   `src/Omega_h_file.cpp` (binary read, `nparts`)
    *   `src/PyOmega_h_*.cpp` and `src/PyOmega_h_numpy_transform.hpp` (bindings, NumPy conversion)
*   DEGAS2 sources consulted (`~/degas2/src`): `geometry.web:1389` (reader), `geomsetup.web:1266` (classic writer), `definegeometry2d.web:1555`/`:2143` (64-bit offset), `netcdf.hweb` (types, dimension checks).
*   External references are collected in `docs/references.rst`: pybind11 cross-module types and ABI, scikit-build-core, cibuildwheel and auditwheel, Kokkos initialize/finalize, netCDF formats, OpenMC geometry XML, pugixml.

### Future Improvements

*   **Partitioned meshes under `mpirun`**: read with one rank per part, gather to rank 0, then run the serial pipeline.
*   **Stage B wheels**: link against omega-h's PyOmega_h wheel, once it ships shared libraries, headers and CMake config.
*   **Re-running device stages on Python-edited arrays**: Alternative M, if a workflow needs it.
*   **Zero-copy host access on OpenMP builds**: an array whose memory is already on the host could be exposed to NumPy without a copy, with lifetime tied to the result object.
*   **Sphinx site**: built from `docs/` with stubs and API pages.

--------------------------------------------------------------------------------

## Detailed Implementation Plan

### Conventions used below

*   **Phase tags.** `[P0]`–`[P8]` mark the phase in Technical Plan §13 where a file is created or changed.
*   **Layout.**
    *   **C++ library:** stays under `src/` (no meaning change for `git log`). Headers and sources are in `src/omegah2csg/`, so includes read `<omegah2csg/…>`.
    *   **Python package:** moves to `python/omegah2csg/`, a layout scikit-build-core detects automatically.
    *   **Bindings and executables:** in `bindings/` and `apps/`.
*   **Executable names.** They keep today's CLI names, `convert2openmc` and `convert2degas2`, set through the CMake cache variables `OMEGAH2CSG_OPENMC_EXE_NAME` and `OMEGAH2CSG_DEGAS2_EXE_NAME`. Renaming later changes only these two variables and the `[project.scripts]` table.
*   **Single tolerance default.** `tol = 1e-10`, absolute, in mesh length units. This is today's CLI default, so the CLI goldens stay valid. The Python default of `1e-6` and its failing `assert tol < 1e-6` go away.
*   **Pseudocode only.** Signatures below are pseudocode, not final code.

### Target Files to Create/Modify

#### Repository root and tooling

##### 1. `CMakeLists.txt` (Modify, rewrite) [P0, P3, P6]

*   **Type / Scope**: Build configuration.
*   **Proposed Changes**:
    *   `cmake_minimum_required(VERSION 3.24)`; `project(omegah2csg VERSION 0.1.0 LANGUAGES CXX)`. The version is still parsed by scikit-build-core's regex.
    *   Options: `OMEGAH2CSG_ENABLE_PYTHON` (default ON if `SKBUILD`), `OMEGAH2CSG_ENABLE_APPS` (ON), `OMEGAH2CSG_ENABLE_OPENMC_IO` (ON), `OMEGAH2CSG_ENABLE_DEGAS2_IO` (ON), `OMEGAH2CSG_ENABLE_TESTS` (OFF), `OMEGAH2CSG_ENABLE_BENCH` (OFF).
    *   Cache variables for the executable names.
    *   `find_package(Omega_h CONFIG REQUIRED)`; fail if `NOT Omega_h_USE_Kokkos`.
    *   `include(cmake/omegah2csgDependencies.cmake)`.
    *   `add_subdirectory(src)`, `apps`, `bindings`, `tests`, `bench`, each behind its option.
    *   Removes the post-build copy of the `.so` into `pythonAPI/omegah2csg/lib/` and the forced `BUILD_SHARED_LIBS`.
    *   Keeps the install and export rules behind `if(NOT SKBUILD)`, so wheels carry no headers.
    *   **Why:** the build must express the new layered targets, optional I/O components and front ends.

##### 2. `cmake/omegah2csgDependencies.cmake` (Create) [P4, P5]

*   **Type / Scope**: CMake module.
*   **Proposed Changes**:
    *   **pugixml.** `find_package(pugixml CONFIG)`. If not found, `FetchContent_Declare(pugixml GIT_TAG v1.14)`, built static with PIC.
    *   **netcdf-c.** `find_package(netCDF CONFIG)`, falling back to `pkg_check_modules(NETCDF IMPORTED_TARGET netcdf)`. Either way, alias the result to `omegah2csg::netcdf`.
    *   **pybind11.** `find_package(pybind11 CONFIG REQUIRED)` when Python is enabled.
    *   **PyOmega_h check.** If a PyOmega_h `.so` is found through `Python_EXECUTABLE -c "import PyOmega_h; print(PyOmega_h.__file__)"`, read its `__pybind11_internals_v*` string with `file(STRINGS … REGEX)` and `message(WARNING)` if it differs from our `PYBIND11_INTERNALS_ID`.
    *   **Why:** one place for dependency discovery, with fallbacks for spack builds that ship no CMake config.

##### 3. `cmake/omegah2csgConfig.cmake.in` (Modify) [P6]

*   **Type / Scope**: CMake package config.
*   **Proposed Changes**:
    *   `find_dependency(Omega_h)` always.
    *   `find_dependency(pugixml)` or `find_dependency(netCDF)` only when the `openmc_io` or `degas2_io` component is requested.
    *   Includes the exported targets.
    *   Removes the stale netcdf-cxx comment.
    *   **Why:** C++ consumers can use the core without pulling in I/O dependencies.

##### 4. `pyproject.toml` (Modify) [P0, P6, P7]

*   **Type / Scope**: Python packaging.
*   **Proposed Changes**:
    *   `[build-system] requires = ["scikit-build-core>=0.11", "pybind11==<pinned from ci/constraints.txt>"]`.
    *   `wheel.packages = ["python/omegah2csg"]`; `build-dir = "build/{wheel_tag}"`; `cmake.version = ">=3.24"`.
    *   `[[tool.scikit-build.generate]]` writes `omegah2csg/_version.py`.
    *   Dependencies: `numpy>=1.26`. Optional extras: `openmc = ["openmc>=0.15"]`, `test = ["pytest", "netCDF4", "pandas"]`.
    *   `[project.scripts]` maps `convert2openmc` and `convert2degas2` to `omegah2csg._cli:openmc_main` and `degas2_main`.
    *   `[tool.pytest.ini_options]` replaces `pytest.toml`: `testpaths=["tests/python"]`, `addopts=["-ra","--strict-markers","--strict-config","--import-mode=importlib"]`, `xfail_strict=true`.
    *   Ruff `extend-select = ["B","I","UP","PT","NPY","SIM","RUF"]`.
    *   Removes the `CMAKE_C(XX)_COMPILER` and `*_ROOT` env passthroughs, because CMake reads `CC`, `CXX` and `<Pkg>_ROOT` itself.
    *   **Why:** this matches the new layout, makes `openmc` and `netCDF4` optional, and pins pybind11.

##### 5. `pytest.toml` (Delete) [P0]

*   **Proposed Changes**: Its content moves into `pyproject.toml`.
*   **Why:** one config file.

##### 6. `.gitignore` (Modify) [P0]

*   **Proposed Changes**:
    *   Add `build/`, `tests/**/_out/` and `bench/_results/`.
    *   Keep `.venv`, the pending change.
    *   Stop ignoring `*.xml` globally. Ignore only root-level outputs, so golden XML files can be committed.
*   **Why:** golden `geometry.xml` files must be tracked.

##### 7. `.pre-commit-config.yaml` (Create) [P0]

*   **Proposed Changes**: Hooks: `ruff` (check and format), `clang-format` v18 (mirrors-clang-format), `cmake-format`, `check-yaml`, `end-of-file-fixer`, `trailing-whitespace`, `validate-pyproject`.
*   **Why:** one command reproduces the CI format checks locally.

##### 8. `ci/versions.env` and `ci/constraints.txt` (Create) [P0]

*   **Proposed Changes**:
    *   `versions.env` pins `OMEGA_H_SHA`, `KOKKOS_VERSION`, `NETCDF_C_VERSION`, `PUGIXML_VERSION` and `OPENMC_VERSION` (for goldens).
    *   `constraints.txt` pins `pybind11==X.Y.Z`, matching the spack omega-h, and the test-tool versions.
*   **Why:** one source of truth for CI, wheels and cache keys. Today omega-h `master` is unpinned and the cache never invalidates.

##### 9. `README.md` (Modify) [P0, P8]

*   **Proposed Changes**:
    *   **[P0]** Update links from `doc/` to `docs/`.
    *   **[P8]** Rewrite usage with:
        *   the executables;
        *   the Python API (`compute_csg`, `compute_degas2`, `to_openmc_universe`, `write_*`);
        *   Kokkos environment variables;
        *   the CPU vs GPU build note;
        *   a link to the migration guide.
*   **Why:** the user-facing entry point must match the new API.

##### 10. `CHANGELOG.md` (Create) [P0]

*   **Proposed Changes**: Keep-a-Changelog format. Every bug-fix PR in P2 adds an entry describing the output change.
*   **Why:** golden changes need a human-readable record.

##### 11. `concepts_brief.md`, `design_draft.md` (root) (Delete) [P0]

*   **Proposed Changes**: Remove these working files after the design is saved under `docs/designs/`.
*   **Why:** avoid stale duplicates at the repo root.

#### CI

##### 12. `.github/workflows/ci.yml` (Modify, rewrite) [P0, P3–P6]

*   **Proposed Changes**:
    *   Triggers: `push` and `pull_request` on `main`, plus `release/*`.
    *   **`lint` job:** `pre-commit run --all-files`, plus a grep guard that fails on `Kokkos::View<` or policy-less `Kokkos::parallel_` in `src/omegah2csg/` and `apps/`.
    *   **`build-test` job:**
        *   Matrix over Python 3.11/3.13 × {Release, Debug}. Debug enables `Kokkos_ENABLE_DEBUG_BOUNDS_CHECK`.
        *   Builds Kokkos (Serial+OpenMP), omega-h (pinned SHA) and netcdf-c from `ci/versions.env` through the composite action.
        *   Runs `pip install -e . --no-build-isolation -Ccmake.define.OMEGAH2CSG_ENABLE_TESTS=ON`.
        *   Runs `ctest --output-on-failure` and `pytest` at `OMP_NUM_THREADS=1` and `4`.
    *   **`cuda-compile` job:** container `nvidia/cuda:12.4.1-devel-ubuntu22.04`, Kokkos `CUDA=ON ARCH_AMPERE80 CUDA_LAMBDA=ON`. Builds omega-h and all targets; no tests run.
*   **Why:** G11, and catching host/device mistakes without a GPU.

##### 13. `.github/workflows/format-check.yml` (Delete) [P0]

*   **Proposed Changes**: Replaced by the `lint` job.
*   **Why:** one source of formatting rules.

##### 14. `.github/workflows/gpu.yml` (Create) [P7]

*   **Proposed Changes**:
    *   `runs-on: [self-hosted, gpu]`, triggered on push to `main`, nightly, and PRs labelled `run-gpu`. Never runs for forks.
    *   Builds against the spack environment.
    *   Runs CTest and pytest on CUDA.
    *   Runs the copy-counter check (G1), `bench/run_bench.py` with the 15% gate (G3), and the DEGAS2 acceptance test (G10).
*   **Why:** GPU runtime and performance coverage.

##### 15. `.github/workflows/build-wheels.yml` (Modify) [P7]

*   **Proposed Changes**:
    *   Triggers: tags `v*`, PRs touching `CMakeLists.txt`, `pyproject.toml`, `src/**`, `bindings/**`, `apps/**` or `.github/**`, and manual dispatch.
    *   `CIBW_BEFORE_ALL_LINUX` builds Kokkos (Serial+OpenMP), omega-h (pinned SHA, no pybind11, MPI off), and netcdf-c (`--disable-netcdf-4 --disable-dap --disable-byterange --disable-nczarr`).
    *   `CIBW_TEST_EXTRAS=test`; `CIBW_TEST_COMMAND` runs `pytest {project}/tests/python -m "not needs_openmc and not needs_pyomega_h"` and both executables' `--version`.
    *   Adds an sdist job and PyPI Trusted Publishing on tags.
    *   Adds a check that the repaired wheel contains exactly one `libomega_h*` and one `libkokkoscore*`.
*   **Why:** Stage A wheels with executables, tested in a clean environment.

##### 16. `.github/actions/install-repo/action.yml` (Modify) [P0]

*   **Proposed Changes**:
    *   Cache key `build-<name>-<ref>-<hash of options>-<hash of ci/versions.env>`.
    *   Accepts a commit SHA as `repo-ref`.
*   **Why:** the current key (`build-<name>-<ref>` with ref `master`) never invalidates.

#### Documentation

##### 17. `doc/` → `docs/` (Move: `DEVELOPERS_GUIDE.md`, `top_bottom_flag.md`, `images/compute_coefficients.svg`) [P0]

*   **Proposed Changes**: `git mv` into `docs/`.
*   **Why:** one documentation folder, as the owner decided.

##### 18. `docs/DEVELOPERS_GUIDE.md` (Modify) [P0, P6]

*   **Proposed Changes**: Add:
    *   the layer and dependency rules (Technical Plan §2) and kernel rules (§4);
    *   building against a spack environment (`spack load omega-h kokkos netcdf-c pugixml py-pybind11`, then `pip install -e . --no-build-isolation`);
    *   the test tiers;
    *   how to regenerate goldens;
    *   pre-commit usage.
*   **Why:** contributors need the new rules in one place.

##### 19. `docs/design_choices.rst` (Create) [P0, updated P8]

*   **Proposed Changes**:
    *   Sphinx-ready design decisions in ADR style (Status / Context / Decision / Consequences) for every decision in this design, including:
        *   C++ pipeline;
        *   flat Omega_h arrays;
        *   `Omega_h::Library` ownership;
        *   no vendoring and wheel stages;
        *   pybind11 with optional mesh overloads;
        *   pugixml with a fallback rule;
        *   CDF-2;
        *   schema-driven DEGAS2;
        *   the edit model;
        *   console scripts running the executables;
        *   golden-first migration;
        *   tolerance default.
    *   A glossary: `offset_face`, `isOnWall`, wall, boundary layer, surface sides, DEGAS2 zone, sector, stratum.
*   **Why:** requested by the owner for the future docs site.

##### 20. `docs/references.rst` (Create) [P0]

*   **Proposed Changes**: Annotated external references (title, URL, one line on relevance):
    *   pybind11 (compiling, cross-module types, changelog);
    *   scikit-build-core;
    *   cibuildwheel and auditwheel;
    *   Kokkos initialize/finalize and Kokkos Tools;
    *   netCDF file formats;
    *   OpenMC geometry XML and `ZConeOneSided`;
    *   pugixml;
    *   Catch2;
    *   the Scientific Python Development Guide;
    *   prior art (pyAMReX, DOLFINx, OpenMC `openmc.lib`).

    Unverified claims are marked as such.
*   **Why:** requested by the owner.

##### 21. `docs/migration-0.1.md` (Create) [P6]

*   **Proposed Changes**: A table mapping every public 0.0.x name and `python -m` entry point to its 0.1 replacement or removal. For example, `OmegaHMesh` → `compute_csg(path)`, `get_all_geometry_info` → `CsgGeometry` fields, and `kokkos_runtime` → removed.
    *   It also records output changes from the P2 bug fixes.
    *   It also records CLI changes:
        *   `-v` now means `--verbose` (it used to mean `--version` in `convert2degas2`);
        *   `--auxfiles` → `-a/--aux-files`;
        *   the new `--aux-dir`;
        *   the Python API's default `tol` changes from 1e-6 to 1e-10.
*   **Why:** clean API break with a documented path.

##### 22. `docs/designs/cpp-pipeline-modernization-design.md` (Create at finalization)

*   **Proposed Changes**: The finalized version of this design.
*   **Why:** EGM finalization target. The brief in the same folder stays as history.

#### C++ library (`src/`)

##### 23. `src/CMakeLists.txt` (Create) [P3–P5]

*   **Proposed Changes**: Targets:

    | Target | Kind | Contents | Links |
    |---|---|---|---|
    | `omegah2csg_geom` | INTERFACE | — | `Omega_h::omega_h` |
    | `omegah2csg_core` | STATIC, PIC | `mesh_input`, `topology`, `csg`, `wall`, `cell_quantities`, `errors`, `transfer`, `runtime_info`, `profiling` | — |
    | `omegah2csg_degas2` | — | `degas2_schema`, `degas2_geometry`, `degas2_assemble` | — |
    | `omegah2csg_openmc_io` | host compiler | `io/openmc_xml` | pugixml |
    | `omegah2csg_degas2_io` | host compiler | `io/degas2_netcdf`, `io/netcdf_file`, `io/degas2_aux`, `io/atomic_file` | netcdf |
    | `omegah2csg_pipeline` | — | `pipeline`, `options` | — |

    *   `kokkos_compilation(TARGET …)` on core, degas2 and pipeline.
    *   Alias targets `omegah2csg::core` and so on.
*   **Why:** one target per layer enforces the dependency rules and allows isolated tests.

##### 24. `src/omegah2csg/types.hpp` (Create) [P3]

*   **Proposed Changes**:
    *   `enum class SurfaceType : I8 { ZPlane = 1, ZCylinder = 2, ZCone = 3 }`. The values match today's `edge_types`.
    *   Column constants for `surface_coeffs`: `kCoeffA=0 … kCoeffSlope=5`, `kNumCoeffs=6`, documented with the equation each column belongs to.
    *   `kFaceEdges=3`, `kBBoxComps=4`, `kCentroidComps=2`.
    *   Index helpers `at(i, comp, ncomp)`.
*   **Why:** names replace positional magic numbers. Today the code and tests disagree on columns.

##### 25. `src/omegah2csg/errors.hpp`, `errors.cpp` (Create) [P3]

*   **Proposed Changes**:
    *   **Exceptions.** `class InputError : std::invalid_argument` for bad paths, missing tags or `nparts > 1`. `class GeometryError : std::runtime_error`, carrying the stage name, entity dimension, entity ID and reason code.
    *   **`DeviceErrors` helper.**
        *   Holds `Write<LO> count(1, 0)`, `Write<LO> first(1, LO_MAX)` and `Write<I8> reason(1, 0)`.
        *   A device-callable `report(entity, reason)` does an atomic increment and an atomic min.
        *   A host `throw_if_any(stage, dim, reason_text_table)` fences, reads three values and throws.
    *   **Why:** G7. Kernels never abort, and errors name the entity.

##### 26. `src/omegah2csg/geom.hpp` (Create, header-only, L0) [P3]

*   **Proposed Changes**: All `OMEGA_H_INLINE`, taking `Omega_h::Vector<2>`, moved and renamed from `compute_surface.h` (the pending build-fix versions):
    *   `classify_edge(p0, p1, tol) -> SurfaceType` (vertical → ZCylinder, horizontal → ZPlane, else ZCone; the order is documented once);
    *   `surface_coeffs(p0, p1, type, out Vector<6>)`;
    *   `evaluate_surface(point, coeffs)`;
    *   `side_of_line(point, slope, intercept, tol) -> I8`;
    *   `face_side(v0, v1, v2, coeffs, type, tol) -> I8`, returning 0 on degenerate input; the caller reports it through `DeviceErrors`;
    *   `triangle_area`, `triangle_centroid`, `toroidal_volume(area, centroid_r)` (2π·r̄·A);
    *   `triangle_bbox`.
*   **Why:** pure math, testable on the host. One classification replaces the two disagreeing copies (Python checks horizontal first, C++ checks vertical first).

##### 27. `src/omegah2csg/mesh_input.hpp`, `mesh_input.cpp` (Create, L1) [P3]

*   **Proposed Changes**:
    *   `void check_mesh_path(path)`: the directory exists, the `nparts` file reads `1`, and the `version` is ≤ the tested maximum. Otherwise it throws `InputError` before omega-h is called.
    *   `Omega_h::Mesh read_mesh(Omega_h::Library&, path)`: calls `binary::read(path, lib.self(), &mesh)` with the self communicator.
    *   `void check_required_tags(const Mesh&, Requirements)`: `offset_face` (dim 2, I32) and `isOnWall` (dim 0, I32) for DEGAS2 and wall features, with exact messages.
    *   `void check_single_rank(Library&)`.
*   **Why:** omega-h's own `Omega_h_fail` aborts on these conditions.

##### 28. `src/omegah2csg/topology.hpp`, `topology.cpp` (Create, L1) [P3]

*   **Proposed Changes**:
    *   `struct Topology { LO nverts, nedges, nfaces; Reals coords; LOs edge_verts, face_verts, face_edges, edge_faces_offsets, edge_faces; LOs offset_face, is_on_wall; bool has_wall_tags; }`.
    *   `Topology build_topology(Omega_h::Mesh&)`: calls `ask_down` and `ask_up` once each, and copies tag arrays by reference. Omega_h arrays are reference-counted, so there are no data copies.
    *   Takes the mesh by reference only.
*   **Why:** builds adjacency once and decouples stages from `Mesh`.

##### 29. `src/omegah2csg/csg_geometry.hpp` (Create) [P3]

*   **Proposed Changes**:
    *   `struct CsgGeometry` with fields as in Technical Plan §3: `surface_type`, `surface_coeffs`, `face_edges`, `face_sides`, `boundary_edges`, `boundary_sides`, `cell_volume`, `cell_centroid`, `cell_bbox`, `wall_edges`, `wall_faces`, `edge_faces_sorted`, `face_region_tag` (a reference to `offset_face` when present; used by the Python builder's tag-to-material map), plus counts.
    *   All fields are `Read<T>`.
*   **Why:** one result type feeds both OpenMC paths and the DEGAS2 assembler.

##### 30. `src/omegah2csg/csg.hpp`, `csg.cpp` (Create, L2) [P3]

*   **Proposed Changes**:
    *   `compute_surfaces(const Topology&, tol, DeviceErrors&) -> {surface_type, surface_coeffs}`: `parallel_for` over edges, using `geom`.
    *   `compute_face_sides(const Topology&, surface_type, surface_coeffs, tol, DeviceErrors&) -> face_sides`: `parallel_for` over faces, reading each edge's 6 coefficients from one contiguous row.
    *   `compute_boundary(const Topology&, face_sides) -> {boundary_edges, boundary_sides}`. Boundary edges come from edges with one adjacent face, marked and then compacted with Omega_h helpers. The side is looked up in that face's row.
*   **Why:** replaces `compute_edge_coefficients`, `calculate_face_connectivity`, `get_boundary_edge_ids` and `insert_inward_sign_with_boundary_edges`. It runs in the right spaces, with no by-value `Mesh` and no debug printing.

##### 31. `src/omegah2csg/wall.hpp`, `wall.cpp` (Create, L2) [P3]

*   **Proposed Changes**:
    *   `compute_wall(const Topology&, face_sides, DeviceErrors&) -> {wall_edges, wall_faces (2 per wall edge: plasma side, wall side)}`. Wall edges are edges whose two faces have different `offset_face`, marked then compacted in ascending order.
    *   Checks with one reduction that the number of wall edges equals the number of `isOnWall` vertices.
    *   `sort_edge_faces(const Topology&, face_edges, face_sides) -> LOs[2·nedges]`: for each edge, the positive-side face first, then the negative-side face, with −1 for a missing side. This replaces `sort_edge_to_face_map` in Python.
    *   `count_interior_edges(const Topology&)`, used by tests.
*   **Why:** removes the double computation and moves the Python edge loop to the device.

##### 32. `src/omegah2csg/cell_quantities.hpp`, `cell_quantities.cpp` (Create, L2) [P3]

*   **Proposed Changes**:
    *   `compute_cell_quantities(const Topology&) -> {cell_volume, cell_centroid, cell_bbox}`: one fused `parallel_for` over faces.
    *   `universal_bounds(cell_bbox) -> {min, max}` and `total_volume(cell_volume)`, using reductions with `Omega_h::policy`.
    *   Bounding-box seeds use ±infinity, replacing the `INT64_MAX` bug.
*   **Why:** these were `capi.cpp` kernels; they now run in the right spaces and in one pass.

##### 33. `src/omegah2csg/transfer.hpp`, `transfer.cpp` (Create) [P3]

*   **Proposed Changes**:
    *   `template<T> void copy_to_host(Read<T> src, T* dst, size_t n)`: wraps `dst` in an unmanaged `Kokkos::View<T*, HostSpace>`, then calls `Kokkos::deep_copy` from `src.view()`, then fences.
    *   `template<T> HostArray<T> to_host(Read<T>, ncomp)`.
    *   This is the only place outside `bindings/` allowed to copy device→host.
*   **Why:** one copy point, which makes G1 countable.

##### 34. `src/omegah2csg/host_array.hpp` (Create, no Kokkos) [P3]

*   **Proposed Changes**:
    *   `struct HostArrayView<T> { const T* data; size_t n; int ncomp; }`.
    *   `struct HostArray<T>` (owning, `std::vector`-backed).
    *   `enum class DType { I8, I32, F64, Char }`.
*   **Why:** the writer interface type, so writers need no Omega_h or Kokkos headers.

##### 35. `src/omegah2csg/degas2_schema.hpp`, `degas2_schema.cpp` (Create) [P5]

*   **Proposed Changes**:
    *   `struct DimRule { name; kind (fixed N | from count: ncells+1, nedges, nboundaries, nneighbors+1, nzones+1, nsectors+1, nwallpoints+1, 3·nwallpoints, …) }`.
    *   `struct VarSpec { name; DType; dims[]; index_into (optional: target dimension name and base 0/1, for range checks) }`.
    *   One `constexpr` table with every variable and dimension currently created in `convert2degas2.py:156-460`, with the same names, order and types.
    *   `validate(const Degas2Geometry&) -> void` (throws `ValueError`-mapped `InputError` with the variable, index and value):
        *   each array's shape matches its schema dimensions;
        *   dimensions shared between variables agree;
        *   index arrays stay in range.
*   **Why:** one definition drives the writer, the Python properties and validation.

##### 36. `src/omegah2csg/degas2_geometry.hpp`, `degas2_geometry.cpp` (Create) [P5]

*   **Proposed Changes**:
    *   `struct Degas2Variable { const VarSpec* spec; std::variant<Read<I32>, Read<Real>, HostArray<I32>, HostArray<double>, HostArray<char>> data; std::vector<size_t> shape; }`.
    *   `struct Degas2Geometry { std::map<std::string, Degas2Variable> vars; Degas2Counts counts; }`.
    *   `host_view(name)` returns the host array, copying from the device on first use.
*   **Why:** device arrays for the large variables, host arrays for constants and for Python edits.

##### 37. `src/omegah2csg/degas2_assemble.hpp`, `degas2_assemble.cpp` (Create, L3) [P5]

*   **Proposed Changes**: `Degas2Geometry assemble_degas2(const Topology&, const CsgGeometry&, const Degas2Options&)`, porting `convert2degas2.py:467-912`:

    | Variable(s) | Method |
    |---|---|
    | `cells` | `parallel_for` over faces. Zone IDs: plasma faces 1..Nplasma; wall-adjacent outside faces get Nplasma+i by scatter, after a uniqueness check; the rest get nzones+1 |
    | `boundaries` | `parallel_for` over faces, plus wall boundaries from `boundary_edges` and `boundary_sides`, with the 1-based signed encoding |
    | `surfaces` | `parallel_for` over edges, with `DeviceErrors` for an edge with no faces |
    | `neighbors` | From `edge_faces_sorted`, 1-based |
    | `surface_coeffs` | 11 columns mapped from the 6 named columns, as in today's code |
    | `surface_points` | Copied from coordinates |
    | Zone arrays | Gathers over plasma faces and wall faces. The last zone volume is a direct reduction over its cells |
    | `surface_sectors` | Mark wall edges, exclusive scan, fill |
    | `sectors`, `sector_surface`, `sector_points`, `sector_zone`, `sector_type_pointer`, `strata`, `diagnostic_sector_tab` | `parallel_for` over wall edges |
    | Constants and option-derived values | Host-filled `HostArray`s: detector placeholders, diagnostic groups, target temperature (K→J), recycling coefficient, `target_material = 6`, `INT_UNUSED` / `DBL_UNUSED` sentinels |

    *   Checks with reductions that plasma faces are numbered first and the last face is a boundary face. Otherwise it throws `GeometryError`.
*   **Why:** moves the 915-line Python converter into device kernels.

##### 38. `src/omegah2csg/options.hpp`, `options.cpp` (Create) [P4, P5]

*   **Proposed Changes**:
    *   `struct CommonOptions { double tol = 1e-10; int verbosity = 1; /* 0 = quiet, 1 = default, 2+ = verbose */ bool timing = false; int debug_entities = 0; bool force = false; }`.
    *   `struct OpenMCOptions : CommonOptions { std::string output = "geometry.xml"; std::string material_ids_file; std::string boundary_type = "reflective"; int cell_id_offset = 0; int surface_id_offset = 0; }`.
    *   `struct Degas2Options : CommonOptions { std::string output = "geometry.nc"; bool aux_files = false; std::string aux_dir; /* empty = directory of output */ double target_temperature_K = 300; double target_recyc_coef = 0.5; NcFormat nc_format = Auto; }`.
    *   A flag table per options struct: long name, short name, help text, a setter, and the default as text. The table is read by `apps/cli.cpp`, the bindings and the `--help` parity test.
    *   `validate()` checks ranges, e.g. `0 < tol < 1e-2` and `0 ≤ recyc ≤ 1`.
*   **Why:** one source of defaults and flags (G8).

##### 39. `src/omegah2csg/pipeline.hpp`, `pipeline.cpp` (Create, L5) [P3–P5]

*   **Proposed Changes**:
    *   `CsgGeometry compute_csg(Omega_h::Mesh&, const CommonOptions&)` and `CsgGeometry compute_csg(Library&, path, const CommonOptions&)`.
    *   `Degas2Geometry compute_degas2(…same two overloads…, const Degas2Options&)`.
    *   `void write_openmc_xml(const CsgGeometry&, const OpenMCOptions&, const HostArrayView<int32>* material_ids)` and `void write_degas2(Degas2Geometry&, const Degas2Options&)`. Both copy device arrays to the host on demand through `transfer`.
    *   `convert_to_openmc(Library&, path, opts)` and `convert_to_degas2(Library&, path, opts)`.
    *   Each stage runs inside `StageTimer`.
*   **Why:** the only place that chains stages.

##### 40. `src/omegah2csg/profiling.hpp` (Create) [P3]

*   **Proposed Changes**: `class StageTimer` (RAII) calls `Kokkos::Profiling::pushRegion("omegah2csg::<stage>")`, fences and records time with `Kokkos::Timer`, and on destruction pops the region and appends to a `TimingReport` that `--timing` prints.
*   **Why:** per-stage timing for benchmarks and users.

##### 41. `src/omegah2csg/runtime_info.hpp`, `runtime_info.cpp` (Create) [P6]

*   **Proposed Changes**: `RuntimeInfo collect_runtime_info()` returns:
    *   the omegah2csg version;
    *   the omega-h version and commit (from `Omega_h_config.h`);
    *   the Kokkos version, execution space name and concurrency;
    *   whether MPI is enabled;
    *   library file paths via `dladdr` on one symbol each from libomega_h and libkokkoscore.
*   **Why:** `--version --verbose` and `show_config()`, and the same-library guard.

##### 42. `src/omegah2csg/io/openmc_xml.hpp`, `openmc_xml.cpp` (Create, L4, host compiler) [P4]

*   **Proposed Changes**:
    *   `void write_openmc_xml(const OpenMCHostTables&, const OpenMCOptions&, path)`.
    *   `OpenMCHostTables` holds host views of surface type, coefficients, face edges and sides, boundary edges, material IDs, and the boundary type per surface.
    *   Builds a pugixml document:
        *   one `<surface id type coeffs [boundary]>` per plane or cylinder;
        *   for each cone, the same surfaces and region expression that `openmc.model.ZConeOneSided` produces (cone plus plane), documented in a comment with the OpenMC version it was checked against;
        *   one `<cell id material|void region universe>` per face.
    *   Numbers are formatted with `std::to_chars`.
    *   Saves through `atomic_file`.
*   **Why:** C++ output path for large meshes (CUJ 4). pugixml was chosen by the owner, with the hand-rolled fallback rule.

##### 43. `src/omegah2csg/io/netcdf_file.hpp`, `netcdf_file.cpp` (Create, host compiler) [P5]

*   **Proposed Changes**: RAII `NcFile` with:
    *   `create(path, format)` (CDF-2 → `NC_64BIT_OFFSET`, CDF-5 → `NC_64BIT_DATA`, netCDF-4 → `NC_NETCDF4|NC_CLASSIC_MODEL`);
    *   `def_dim`, `def_var`, `enddef`, `put(var, HostArrayView)`, `close`.
    *   Every netcdf-c status code is checked and raised as `std::runtime_error` with `nc_strerror`.
*   **Why:** a small, checked wrapper over the C API instead of netcdf-cxx4.

##### 44. `src/omegah2csg/io/degas2_netcdf.hpp`, `degas2_netcdf.cpp` (Create, L4) [P5]

*   **Proposed Changes**: `void write_degas2_netcdf(Degas2Geometry&, path, NcFormat, fetch_fn)`:
    1.  Compute every variable's byte size. In `Auto` mode, if any non-last fixed variable exceeds the CDF-2 limit, pick CDF-5 and log it.
    2.  Define all dimensions and variables from the schema.
    3.  Call `enddef`.
    4.  For each variable, call `fetch_fn(name)` (device → host if needed), then `put`, then release the host copy unless Python cached it.
*   **Why:** G10, with peak host memory bounded by the largest single variable.

##### 45. `src/omegah2csg/io/degas2_aux.hpp`, `degas2_aux.cpp` (Create, L4) [P5]

*   **Proposed Changes**: Writes `wallfile.txt`, `plasmafile.txt` and `sourcefile.txt` into `Degas2Options::aux_dir`, or the output's directory when that is empty. Formats are byte-identical to `convert2degas2.py:35-77`. The directory must already exist; if not, an `InputError` names it. The open-file leak in `sourcefile` is fixed.
*   **Why:** aux-file parity, with a configurable location.

##### 46. `src/omegah2csg/io/atomic_file.hpp`, `atomic_file.cpp` (Create) [P4]

*   **Proposed Changes**: `AtomicOutput(path, force)` checks whether the file exists (fails unless `force`), opens `path.tmp.<pid>`, and on `commit()` renames it into place. The destructor removes the temporary file if the write was not committed.
*   **Why:** no partial files on failure.

##### 47. `src/compute_surface.cpp`, `src/compute_surface.h` (Delete) [P3]

*   **Proposed Changes**: Live functions move to `geom`, `csg` and `wall`. Dead code is dropped: `read_cli`, `sort_curve`, `walk_on_curve_*`, `get_lower_left_corner`, `get_closest_wall_node`, `get_number_of_wall_nodes`, `get_closest_node_to_vertical_axis`, `findTwoSmallestIndices`.
    *   The pending uncommitted build fix is committed first in P0, so history keeps it.
*   **Why:** replaced by the layered modules.

##### 48. `src/capi.cpp`, `src/capi.h` (Delete) [P6]

*   **Proposed Changes**: Kept until P6 as the reference for the array equivalence test in P3, then removed.
*   **Why:** the ctypes layer is replaced.

##### 49. `src/Timer.cpp`, `src/Timer.h` (Delete) [P3]

*   **Why:** unused; replaced by `StageTimer`.

#### Native executables (`apps/`)

##### 50. `apps/CMakeLists.txt` (Create) [P6]

*   **Proposed Changes**:
    *   Two executables, `${OMEGAH2CSG_OPENMC_EXE_NAME}` and `${OMEGAH2CSG_DEGAS2_EXE_NAME}`, linking `omegah2csg::pipeline` and the I/O targets, with `kokkos_compilation`.
    *   Installed to `bin/` (non-SKBUILD) or `omegah2csg/bin/` (SKBUILD), with RPATH `$ORIGIN/../lib` and `$ORIGIN/../../omegah2csg.libs`.
*   **Why:** G8, CUJ 1.

##### 51. `apps/cli.hpp`, `apps/cli.cpp` (Create) [P6]

*   **Proposed Changes**:
    *   A small parser driven by the options flag tables, with no external dependency.
    *   Handles `--help` (generated from the table), `--version` (with `-v`, adds the `runtime_info` build configuration) and `--check`.
    *   `-v` is repeatable and raises `verbosity`. `-q` sets it to 0.
    *   Progress messages go to stderr according to `verbosity`. Errors always print.
    *   Unknown `--kokkos-*` and `--osh-*` flags are left for `Omega_h::Library`, which reads `argc` and `argv`.
    *   Exit codes: 0 success, 1 usage error, 2 input or geometry error, 3 I/O error.
*   **Why:** one CLI implementation used by both front ends.

##### 52. `apps/openmc_main.cpp`, `apps/degas2_main.cpp` (Create) [P6]

*   **Proposed Changes**:
    1.  Construct `Omega_h::Library lib(&argc, &argv)`.
    2.  Run `check_single_rank`.
    3.  Parse flags.
    4.  Call `pipeline::convert_to_*`.
    5.  Catch `InputError`, `GeometryError` and `std::exception` into exit codes.
*   **Why:** native entry points.

#### Python bindings (`bindings/`)

##### 53. `bindings/CMakeLists.txt` (Create) [P6]

*   **Proposed Changes**:
    *   `pybind11_add_module(_core MODULE … ${NO_EXTRAS_IF_CUDA})`, linking the pipeline and I/O targets, with `kokkos_compilation`.
    *   `install(TARGETS _core DESTINATION omegah2csg)`.
    *   RPATH `$ORIGIN` and `$ORIGIN/../omegah2csg.libs`.
*   **Why:** Python front end.

##### 54. `bindings/module.cpp` (Create) [P6]

*   **Proposed Changes**: `PYBIND11_MODULE(_core, m)`:
    *   registers exception translators (`InputError` → `ValueError`, `GeometryError` → a custom `omegah2csg.GeometryError(RuntimeError)`);
    *   calls the `bind_*` functions;
    *   sets `m.attr("__pybind11_internals_id__")`.

    It does **not** import PyOmega_h.
*   **Why:** module entry.

##### 55. `bindings/runtime.hpp`, `runtime.cpp` (Create) [P6]

*   **Proposed Changes**:
    *   `Omega_h::Library& library()` creates a static `std::shared_ptr<Library>` on first call, and registers `Py_AtExit(release_library)` once.
    *   `initialize(kokkos_args)` builds `argc`/`argv` from the list and constructs the Library with them. It raises an error if the Library already exists.
*   **Why:** G9, runtime ownership through omega-h's own `Library`.

##### 56. `bindings/mesh_guard.hpp`, `mesh_guard.cpp` (Create) [P6]

*   **Proposed Changes**: `void require_same_libomega_h(const Omega_h::Mesh&)`:
    *   uses `dladdr` on `&Omega_h::Mesh::nelems` from `_core`'s side;
    *   uses `dladdr` on the mesh object's vtable pointer;
    *   compares the two `dli_fname` results with `realpath`. If they differ, it raises `TypeError` naming both paths.
*   **Why:** the same-library guard for PyOmega_h interop (CUJ 5).

##### 57. `bindings/numpy_transfer.hpp` (Create) [P6]

*   **Proposed Changes**:
    *   `template<T> py::array_t<T> to_numpy(Read<T>, shape)`: allocates a C-contiguous NumPy array and calls `transfer::copy_to_host` into its buffer, so there is one copy.
    *   `template<T> HostArrayView<T> from_numpy(py::array_t<T, c_style|forcecast>)`.
*   **Why:** the only NumPy marshalling.

##### 58. `bindings/bind_options.cpp` (Create) [P6]

*   **Proposed Changes**: Binds `CommonOptions`, `OpenMCOptions`, `Degas2Options` and `NcFormat`, with keyword constructors whose defaults come from the C++ structs, plus `validate()`.
*   **Why:** defaults defined once.

##### 59. `bindings/bind_results.cpp` (Create) [P6]

*   **Proposed Changes**:
    *   **`CsgGeometry` (Python class).**
        *   Read-write properties per field, returning cached NumPy arrays with documented shapes. Examples: `surface_coeffs` → `(nedges, 6)`, `face_sides` → `(nfaces, 3)`.
        *   A `surface_type` returns a NumPy array of `SurfaceType` ints, and `SurfaceType` is bound as an enum.
        *   `material_ids` and `boundary_type` are user-settable host arrays, initialised to void and to the options' boundary type.
    *   **`Degas2Geometry`.**
        *   `__getitem__` / `__setitem__` by DEGAS2 variable name, plus `keys()` and `schema(name)`.
        *   Setting replaces the host array after a dtype check; shape is checked at write time by `validate`.
    *   The cache lives in the C++ object, which keeps a host override per field.
*   **Why:** G5, editable arrays.

##### 60. `bindings/bind_pipeline.cpp` (Create) [P6]

*   **Proposed Changes**:
    *   `compute_csg(source: str | os.PathLike, options)` and `compute_csg(mesh: Omega_h::Mesh&, options)`, with the GIL released after topology is built.
    *   The same pair for `compute_degas2`.
    *   `write_openmc_xml(geom, path, options)`, `write_degas2(geom, path, options)`, `convert_to_openmc(path, options)`, `convert_to_degas2(path, options)`.
    *   `initialize(kokkos_args)` and `show_config() -> dict`.
    *   Mesh overloads call `require_same_libomega_h` first.
*   **Why:** the Python API surface.

#### Python package (`python/omegah2csg/`)

##### 61. `python/omegah2csg/__init__.py` (Create) [P6]

*   **Proposed Changes**:
    *   Re-exports from `_core`: `compute_csg`, `compute_degas2`, `write_openmc_xml`, `write_degas2`, `convert_to_openmc`, `convert_to_degas2`, `CsgGeometry`, `Degas2Geometry`, `OpenMCOptions`, `Degas2Options`, `SurfaceType`, `GeometryError`, `initialize`, `show_config`.
    *   `to_openmc_universe` is loaded lazily from `openmc_builder`, so `openmc` is imported only when used.
    *   `__version__` comes from `_version.py`.
*   **Why:** the public API, with no heavy imports at start-up.

##### 62. `python/omegah2csg/openmc_builder.py` (Create) [P4]

*   **Proposed Changes**:
    *   `to_openmc_universe(geom, *, materials=None, boundary_type=None, cell_id_offset=0, surface_id_offset=0, universe_id=None) -> openmc.Universe`.
    *   Builds surfaces from `surface_type` and `surface_coeffs` (`ZPlane`, `ZCylinder`, `model.ZConeOneSided`).
    *   Builds cells from `face_edges` and `face_sides` as intersections of halfspaces.
    *   `materials`: `None` → void; a `Material` → all cells; a `dict` → by `offset_face` tag value; a sequence → per cell.
    *   Boundary types are applied from `boundary_edges`. This fixes the negative-index bug by construction.
    *   Raises `ImportError` with an install hint if `openmc` is missing.
*   **Why:** CUJ 3 and G5.

##### 63. `python/omegah2csg/_cli.py` (Create) [P6]

*   **Proposed Changes**:
    *   `openmc_main()` / `degas2_main()` locate the executable: first `importlib.resources.files("omegah2csg") / "bin" / <name>`, then `shutil.which(<name>)`, excluding this script's own path.
    *   Then `os.execv` with `sys.argv[1:]`.
    *   If neither is found, prints how to install it and exits 1.
*   **Why:** one CLI implementation.

##### 64. `python/omegah2csg/_core.pyi`, `python/omegah2csg/py.typed` (Create) [P6]

*   **Proposed Changes**: Stubs generated with `pybind11-stubgen omegah2csg._core` and committed. A `stubs` check in CI regenerates them and fails on any difference.
*   **Why:** typing and IDE help, and later API docs without importing the compiled module.

##### 65. `pythonAPI/` tree (Delete) [P6]

*   **Files:**
    *   `pythonAPI/omegah2csg/{__init__.py, _lib.py, config.py, OmegaHMesh.py, lib/__init__.py}`
    *   `pythonAPI/omegah2csg/convert2openmc/{__init__.py, __main__.py, convert2openmc.py, convert2openmc_cli.py}`
    *   `pythonAPI/omegah2csg/convert2degas2/{__init__.py, __main__.py, convert2degas2.py, convert2degas2_cli.py}`
*   **Proposed Changes**:
    *   Before deletion, `convert2openmc.py` and `convert2degas2.py` are copied to `tests/reference/` (item 74).
    *   They are also modified earlier, in P2, to carry the bug fixes:
        *   `convert2openmc.py:168-169` uses `abs()` IDs;
        *   `convert2openmc.py:171-176` uses a void default material;
        *   `convert2openmc.py:212` uses a range check;
        *   `convert2degas2.py:112` calls the method;
        *   `convert2degas2.py:624` computes the last zone volume directly.
*   **Why:** replaced by the C++ pipeline and the new package.

#### Tests

##### 66. `tests/CMakeLists.txt` (Create) [P3]

*   **Proposed Changes**:
    *   `find_package(Catch2 3 REQUIRED)` and `catch_discover_tests` for `tests/cpp/*`.
    *   Passes `OMEGAH2CSG_TEST_ASSETS=${CMAKE_SOURCE_DIR}/tests/assets` as a compile definition.
    *   When Python is enabled, registers `pytest tests/python` as a CTest test labelled `python`.
*   **Why:** one `ctest` command runs everything.

##### 67. `tests/cpp/support/topology_builders.hpp` (Create) [P3]

*   **Proposed Changes**: Hand-built `Topology` fixtures:
    *   a single triangle;
    *   two triangles sharing a vertical, a horizontal and a slanted edge;
    *   a 4-triangle "wall" strip with `offset_face` and `isOnWall`.

    Arrays are built from initializer lists.
*   **Why:** stage tests without mesh files (CUJ 6).

##### 68. `tests/cpp/test_geom.cpp` (Create) [P3]

*   **Proposed Changes**: Table-driven tests on the host:
    *   classification of vertical, horizontal, slanted and near-threshold edges;
    *   coefficient values for each type;
    *   `face_side` for points above and below a cone, inside and outside a cylinder, and on either side of a plane;
    *   a degenerate triangle returns 0;
    *   area, centroid, toroidal volume and bounding box against hand values.
*   **Why:** L0 correctness.

##### 69. `tests/cpp/test_topology.cpp`, `test_csg.cpp`, `test_wall.cpp`, `test_cell_quantities.cpp` (Create) [P3]

*   **Proposed Changes**:
    *   **Invariants.**
        *   Every interior edge's two faces have opposite sides.
        *   Boundary edges have exactly one face.
        *   Wall-edge count equals the `isOnWall` count.
        *   `sort_edge_faces` puts the positive face first.
        *   Volumes sum to the analytic torus-section volume for a rectangle mesh.
    *   **Asset meshes.** Comparisons against `tests/golden/*.npz` for `6elem`, `16elem` and `tagged-dg2mesh`.
    *   **Errors.** A malformed `Topology` makes `DeviceErrors` throw `GeometryError` with the right entity, and does not abort.
*   **Why:** stage-level coverage and G7.

##### 70. `tests/cpp/test_mesh_input.cpp`, `test_degas2_schema.cpp`, `test_degas2_netcdf.cpp`, `test_openmc_xml.cpp`, `test_options.cpp` (Create) [P3–P6]

*   **Proposed Changes**:
    *   `mesh_input`: `nparts = 2` rejected with the message; missing tag named.
    *   Schema: every variable in today's writer is present, with the right types and dimensions; `validate` catches a wrong shape and an out-of-range index.
    *   netCDF: round trip in CDF-2 and CDF-5; auto switch when a synthetic variable exceeds the limit; error mapping.
    *   XML: a tiny `CsgGeometry` writes the expected nodes; the cone expansion matches the pinned snippet; IDs offset.
    *   Options: defaults and validation ranges.
*   **Why:** unit coverage of the input, schema, writer and options layers.

##### 71. `tests/python/conftest.py` (Create) [P1]

*   **Proposed Changes**:
    *   Asset paths; a `tmp_path`-based output fixture.
    *   Markers `needs_openmc`, `needs_pyomega_h`, `needs_degas2`, `slow`, skipped automatically when the dependency is absent.
    *   Parametrisation over `OMEGAH2CSG_BACKEND` in P4–P8.
*   **Why:** shared fixtures; no test writes to the working directory (today's tests do).

##### 72. `tests/python/test_golden_degas2.py`, `test_golden_openmc.py`, `test_golden_arrays.py` (Create) [P1]

*   **Proposed Changes**:
    *   Run the converter on each asset mesh: Python before P5/P4, C++ after, through the backend switch.
    *   Compare against `tests/golden/` using `tests/support/compare.py`:
        *   **DEGAS2**: every variable, its dimensions and dtypes.
        *   **OpenMC**: XML compared by meaning, plus centroid point classification.
        *   **Arrays**: `.npz`.
    *   Also cover non-default options: `--target-temperature 500 --target-recyc-coef 0.2`, and tolerances 1e-10, 1e-6 and 1e-4.
*   **Why:** G4.

##### 73. `tests/python/test_api.py`, `test_edit_and_write.py`, `test_openmc_builder.py`, `test_openmc_equivalence.py`, `test_cli.py`, `test_runtime.py`, `test_pyomega_h_interop.py`, `test_bugfixes.py` (Create) [P4–P6]

*   **Proposed Changes**:

    | Test file | What it covers |
    |---|---|
    | `test_api.py` | Shapes and dtypes of every property |
    | `test_edit_and_write.py` | CUJ 2: edits appear in the file, untouched variables equal the direct conversion, invalid edits raise `ValueError` naming the variable |
    | `test_openmc_builder.py` | Materials handling (`None`, single, dict, sequence), boundary types, ID offsets |
    | `test_openmc_equivalence.py` | G6 |
    | `test_cli.py` | `--help` parity between the executable and the console script, exit codes, `--check`, `--force`, atomic output, `--aux-dir` placement and missing-directory error, `-v`/`-q` output levels |
    | `test_runtime.py` | Import order with and without PyOmega_h in subprocesses; `initialize` after start raises |
    | `test_pyomega_h_interop.py` | CUJ 5, including the mismatch message (`needs_pyomega_h`) |
    | `test_bugfixes.py` | Pins each P2 fix: reflective surfaces correct, boundary-layer check fires, `tol` default accepted |

*   **Why:** CUJ coverage.

##### 74. `tests/reference/convert2openmc_ref.py`, `tests/reference/convert2degas2_ref.py` (Create from move) [P6]

*   **Proposed Changes**: The P2-fixed Python converters. Used by `tests/golden/generate.py` and by the `python` backend until 0.1.0, then only for regenerating goldens.
*   **Why:** the oracle stays available.

##### 75. `tests/support/compare.py`, `tests/support/tolerances.toml` (Create) [P1]

*   **Proposed Changes**:
    *   Comparators for netCDF (per variable: integers exact, floats per tolerance table), XML (parse, normalise IDs, compare surfaces by type and rounded coefficients and cells by region), and `.npz`.
    *   Messages name the variable and the first differing index.
    *   `tolerances.toml` lists `rtol` and `atol` per variable, e.g. `zone_volume` atol relative to `universal_cell_vol`.
*   **Why:** G4 with backend-tolerant comparisons.

##### 76. `tests/golden/README.md`, `tests/golden/generate.py`, `tests/golden/<mesh>/…` (Create) [P1, regenerated P2]

*   **Proposed Changes**:
    *   `generate.py` runs the reference converters at a recorded commit and OpenMC version, writing:
        *   `geometry.nc`;
        *   the aux files;
        *   `geometry.xml`;
        *   `arrays.npz` (coefficients, types, sides, boundary, wall, cell quantities);
        *   for each of `6elem`, `16elem` and `simple_degas2_case`.
    *   The README records the commit, versions, command and date.
    *   Also adds a check that no golden edge lies within 100·eps of the tolerance threshold.
*   **Why:** frozen oracle.

##### 77. Existing `tests/test_mesh.py`, `tests/test_degas2_mesh_functions.py`, `tests/test_openmc_geometry_creation.py`, `tests/test_openmc_geometry_functions.py`, `tests/__init__.py` (Modify P1–P2, Delete P6)

*   **Proposed Changes**:
    *   **P1:** switch outputs to `tmp_path`, and declare `pandas` in the test extra.
    *   **P6:** delete. Their assertions are covered by the golden, unit and API tests. `read_*_from_file` helpers move to `tests/support/legacy_readers.py` for the `.dat` assets.
*   **Why:** replaced by the new suite.

#### Benchmarks (`bench/`)

##### 78. `bench/CMakeLists.txt`, `bench/copy_counter/copy_counter.cpp` (Create) [P7]

*   **Proposed Changes**: A Kokkos Tools shared library implementing `kokkosp_begin_deep_copy`. It counts copies between different memory spaces and writes JSON at finalize. It is loaded with `KOKKOS_TOOLS_LIBS`.
*   **Why:** verifies G1.

##### 79. `bench/make_mesh.py` (Create) [P7]

*   **Proposed Changes**:
    *   Uses PyOmega_h `build_box` (2D) to make meshes of 1e4, 1e5, 1e6 and 1e7 triangles.
    *   Shifts them to r > 0 and applies seeded vertex jitter so cone edges appear.
    *   Tags an inner region as plasma (`offset_face=0`), an outer ring as boundary layer (`offset_face=1`), and `isOnWall` on the interface loop.
    *   Writes `.osh` files with a manifest. Runs on the self-hosted runner, where PyOmega_h exists.
*   **Why:** scalable inputs without storing large files.

##### 80. `bench/run_bench.py`, `bench/meshes.toml`, `bench/README.md` (Create) [P7]

*   **Proposed Changes**:
    *   Runs each executable with `--timing` five times after one warm-up, and collects:
        *   per-stage median;
        *   peak RSS (`/usr/bin/time -v`);
        *   GPU memory high-water (`nvidia-smi --query` sampling);
        *   copy count.
    *   Writes JSON.
    *   Compares against the stored `main` baseline artifact: fails on more than 15% slower per stage, and on the pugixml memory rule.
    *   `meshes.toml` lists real XGC/DEGAS2 meshes with URLs and sha256.
*   **Why:** G2, G3 and the XML fallback rule.

### Verification Plan

*   **Compilation/Build**:
    *   `cmake -S . -B build -DOMEGAH2CSG_ENABLE_TESTS=ON && cmake --build build`, against Serial/OpenMP and CUDA Kokkos.
    *   `pip install -e . --no-build-isolation`.
    *   `cibuildwheel --platform linux`.
*   **Unit Tests**: `ctest --test-dir build --output-on-failure` (Catch2 stages, schema, writers, options).
*   **Python Tests**: `pytest`, at `OMP_NUM_THREADS=1` and `4`, and on the GPU runner.
*   **Static Analysis & Linting**:
    *   `pre-commit run --all-files` (ruff, clang-format 18, cmake-format).
    *   The CI grep guard for Kokkos spaces.
    *   The stub-diff check.
*   **Golden Equivalence**: `pytest -m "not slow" tests/python/test_golden_*` passes for both backends during P4–P8.
*   **Performance**: `bench/run_bench.py` on the self-hosted runner:
    *   linear-scaling check (t(1e5)/t(1e4) < 15);
    *   no stage more than 15% slower than baseline;
    *   copy count equal to the number of output arrays;
    *   pugixml peak RSS ≤ 4× file size at 1e7 triangles.
*   **E2E/Integration**:
    *   DEGAS2 `defineback`/`flighttest` on the converted `simple_degas2_case` (G10).
    *   OpenMC `openmc.Geometry.from_xml` plus centroid classification on the converted meshes.
    *   The wheel smoke test (`--version` and a small conversion) in a clean virtualenv.
*   **Test Coverage Requirement**:
    *   Every stage and writer has at least one unit test and one golden test.
    *   Every bug fixed in P2 has a test in `test_bugfixes.py`.
    *   Line-coverage percentages are not gated.

--------------------------------------------------------------------------------

<!-- Provenance: Indicates initial structural drafting via EGM. -->

*Drafted with EGM*
