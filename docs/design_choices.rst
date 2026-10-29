Design choices
==============

Each record gives the context, the decision and its consequences. The full
design is in ``docs/designs/cpp-pipeline-modernization-design.md``; sources are
in :doc:`references`.

.. contents::
   :local:

DC-001 All computation in C++ on the device
-------------------------------------------
:Status: Accepted (0.1.0)

**Context.** Meshes have millions of triangles. In 0.0.x most work ran in
per-element Python loops (one of them O(edges × wall edges)), and the C API
copied every result to the host element by element.

**Decision.** Every stage (topology, surfaces, face sides, boundary, wall, cell
quantities, DEGAS2 arrays) is a Kokkos kernel over Omega_h arrays. Python only
reads, edits and writes results.

**Consequences.** Kernels finish in milliseconds at 1e6 triangles on a GPU.
The time that remains is in output formatting (see DC-007). Adding a feature
means adding C++ code, not Python code.

DC-002 Omega_h memory and execution spaces only
-----------------------------------------------
:Status: Accepted

**Context.** Omega_h may run on ``Kokkos::OpenMP`` even in CUDA builds, and may
use shared or pinned memory. 0.0.x allocated views in Kokkos's default spaces.

**Decision.** Allocate with ``Omega_h::Write<T>``; launch with
``Omega_h::parallel_for`` or ``Kokkos::parallel_*`` with ``Omega_h::policy``.
A CI check forbids ``Kokkos::View<`` and ``Kokkos::parallel_for`` outside
``transfer.hpp``.

**Consequences.** Code runs wherever omega-h runs. Arrays are flat
(``Omega_h::LO`` sizes, 32-bit), with documented component counts and named
column constants (``types.hpp``).

DC-003 Layered stages and one pipeline
--------------------------------------
:Status: Accepted

**Decision.** L0 ``geom`` (per-entity math), L1 ``mesh_input``/``topology``
(adjacency built once), L2 ``csg``/``wall``/``cell_quantities``, L3 DEGAS2
assembly, L4 host-only writers, L5 ``pipeline`` (the only code that chains
stages).

**Consequences.** Stages take a ``Topology``, not a mesh, so they can be tested
on hand-built inputs. No stage repeats work done by another.

DC-004 Errors are exceptions, never aborts
------------------------------------------
:Status: Accepted

**Decision.** Kernels count failures and record the first bad entity
(``DeviceErrors``); the host throws ``GeometryError``. Input problems
(missing file, ``nparts > 1``, missing tags) raise ``InputError``, checked
before omega-h is called. Python maps these to ``RuntimeError`` and
``ValueError`` subclasses. The executables exit with code 2.

**Consequences.** A bad mesh does not kill an OpenMC Python session. omega-h can
still abort on a corrupt ``.osh`` file.

DC-005 Runtime owned by ``Omega_h::Library``
--------------------------------------------
:Status: Accepted

**Decision.** The executables construct ``Omega_h::Library(&argc, &argv)``.
``_core`` creates one Library lazily and releases it with ``Py_AtExit``. The
Library starts Kokkos/MPI only if nothing else has, and finalizes only what it
started, so it coexists with PyOmega_h in either import order.

**Consequences.** No custom Kokkos init/finalize code. Kokkos flags
(``--kokkos-*``) and omega-h flags (``--osh-*``) work on the executables.

DC-006 Python results are host copies, made on demand
-----------------------------------------------------
:Status: Accepted

**Decision.** ``CsgGeometry`` and ``Degas2Geometry`` keep device arrays. Each
property makes one NumPy copy on first access and caches it. Edits change the
host copy, and writers use the host copy when it exists. Edits are never copied
back to the device.

**Consequences.** Users can modify and extend arrays. The OpenMC writer checks
sizes, edge indices, sides and codes. The DEGAS2 writer checks that every
array's shape fits the schema and that variables sharing a dimension agree. It
does not check physics consistency. Re-running device stages on edited data is
not supported.

DC-007 OpenMC output: C++ writer plus Python builder
----------------------------------------------------
:Status: Accepted

**Decision.** ``write_openmc_xml`` (pugixml) and ``to_openmc_universe`` read the
same arrays and follow the same rules. Surface id = edge + 1 + offset; a cone's
disambiguation plane = nedges + edge + 1 + offset; cell id = face + 1 + offset.
Cones are OpenMC ``ZConeOneSided`` (``-`` side = ``-cone`` and ``±plane``;
``+`` = complement). The plane never carries a boundary condition.

**Consequences.** A test checks that both paths produce identical regions. The
XML writer dominates run time at 1e6 triangles (about 2 s); a hand-written
streaming writer can replace pugixml behind the same interface if needed.

DC-008 DEGAS2 output from one schema table
------------------------------------------
:Status: Accepted

**Decision.** ``degas2_schema.cpp`` lists every dimension and variable (names,
types and order as in DEGAS2). Dimension sizes are inferred from the array
shapes and checked for consistency. The default format is CDF-2 (64-bit
offset), which every DEGAS2 build reads (DEGAS2 opens files with ``ncopen``).
The writer switches to CDF-5 automatically only when a variable exceeds the
CDF-2 limit. netCDF-4 is available on request. netcdf-c is used directly.

**Consequences.** No HDF5 in wheels, so there is no clash with the HDF5 that
``h5py`` loads. Variables are written one at a time.

DC-009 pybind11, and file paths as the main input
-------------------------------------------------
:Status: Accepted

**Decision.** Bindings use pybind11, pinned to the version PyOmega_h uses,
because nanobind cannot share types with it. Functions take ``.osh`` paths.
PyOmega_h mesh objects are also accepted when PyOmega_h uses the same
libomega_h (checked with ``dladdr``) and the same pybind11 internals (warned
about at configure time).

**Consequences.** Wheels do not vendor PyOmega_h. Once omega-h publishes its
own wheel with shared libraries and a CMake config, omegah2csg can link against
it (stage B) instead of bundling its own copy.

DC-010 Native executables and console scripts
---------------------------------------------
:Status: Accepted

**Decision.** ``convert2openmc`` and ``convert2degas2`` are C++ programs. The
wheel's console scripts ``exec`` the bundled binaries, so there is one CLI
implementation. Flags and defaults come from the options structs. ``-v``
means verbose and ``--version`` has no short form.

DC-011 Dependencies are required
--------------------------------
:Status: Accepted

**Decision.** Omega_h (with Kokkos), pugixml, netcdf-c and pybind11 are found
with ``REQUIRED``. Nothing is fetched at configure time.

DC-012 Known limits
-------------------
- Partitioned meshes (``nparts > 1``) are rejected. Computation is serial
  (one MPI rank).
- 32-bit ids: Omega_h ``LO``, DEGAS2 ``i4`` and OpenMC ids are checked on the
  host.
- Wheels are OpenMP-only; GPU builds come from source or spack.

Glossary
--------
offset_face
  Face tag: 0 = plasma face, 1 = boundary-layer face.
isOnWall
  Vertex tag: 1 on the wall, the interface between plasma and boundary layer.
wall edge
  Edge whose two faces have different ``offset_face``. This is not the mesh
  boundary, whose edges have one face.
face side
  Side (+1/-1) of an edge's surface on which a face lies.
zone (DEGAS2)
  Region with uniform plasma properties: one per plasma face, one per
  boundary-layer face next to the wall, plus one catch-all zone.
sector (DEGAS2)
  One side of a wall edge where particles are scored.
stratum (DEGAS2)
  Particle source, one per wall point.
