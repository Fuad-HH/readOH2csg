References
==========

omega-h
-------
- SCOREC/omega_h, https://github.com/SCOREC/omega_h. Commit ``7647c8a`` was
  used: ``Omega_h_library.cpp`` (Library starts and finalizes only what it
  owns), ``Omega_h_kokkos.hpp`` (execution and memory spaces),
  ``Omega_h_file.cpp`` (``binary::read`` fails when ``nparts > ranks``),
  ``PyOmega_h_*.cpp`` (pybind11 bindings, import-time runtime).
- PyOmega_h lifecycle (``Py_AtExit``), https://github.com/SCOREC/omega_h/pull/233
- pybind11 3.1 ``keep_alive`` segfault, https://github.com/SCOREC/omega_h/issues/252

pybind11 and packaging
----------------------
- pybind11 build systems, https://pybind11.readthedocs.io/en/stable/compiling.html
- Sharing classes across modules (``module_local``),
  https://pybind11.readthedocs.io/en/stable/advanced/classes.html
- pybind11 changelog (3.0 ABI bump), https://pybind11.readthedocs.io/en/stable/changelog.html
- Internals v11 vs v12 breakage example, https://github.com/conda-forge/pyceres-feedstock/pull/35
- nanobind cannot interoperate with pybind11 types,
  https://nanobind.readthedocs.io/en/latest/porting.html
- scikit-build-core configuration, https://scikit-build-core.readthedocs.io/en/latest/configuration/index.html
- Scientific Python development guide (compiled packages),
  https://learn.scientific-python.org/development/guides/packaging-compiled/
- cibuildwheel options, https://cibuildwheel.pypa.io/en/stable/options/
- auditwheel, https://github.com/pypa/auditwheel

Kokkos
------
- Initialize/finalize rules, https://kokkos.org/kokkos-core-wiki/API/core/initialize_finalize/finalize.html
- Kokkos Tools callbacks (used by ``bench/copy_counter.cpp``),
  https://github.com/kokkos/kokkos-tools

Output formats
--------------
- netCDF file formats (CDF-1/2/5, netCDF-4),
  https://docs.unidata.ucar.edu/netcdf-c/current/file_format_specifications.html
- DEGAS2 (reader ``geometry.web`` uses ``ncopen``; ``netcdf.hweb`` types and
  dimension checks), https://github.com/gjwilkie/degas2
- OpenMC geometry XML, https://docs.openmc.org/en/stable/io_formats/geometry.html
- OpenMC ``ZConeOneSided``,
  https://docs.openmc.org/en/stable/pythonapi/generated/openmc.model.ZConeOneSided.html
- pugixml, https://pugixml.org/docs/manual.html

Prior art
---------
- pyAMReX (pybind11 + CMake, explicit initialize/finalize), https://github.com/AMReX-Codes/pyamrex
- DOLFINx Python package (nanobind + scikit-build-core), https://github.com/FEniCS/dolfinx
- OpenMC C API from Python (``openmc.lib``, ctypes, like omegah2csg 0.0.x),
  https://docs.openmc.org/en/stable/pythonapi/capi.html
