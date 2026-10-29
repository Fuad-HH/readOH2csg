find_package(
  Python 3.11 REQUIRED
  COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# Interop with PyOmega_h needs the same pybind11 internals ABI.
execute_process(
  COMMAND ${Python_EXECUTABLE} -c "import PyOmega_h; print(PyOmega_h.__file__)"
  OUTPUT_VARIABLE pyomega_h_file
  OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(pyomega_h_file)
  file(STRINGS ${pyomega_h_file} pyomega_h_internals
       REGEX "__pybind11_internals_v[0-9]+" LIMIT_COUNT 1)
  string(REGEX MATCH "__pybind11_internals_v[0-9]+" pyomega_h_internals
               "${pyomega_h_internals}")
  message(STATUS "PyOmega_h: ${pyomega_h_file} (${pyomega_h_internals})")
  if(NOT pyomega_h_internals MATCHES "_v${pybind11_INTERNALS_VERSION}$"
     AND DEFINED pybind11_INTERNALS_VERSION)
    message(
      WARNING "PyOmega_h uses ${pyomega_h_internals}, but pybind11 "
              "${pybind11_VERSION} uses internals v${pybind11_INTERNALS_VERSION}; "
              "passing PyOmega_h meshes to omegah2csg will fail.")
  endif()
endif()

if(Kokkos_ENABLE_CUDA)
  set(omegah2csg_pybind_opts NO_EXTRAS)
endif()
pybind11_add_module(
  _core MODULE ${omegah2csg_pybind_opts} module.cpp runtime.cpp
  bind_options.cpp bind_results.cpp bind_pipeline.cpp)
target_link_libraries(_core PRIVATE omegah2csg_core)
target_compile_definitions(_core PRIVATE OMEGAH2CSG_VERSION="${PROJECT_VERSION}")
omegah2csg_kokkos_target(_core)
set_target_properties(_core PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
if(DEFINED SKBUILD)
  install(TARGETS _core DESTINATION ${SKBUILD_PROJECT_NAME})
else()
  install(TARGETS _core DESTINATION ${CMAKE_INSTALL_LIBDIR}/python/omegah2csg)
endif()
