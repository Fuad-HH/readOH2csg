cmake_minimum_required(VERSION 3.24)
project(
  omegah2csg
  VERSION 0.1.0
  LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ==================== OPTIONS ========================
option(OMEGAH2CSG_ENABLE_PYTHON "Build the omegah2csg._core Python module" ON)
if(DEFINED SKBUILD AND NOT OMEGAH2CSG_ENABLE_PYTHON)
  message(FATAL_ERROR "OMEGAH2CSG_ENABLE_PYTHON must be ON for scikit-build")
endif()
option(OMEGAH2CSG_ENABLE_APPS "Build the native executables" ON)
option(OMEGAH2CSG_ENABLE_TESTS "Register tests with CTest" OFF)
option(OMEGAH2CSG_ENABLE_BENCH "Build benchmark helpers" OFF)
set(OMEGAH2CSG_OPENMC_EXE_NAME
    convert2openmc
    CACHE STRING "Name of the OpenMC executable")
set(OMEGAH2CSG_DEGAS2_EXE_NAME
    convert2degas2
    CACHE STRING "Name of the DEGAS2 executable")

# =================== DEPENDENCIES ===================
find_package(Omega_h 11.4.0 CONFIG REQUIRED)
if(NOT Omega_h_USE_Kokkos)
  message(FATAL_ERROR "Omega_h must be built with Kokkos support enabled.")
endif()
find_package(pugixml CONFIG REQUIRED)

# netcdf-c ships a CMake config only when built with CMake; otherwise use
# its pkg-config file. Either way it is required.
find_package(netCDF CONFIG QUIET)
if(TARGET netCDF::netcdf)
  set(OMEGAH2CSG_NETCDF_TARGET netCDF::netcdf)
else()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(NETCDF REQUIRED IMPORTED_TARGET netcdf)
  set(OMEGAH2CSG_NETCDF_TARGET PkgConfig::NETCDF)
endif()

# Kokkos-dependent targets compile through Kokkos' compiler launcher
# (nvcc_wrapper on CUDA); host-only targets use the host compiler.
function(omegah2csg_kokkos_target target)
  if(COMMAND kokkos_compilation)
    kokkos_compilation(TARGET ${target})
  endif()
endfunction()

# =================== TARGETS ===================
add_subdirectory(src)
if(OMEGAH2CSG_ENABLE_APPS)
  add_subdirectory(apps)
endif()
if(OMEGAH2CSG_ENABLE_PYTHON)
  add_subdirectory(bindings)
endif()
if(OMEGAH2CSG_ENABLE_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
if(OMEGAH2CSG_ENABLE_BENCH)
  add_subdirectory(bench)
endif()

# =================== INSTALLATION (C++ consumers) ===================
if(NOT DEFINED SKBUILD)
  install(
    TARGETS omegah2csg_core omegah2csg_host
    EXPORT omegah2csg-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
  install(
    DIRECTORY src/omegah2csg
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING
    PATTERN "*.hpp")
  configure_package_config_file(
    cmake/omegah2csgConfig.cmake.in
    ${PROJECT_BINARY_DIR}/omegah2csg-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/omegah2csg)
  write_basic_package_version_file(
    ${PROJECT_BINARY_DIR}/omegah2csg-config-version.cmake
    COMPATIBILITY SameMinorVersion)
  install(FILES ${PROJECT_BINARY_DIR}/omegah2csg-config.cmake
                ${PROJECT_BINARY_DIR}/omegah2csg-config-version.cmake
          DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/omegah2csg)
  install(
    EXPORT omegah2csg-targets
    FILE omegah2csg-targets.cmake
    NAMESPACE omegah2csg::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/omegah2csg)
endif()
