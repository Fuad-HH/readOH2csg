cmake_minimum_required(VERSION 3.21)
project(
  omegah2csg
  VERSION 0.0.1
  LANGUAGES CXX C)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# ==================== OPTIONS ========================
option(omegah2csg_ENABLE_PYTHON "Build for Python bindings" ON)

# =================== CMAKE SETTINGS ===================
# RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
file(RELATIVE_PATH relDir ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}
     ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_INSTALL_RPATH $ORIGIN $ORIGIN/${relDir})

# =================== DEPENDENCIES ===================
find_package(Omega_h REQUIRED)

if(NOT Omega_h_USE_Kokkos)
  message(FATAL_ERROR "Omega_h must be built with Kokkos support enabled.")
endif()

# =================== BUILD TARGETS ===================
# Always create only one library libomegah2csg
set(omegah2csg_SOURCES src/Timer.cpp src/compute_surface.cpp src/capi.cpp)
set(omegah2csg_HEADERS src/Timer.h src/compute_surface.h src/capi.h)

if(omegah2csg_ENABLE_PYTHON)
  if(NOT BUILD_SHARED_LIBS)
    message(
      STATUS
        "Building Python bindings requires shared libraries. Overriding BUILD_SHARED_LIBS to ON."
    )
  endif()
  set(BUILD_SHARED_LIBS ON)
endif()

add_library(omegah2csg ${omegah2csg_SOURCES})
target_include_directories(
  omegah2csg PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
                    $<INSTALL_INTERFACE:include>)
target_link_libraries(omegah2csg PUBLIC Omega_h::omega_h)

# =================== Python Package ===================
if(omegah2csg_ENABLE_PYTHON)
  set(_py_lib_dir ${PROJECT_SOURCE_DIR}/pythonAPI/omegah2csg/lib)

  # Copy omegah2csg and its runtime dependencies into the Python package
  add_custom_command(
    TARGET omegah2csg
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:omegah2csg> ${_py_lib_dir}/
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:Omega_h::omega_h>
            ${_py_lib_dir}/
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:Kokkos::kokkoscore>
            ${_py_lib_dir}/
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:Kokkos::kokkoscontainers>
            ${_py_lib_dir}/
    COMMENT
      "Copying $<TARGET_FILE:omegah2csg> and other dependencies to ${PROJECT_SOURCE_DIR}/pythonAPI/omegah2csg/lib/ for Python bindings."
  )

  # Patch RPATH to $ORIGIN so bundled libraries find each other at runtime
  find_program(PATCHELF patchelf REQUIRED)
  set(_origin_rpath "$ORIGIN")
  add_custom_command(
    TARGET omegah2csg
    POST_BUILD
    COMMAND ${PATCHELF} --set-rpath ${_origin_rpath}
            ${_py_lib_dir}/$<TARGET_FILE_NAME:omegah2csg>
    COMMAND ${PATCHELF} --set-rpath ${_origin_rpath}
            ${_py_lib_dir}/$<TARGET_FILE_NAME:Omega_h::omega_h>
    COMMAND ${PATCHELF} --set-rpath ${_origin_rpath}
            ${_py_lib_dir}/$<TARGET_FILE_NAME:Kokkos::kokkoscore>
    COMMAND ${PATCHELF} --set-rpath ${_origin_rpath}
            ${_py_lib_dir}/$<TARGET_FILE_NAME:Kokkos::kokkoscontainers>
    COMMENT
      "Patching RPATH of bundled libraries to $ORIGIN for Python bindings."
    VERBATIM)

  # Create SONAME copies (e.g. libkokkoscore.so.5.0 from libkokkoscore.so.5.0.99)
  add_custom_command(
    TARGET omegah2csg
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -DLIB_DIR=${_py_lib_dir} -DPATCHELF=${PATCHELF} -P
            ${PROJECT_SOURCE_DIR}/cmake/create_soname_symlinks.cmake
    COMMENT "Creating SONAME symlinks for bundled libraries")
endif()

# =================== INSTALLATION ===================

# Modules Directory
set(cmakeModulesDir cmake)
configure_package_config_file(
  ${cmakeModulesDir}/omegah2csgConfig.cmake.in
  ${CMAKE_BINARY_DIR}/omegah2csg-config.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/omegah2csg)

install(
  TARGETS omegah2csg
  EXPORT omegah2csg-targets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(FILES ${omegah2csg_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(FILES ${PROJECT_BINARY_DIR}/omegah2csg-config.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/omegah2csg)

install(
  EXPORT omegah2csg-targets
  FILE omegah2csg-targets.cmake
  NAMESPACE omegah2csg::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/omegah2csg)
