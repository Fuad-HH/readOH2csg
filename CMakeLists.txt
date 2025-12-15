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
# Temporary solution if PKG_CONFIG_PATH is not set in the environment It should
# search CMAKE_PREFIX_PATH but currently it is not searching there for some
# reason
if(DEFINED NETCDF_CXX4_PKG_PATH AND DEFINED NETCDF_C_PKG_PATH)
  set(ENV{PKG_CONFIG_PATH}
      "${NETCDF_CXX4_PKG_PATH}:$ENV{PKG_CONFIG_PATH}:${NETCDF_C_PKG_PATH}")
endif()

find_package(PkgConfig REQUIRED)
find_package(Omega_h REQUIRED)

if(NOT Omega_h_USE_Kokkos)
  message(FATAL_ERROR "Omega_h must be built with Kokkos support enabled.")
endif()

pkg_check_modules(netcdf-cxx4 REQUIRED IMPORTED_TARGET netcdf-cxx4)

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
target_link_libraries(omegah2csg PUBLIC Omega_h::omega_h PkgConfig::netcdf-cxx4)

add_executable(readOH src/readOH.cc)
target_link_libraries(readOH PRIVATE Omega_h::omega_h PkgConfig::netcdf-cxx4)

add_executable(get_adj_info src/get_adj_info.cc)
target_link_libraries(get_adj_info PRIVATE omegah2csg)

# =================== Python Package ===================
if(omegah2csg_ENABLE_PYTHON)
  add_custom_command(
    TARGET omegah2csg
    POST_BUILD
    COMMAND
      ${CMAKE_COMMAND} -E copy $<TARGET_FILE:omegah2csg>
      ${CMAKE_CURRENT_SOURCE_DIR}/pythonAPI/omegah2csg/lib/$<TARGET_FILE_NAME:omegah2csg>
    COMMENT
      "Copying $<TARGET_FILE:omegah2csg> to ${CMAKE_CURRENT_SOURCE_DIR}/pythonAPI/omegah2csg/lib/$<TARGET_FILE_NAME:omegah2csg>"
  )
endif()

# =================== INSTALLATION ===================

# Modules Directory
set(cmakeModulesDir cmake)
configure_package_config_file(
  ${cmakeModulesDir}/omegah2csgConfig.cmake.in
  ${CMAKE_BINARY_DIR}/omegah2csg-config.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/omegah2csg)

install(
  TARGETS readOH get_adj_info omegah2csg
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
