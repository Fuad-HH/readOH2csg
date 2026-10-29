# Host-only code: options, schema and writers (no Omega_h/Kokkos headers).
add_library(
  omegah2csg_host STATIC
  omegah2csg/options.cpp omegah2csg/degas2_schema.cpp
  omegah2csg/io/atomic_file.cpp omegah2csg/io/openmc_xml.cpp
  omegah2csg/io/degas2_netcdf.cpp omegah2csg/io/degas2_aux.cpp)
target_include_directories(
  omegah2csg_host PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(
  omegah2csg_host PRIVATE $<BUILD_INTERFACE:pugixml::pugixml>
                          $<BUILD_INTERFACE:${OMEGAH2CSG_NETCDF_TARGET}>)

# Device code: stages, DEGAS2 assembly and the pipeline.
add_library(
  omegah2csg_core STATIC
  omegah2csg/mesh_input.cpp
  omegah2csg/topology.cpp
  omegah2csg/csg.cpp
  omegah2csg/wall.cpp
  omegah2csg/cell_quantities.cpp
  omegah2csg/degas2_geometry.cpp
  omegah2csg/degas2_assemble.cpp
  omegah2csg/profiling.cpp
  omegah2csg/runtime_info.cpp
  omegah2csg/pipeline.cpp)
target_link_libraries(omegah2csg_core PUBLIC omegah2csg_host Omega_h::omega_h
                                             ${CMAKE_DL_LIBS})
target_compile_definitions(
  omegah2csg_core PRIVATE OMEGAH2CSG_VERSION="${PROJECT_VERSION}")
omegah2csg_kokkos_target(omegah2csg_core)

add_library(omegah2csg::host ALIAS omegah2csg_host)
add_library(omegah2csg::core ALIAS omegah2csg_core)
