cmake_minimum_required(VERSION 3.20)
project(omega_h_read)

include(GNUInstallDirs)

# RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
file(RELATIVE_PATH relDir ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}
     ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CMAKE_INSTALL_RPATH $ORIGIN $ORIGIN/${relDir})

# Temporary solution if PKG_CONFIG_PATH is not set in the environment It should
# search CMAKE_PREFIX_PATH but currently it is not searching there for some
# reason
if(DEFINED NETCDF_CXX4_PKG_PATH AND DEFINED NETCDF_C_PKG_PATH)
  set(ENV{PKG_CONFIG_PATH}
      "${NETCDF_CXX4_PKG_PATH}:$ENV{PKG_CONFIG_PATH}:${NETCDF_C_PKG_PATH}")
endif()

find_package(PkgConfig REQUIRED)
find_package(Omega_h REQUIRED)
pkg_check_modules(netcdf-cxx4 REQUIRED IMPORTED_TARGET netcdf-cxx4)

add_executable(readOH readOH.cc)
target_link_libraries(readOH PRIVATE Omega_h::omega_h PkgConfig::netcdf-cxx4)

add_executable(get_adj_info get_adj_info.cc)
target_link_libraries(get_adj_info PRIVATE Omega_h::omega_h
                                           PkgConfig::netcdf-cxx4)

# ######### INSTALLATION ##########

install(TARGETS readOH get_adj_info RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
