cmake_minimum_required(VERSION 3.20)
project(omega_h_read)
find_package(PkgConfig REQUIRED)


find_package(Omega_h REQUIRED)
pkg_check_modules(netcdf-cxx4 REQUIRED IMPORTED_TARGET netcdf-cxx4)


add_executable(readOH readOH.cc)
target_link_libraries(readOH PRIVATE Omega_h::omega_h PkgConfig::netcdf-cxx4)
