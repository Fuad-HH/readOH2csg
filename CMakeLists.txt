cmake_minimum_required(VERSION 3.20)
project(omega_h_read)


find_package(Omega_h REQUIRED)
# if Omega_h_USE_KOKKOS is off, error
if(NOT Omega_h_USE_Kokkos)
  # find an alternative later
  message(FATAL_ERROR "Omega_h must be built with Kokkos.")
endif()


add_executable(readOH readOH.cc)
target_link_libraries(readOH PUBLIC Omega_h::omega_h)
