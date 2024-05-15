cmake_minimum_required(VERSION 3.20)
project(omega_h_read)


find_package(Omega_h REQUIRED)


add_executable(readOH readOH.cc)
target_link_libraries(readOH PRIVATE Omega_h::omega_h)
