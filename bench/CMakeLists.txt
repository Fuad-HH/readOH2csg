# Kokkos Tools library counting copies between memory spaces
# (load with KOKKOS_TOOLS_LIBS=<path>/libomegah2csg_copy_counter.so).
add_library(omegah2csg_copy_counter SHARED copy_counter.cpp)

# Writes a jittered 2D box mesh (.osh) for scaling runs.
add_executable(omegah2csg_make_mesh make_mesh.cpp)
target_link_libraries(omegah2csg_make_mesh PRIVATE Omega_h::omega_h)
