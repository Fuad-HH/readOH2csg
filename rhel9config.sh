CC=mpicc CXX=mpicxx \
cmake -S . -B build \
  -DOmega_h_DIR=/lore/hasanm4/wsources/omega_h_scorec/install/lib64/cmake/Omega_h \
  -DKokkos_DIR=/lore/hasanm4/Kokkos/kokkosInstallrhel9omp/lib64/cmake/Kokkos/ \
  -DNETCDF_CXX4_RHEL9_ROOT=$NETCDF_CXX4_RHEL9_ROOT \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON 

# only work for rhel9 machines

