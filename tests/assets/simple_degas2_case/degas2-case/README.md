# Run Degas2 Case with the Created Geometry File

1. First, install [`degas2`](https://github.com/gjwilkie/degas2) with its related dependencies.

Degas2 has a extensive list of dependencies and can be difficult to install. I have this following [`Spack`](https://spack.io/) configuration file that can be used to install `degas2` dependencies.

```yaml
spack:
  # add package specs to the `specs` list
  specs:
  - kokkos@4+openmp+serial
  - cabana@0.7.0
  - fftw
  - netlib-lapack
  - googletest
  - libszip
  - hdf5+hl+mpi
  - netcdf-c+mpi
  - netcdf-fortran
  - catch2
  #- petsc+fortran+metis+scalapack@3.21.4
  #- petsc+fortran+metis+scalapack@3.22.5
  - kokkos-kernels
  - cmake
  - python@3.9
  - petsc@3.15.0+fortran+metis+scalapack ^python@3.9
  - mpich@4
  - py-netcdf4
  - adios2@2.10.2
  view: true
  concretizer:
    unify: true
  packages:
    gcc:
      externals:
      - spec: gcc@11.5.0 languages:='c,c++,fortran'
        prefix: /usr
        extra_attributes:
          compilers:
            c: /usr/bin/gcc
            cxx: /usr/bin/g++
            fortran: /usr/bin/gfortran
```

after installing the dependencies, install `degas2` using the following command:

> [!IMPORTANT]
> Remember activating the spack environment before running the command below.

```bash
cmake -B build -S <your-degas2-source-dir> \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_INSTALL_PREFIX=build/install \
  -DCMAKE_C_COMPILER=`which mpicc` \
  -DCMAKE_CXX_COMPILER=`which mpicxx` \
  -DCMAKE_Fortran_COMPILER=`which mpifort` \
  -DUSE_MPI=OFF \
  -DSILO_HDF5=ON \
  -DCMAKE_LIBRARY_PATH=<your-degas2-source-dir>/deps/silo/lib \
  -DSILO_INCLUDE_DIRS=<your-degas2-source-dir>/deps/silo/include \
  -DGRAPH_FILE=SILO

cmake --build build -j 16 --target flighttest datasetup defineback tallysetup problemsetup
```

> [!TIP]
> Installation may not work since some of the targets may not compile. Just use them from the build directory.

2. Go to the [`degas2-case`](./) folder.
3. Modify the [`degas2.in`](./degas2.in) file to point to your correct `degas2` source directory. Only the first parts of the
paths need to be modified (until the `degas2` folder). The rest of the path should be the same as the structure of the `degas2` source directory.
4. Run `datasetup` and it should not throw any error if the file names in `degas2.in` are correct.
5. Run `problemsetup`.
6. Run `defineback db.in` which will create `background.nc` file. Here we are only running with volume source.
7. Run `tallysetup`.
8. Run `flighttest`. It will create `output.nc`, `density.out` etc. which contain the results of the simulation.