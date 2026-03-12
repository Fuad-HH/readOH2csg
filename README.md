# Convert `Omega_h` Mesh to CSG Format
[![License](https://img.shields.io/badge/license-BSD--3--Clause-02B36C)](https://github.com/Fuad-HH/readOH2csg/blob/parallel/LICENSE)
[![GitHub Actions build status (Linux)](https://github.com/Fuad-HH/readOH2csg/actions/workflows/ci.yml/badge.svg?branch=parallel)](https://github.com/Fuad-HH/readOH2csg/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/Fuad-HH/readOH2csg?include_prereleases)](https://github.com/Fuad-HH/readOH2csg/releases)

Efficiently convert Tokamak mesh of `Omega_h` (`.osh`) format
to Constructive Solid Geometry (CSG). It supports creating
CSG representation in
1. `Degas2`'s `geometry.nc` format
2. `OpenMC`'s geometry XML or a Python [`openmc.Universe`](https://docs.openmc.org/en/stable/pythonapi/generated/openmc.Universe.html) object.


## Installation

### 1. Pre-built Wheel
This is the recommended way to install this package. We provide a pre-built wheel (`Kokkos` with `OpenMP` backend) for Python. Follow these steps to install the latest version of `readOH2csg` using pip:

Wheels are built in CI and uploaded as workflow artifacts for each Python version and build type.

1. On any Linux system, make sure you have Python (Version ≥ 3.11) installed. Go to the terminal
and `cd` to the directory where you want to install the Python virtual environment.
```bash
python -m venv readoh2csg-env # or any name you prefer
source readoh2csg-env/bin/activate
```
>[!TIP]
> Make sure that the Python version of the virtual environment is 3.11 or higher. You can check it by running `python --version`.

2. Install dependencies:
```bash
pip install --extra-index-url https://shimwell.github.io/wheels openmc
pip install netCDF4==1.7.2
```
3. Install `readOH2csg`:

```bash
pip install omegah2csg
```
Or if you want to install the latest release from GitHub:
```bash
pip install <link-to-release-wheel-file> omegah2csg
```
>[!TIP]
> Find the latest release on the [Releases Page](https://github.com/Fuad-HH/readOH2csg/releases/)
> and copy the link to the wheel file for your Python version and system architecture.

4. Try running the CLI tool, and it should print the help message:
```bash
convert2degas2 --help
```


### 2. From Source
It depends on [`Omega_h`](https://github.com/SCOREC/omega_h). [`Omega_h`](https://github.com/SCOREC/omega_h) has to be built
with [`Kokkos`](https://github.com/kokkos/kokkos) and it supports any [`Kokkos`](https://github.com/kokkos/kokkos) backend.

Follow these steps to build and install `readOH2csg` from source:
1. Make sure you have compilers (at least `g++` or some alternative), `Python` (Version ≥ 3.11), and `CMake` available.
2. Install [`Kokkos`](https://github.com/kokkos/kokkos) following the instructions in
[Kokkos Build Documentation](https://kokkos.org/kokkos-core-wiki/get-started/building-from-source.html#configuring-and-building-kokkos). You can choose any backend supported. Here's an example for building with the `OpenMP` backend:
```bash
git clone --depth=2 --branch 4.7.02 https://github.com/kokkos/kokkos.git
cd kokkos
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=g++ \
  -DBUILD_SHARED_LIBS=ON \
  -DKokkos_ENABLE_OPENMP=ON \
  -DCMAKE_INSTALL_PREFIX=<Kokkos_install_dir>

cmake --build build -j4 --target install
```
3. Install [`Omega_h`](https://github.com/SCOREC/omega_h) with `Kokkos` support:
```bash
git clone https://github.com/SCOREC/omega_h.git
cd omega_h
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=g++ \
  -DBUILD_SHARED_LIBS=ON \
  -DOmega_h_USE_Kokkos=ON \
  -DKokkos_ROOT=<Kokkos_install_dir> \
  -DCMAKE_INSTALL_PREFIX=<Omega_h_install_dir>

cmake --build build -j4 --target install
```

4. Now, install `readOH2csg`:
```bash
git clone https://github.com/Fuad-HH/readOH2csg.git --branch parallel
cd readOH2csg

# configure
cmake -S . -B build \
  -DOmega_h_ROOT=<Omega_h_install_dir> \
  -DKokkos_ROOT=<Kokkos_install_dir> \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=<readOH2csg_install_dir>

# build
cmake --build build -j4 --target install
```
>[!WARNING]
> For now, only the Python API and CLI work. Please do not turn off the option for Python bindings. It is
> enabled by default.

5. Create Python virtual environment and install dependencies as described in the [Pre-built Wheel](#1-pre-built-wheel) section.
6. Install `readOH2csg` and run the CLI tool, and it should print the help message:
```bash
# from the source directory
export KOKKOS_ROOT=<Kokkos_install_dir>
export OMEGA_H_ROOT=<Omega_h_install_dir>
python -m pip install .
convert2degas2 --help
```

## Usage
Check an example case in [`tests/assets/simple_degas2_case/degas2-case`](tests/assets/simple_degas2_case/degas2-case/). Please read the [`README.md`](tests/assets/simple_degas2_case/degas2-case/README.md) file first for detailed instructions on how to run the example case.

To run an `OpenMC` example, running `convert2openmc <mesh-name>` will generate the `geometry.xml` file and,
run the case following [OpenMC Documentation](https://docs.openmc.org).

>[!TIP]
> If `omegah2csg` was installed with `Kokkos` `OpenMP` backend or using `pip`, try setting `OpenMP` environment variables for faster execution.
> ```bash
> export OMP_PROC_BIND=spread && export OMP_PLACES=threads && export OMP_NUM_THREADS=<number of threads you want>
> ```

## Documentation
More details on strategies, logic, and math are included in the [`doc/`](doc/) directory.
### Developer's Guide
For myself and future development, use the [Developer's Guide](doc/DEVELOPERS_GUIDE.md) to understand the code structure, design decisions, and how to contribute to this project.
