/// @file make_mesh.cpp
/// @brief Writes a 2D box mesh with ~N triangles in r in [1, 3], z in [-1, 1],
/// interior vertices jittered (seeded) so edges become cones as well as
/// planes and cylinders. Usage: omegah2csg_make_mesh N out.osh

#include <Omega_h_build.hpp>
#include <Omega_h_file.hpp>
#include <Omega_h_for.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_mesh.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>

int main(int argc, char **argv) {
  Omega_h::Library lib(&argc, &argv);
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s NTRIANGLES out.osh\n", argv[0]);
    return 1;
  }
  const auto n = static_cast<Omega_h::LO>(std::sqrt(std::atof(argv[1]) / 2));
  auto mesh =
      Omega_h::build_box(lib.world(), OMEGA_H_SIMPLEX, 2.0, 2.0, 0.0, n, n, 0);
  const auto old = mesh.coords();
  const Omega_h::Real h = 2.0 / n;
  Omega_h::Write<Omega_h::Real> xy(old.size());
  auto f = OMEGA_H_LAMBDA(Omega_h::LO v) {
    const Omega_h::Real x = old[2 * v], y = old[2 * v + 1];
    const bool inside =
        x > 1e-12 && x < 2 - 1e-12 && y > 1e-12 && y < 2 - 1e-12;
    // deterministic pseudo-random jitter of a quarter cell
    const Omega_h::Real j = inside ? 0.25 * h * Kokkos::sin(12.9898 * v) : 0.0;
    xy[2 * v] = 1.0 + x + j;
    xy[2 * v + 1] = y - 1.0 - j;
  };
  Omega_h::parallel_for(mesh.nverts(), f, "omegah2csg::bench::jitter");
  mesh.set_coords(xy);
  Omega_h::binary::write(argv[2], &mesh);
  std::printf("%s: %d triangles\n", argv[2], mesh.nfaces());
  return 0;
}
