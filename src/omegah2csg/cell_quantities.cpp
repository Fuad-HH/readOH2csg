#include "omegah2csg/cell_quantities.hpp"
#include "omegah2csg/geom.hpp"

#include <Omega_h_for.hpp>

namespace omegah2csg {

CellQuantities compute_cell_quantities(const Topology &t) {
  const auto coords = t.coords;
  const auto fv = t.face_verts;
  Omega_h::Write<Omega_h::Real> volume(t.nfaces);
  Omega_h::Write<Omega_h::Real> centroid(kCentroidComps * t.nfaces);
  Omega_h::Write<Omega_h::Real> bbox(kBBoxComps * t.nfaces);
  auto f = OMEGA_H_LAMBDA(Omega_h::LO face) {
    const auto a = Omega_h::get_vector<2>(coords, fv[3 * face]);
    const auto b = Omega_h::get_vector<2>(coords, fv[3 * face + 1]);
    const auto c = Omega_h::get_vector<2>(coords, fv[3 * face + 2]);
    const auto ctr = triangle_centroid(a, b, c);
    const auto box = triangle_bbox(a, b, c);
    volume[face] = toroidal_volume(triangle_area(a, b, c), ctr[0]);
    for (int k = 0; k < kCentroidComps; ++k)
      centroid[kCentroidComps * face + k] = ctr[k];
    for (int k = 0; k < kBBoxComps; ++k)
      bbox[kBBoxComps * face + k] = box[k];
  };
  Omega_h::parallel_for(t.nfaces, f, "omegah2csg::cell_quantities");
  return {volume, centroid, bbox};
}

Omega_h::Vector<4> universal_bounds(Omega_h::Reals bbox) {
  const Omega_h::LO n = bbox.size() / kBBoxComps;
  Omega_h::Vector<4> out;
  for (int k = 0; k < 2; ++k) {
    Kokkos::parallel_reduce(
        "omegah2csg::bounds_min", Omega_h::policy(n),
        OMEGA_H_LAMBDA(Omega_h::LO i, Omega_h::Real & v) {
          v = Kokkos::min(v, bbox[kBBoxComps * i + k]);
        },
        Kokkos::Min<Omega_h::Real>(out[k]));
    Kokkos::parallel_reduce(
        "omegah2csg::bounds_max", Omega_h::policy(n),
        OMEGA_H_LAMBDA(Omega_h::LO i, Omega_h::Real & v) {
          v = Kokkos::max(v, bbox[kBBoxComps * i + 2 + k]);
        },
        Kokkos::Max<Omega_h::Real>(out[2 + k]));
  }
  return out;
}

} // namespace omegah2csg
