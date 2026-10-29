#include "omegah2csg/csg.hpp"
#include "omegah2csg/device_errors.hpp"
#include "omegah2csg/geom.hpp"

#include <Omega_h_for.hpp>
#include <Omega_h_map.hpp>

namespace omegah2csg {

Surfaces compute_surfaces(const Topology &t, Omega_h::Real tol) {
  const auto coords = t.coords;
  const auto ev = t.edge_verts;
  Omega_h::Write<Omega_h::I8> type(t.nedges);
  Omega_h::Write<Omega_h::Real> coeffs(t.nedges * kNumCoeffs);
  auto f = OMEGA_H_LAMBDA(Omega_h::LO e) {
    const auto p0 = Omega_h::get_vector<2>(coords, ev[2 * e]);
    const auto p1 = Omega_h::get_vector<2>(coords, ev[2 * e + 1]);
    const SurfaceType st = classify_edge(p0, p1, tol);
    const auto c = surface_coeffs(p0, p1, st);
    type[e] = static_cast<Omega_h::I8>(st);
    for (int k = 0; k < kNumCoeffs; ++k)
      coeffs[e * kNumCoeffs + k] = c[k];
  };
  Omega_h::parallel_for(t.nedges, f, "omegah2csg::csg::surfaces");
  return {type, coeffs};
}

Omega_h::Read<Omega_h::I8>
compute_face_sides(const Topology &t, const Surfaces &s, Omega_h::Real tol) {
  const auto coords = t.coords;
  const auto fv = t.face_verts;
  const auto fe = t.face_edges;
  const auto type = s.type;
  const auto coeffs = s.coeffs;
  Omega_h::Write<Omega_h::I8> sides(t.nfaces * kFaceEdges);
  DeviceErrors errors;
  auto f = OMEGA_H_LAMBDA(Omega_h::LO face) {
    const auto v0 = Omega_h::get_vector<2>(coords, fv[3 * face]);
    const auto v1 = Omega_h::get_vector<2>(coords, fv[3 * face + 1]);
    const auto v2 = Omega_h::get_vector<2>(coords, fv[3 * face + 2]);
    for (int j = 0; j < kFaceEdges; ++j) {
      const Omega_h::LO e = fe[kFaceEdges * face + j];
      const auto c = Omega_h::get_vector<kNumCoeffs>(coeffs, e);
      const auto side =
          face_side(v0, v1, v2, c, static_cast<SurfaceType>(type[e]), tol);
      if (side == 0)
        errors.report(face);
      sides[kFaceEdges * face + j] = side;
    }
  };
  Omega_h::parallel_for(t.nfaces, f, "omegah2csg::csg::face_sides");
  errors.throw_if_any("face_sides",
                      "cannot decide the side of an edge surface; try a "
                      "smaller --tol");
  return sides;
}

Boundary compute_boundary(const Topology &t,
                          Omega_h::Read<Omega_h::I8> face_sides) {
  const auto off = t.edge_faces_offsets;
  const auto ef = t.edge_faces;
  const auto fe = t.face_edges;
  Omega_h::Write<Omega_h::I8> marks(t.nedges);
  auto mark = OMEGA_H_LAMBDA(Omega_h::LO e) {
    marks[e] = (off[e + 1] - off[e]) == 1;
  };
  Omega_h::parallel_for(t.nedges, mark, "omegah2csg::csg::mark_boundary");
  const Omega_h::LOs edges = Omega_h::collect_marked(marks);
  Omega_h::Write<Omega_h::I8> sides(edges.size());
  auto side = OMEGA_H_LAMBDA(Omega_h::LO i) {
    const Omega_h::LO e = edges[i];
    const Omega_h::LO face = ef[off[e]];
    for (int j = 0; j < kFaceEdges; ++j)
      if (fe[kFaceEdges * face + j] == e)
        sides[i] = face_sides[kFaceEdges * face + j];
  };
  Omega_h::parallel_for(edges.size(), side, "omegah2csg::csg::boundary_sides");
  return {edges, sides};
}

} // namespace omegah2csg
