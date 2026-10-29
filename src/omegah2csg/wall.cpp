#include "omegah2csg/wall.hpp"
#include "omegah2csg/device_errors.hpp"
#include "omegah2csg/types.hpp"

#include <Omega_h_array_ops.hpp>
#include <Omega_h_for.hpp>
#include <Omega_h_map.hpp>

namespace omegah2csg {

namespace {
/// Marks edges with two faces for which `pred(face0, face1)` holds.
template <class Pred>
Omega_h::Read<Omega_h::I8> mark_two_face_edges(const Topology &t, Pred pred) {
  const auto off = t.edge_faces_offsets;
  const auto ef = t.edge_faces;
  Omega_h::Write<Omega_h::I8> marks(t.nedges);
  auto f = OMEGA_H_LAMBDA(Omega_h::LO e) {
    marks[e] = (off[e + 1] - off[e]) == 2 && pred(ef[off[e]], ef[off[e] + 1]);
  };
  Omega_h::parallel_for(t.nedges, f, "omegah2csg::wall::mark");
  return marks;
}
} // namespace

Wall compute_wall(const Topology &t) {
  const auto tag = t.offset_face;
  const auto edges = Omega_h::collect_marked(mark_two_face_edges(
      t, OMEGA_H_LAMBDA(Omega_h::LO a, Omega_h::LO b) {
        return tag[a] != tag[b];
      }));
  const Omega_h::LO npoints = Omega_h::get_sum(t.is_on_wall);
  if (edges.size() != npoints)
    throw GeometryError("wall", edges.size(),
                        "number of wall edges differs from the " +
                            std::to_string(npoints) + " isOnWall vertices");
  const auto off = t.edge_faces_offsets;
  const auto ef = t.edge_faces;
  Omega_h::Write<Omega_h::LO> faces(2 * edges.size());
  auto f = OMEGA_H_LAMBDA(Omega_h::LO i) {
    const Omega_h::LO f0 = ef[off[edges[i]]], f1 = ef[off[edges[i]] + 1];
    const bool swap = tag[f0] == 1;
    faces[2 * i] = swap ? f1 : f0;
    faces[2 * i + 1] = swap ? f0 : f1;
  };
  Omega_h::parallel_for(edges.size(), f, "omegah2csg::wall::faces");
  return {edges, faces};
}

Omega_h::LOs sort_edge_faces(const Topology &t,
                             Omega_h::Read<Omega_h::I8> face_sides) {
  const auto off = t.edge_faces_offsets;
  const auto ef = t.edge_faces;
  const auto fe = t.face_edges;
  Omega_h::Write<Omega_h::LO> sorted(2 * t.nedges);
  DeviceErrors errors;
  auto f = OMEGA_H_LAMBDA(Omega_h::LO e) {
    const Omega_h::LO f0 = ef[off[e]];
    const Omega_h::LO f1 = (off[e + 1] - off[e]) == 2 ? ef[off[e] + 1] : -1;
    Omega_h::I8 side = 0;
    for (int j = 0; j < kFaceEdges; ++j)
      if (fe[kFaceEdges * f0 + j] == e)
        side = face_sides[kFaceEdges * f0 + j];
    if (side == 0)
      errors.report(e);
    sorted[2 * e] = side == -1 ? f1 : f0;
    sorted[2 * e + 1] = side == -1 ? f0 : f1;
  };
  Omega_h::parallel_for(t.nedges, f, "omegah2csg::wall::sort_edge_faces");
  errors.throw_if_any("sort_edge_faces", "edge has no side in its face");
  return sorted;
}

Omega_h::LO count_interior_edges(const Topology &t) {
  const auto tag = t.offset_face;
  return Omega_h::get_sum(mark_two_face_edges(
      t, OMEGA_H_LAMBDA(Omega_h::LO a, Omega_h::LO b) {
        return tag[a] == 0 && tag[b] == 0;
      }));
}

} // namespace omegah2csg
