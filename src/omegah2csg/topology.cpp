#include "omegah2csg/topology.hpp"

namespace omegah2csg {

Topology build_topology(Omega_h::Mesh &mesh) {
  Topology t;
  t.nverts = mesh.nverts();
  t.nedges = mesh.nedges();
  t.nfaces = mesh.nfaces();
  t.coords = mesh.coords();
  t.edge_verts = mesh.ask_down(Omega_h::EDGE, Omega_h::VERT).ab2b;
  t.face_verts = mesh.ask_down(Omega_h::FACE, Omega_h::VERT).ab2b;
  t.face_edges = mesh.ask_down(Omega_h::FACE, Omega_h::EDGE).ab2b;
  const auto up = mesh.ask_up(Omega_h::EDGE, Omega_h::FACE);
  t.edge_faces_offsets = up.a2ab;
  t.edge_faces = up.ab2b;
  t.has_wall_tags = mesh.has_tag(Omega_h::FACE, "offset_face") &&
                    mesh.has_tag(Omega_h::VERT, "isOnWall");
  if (t.has_wall_tags) {
    t.offset_face = mesh.get_array<Omega_h::LO>(Omega_h::FACE, "offset_face");
    t.is_on_wall = mesh.get_array<Omega_h::LO>(Omega_h::VERT, "isOnWall");
  }
  return t;
}

} // namespace omegah2csg
