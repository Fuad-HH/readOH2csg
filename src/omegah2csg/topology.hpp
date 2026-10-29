#ifndef OMEGAH2CSG_TOPOLOGY_HPP
#define OMEGAH2CSG_TOPOLOGY_HPP
/// @file topology.hpp
/// @brief Mesh adjacency extracted once (L1); every stage reads only this.

#include <Omega_h_mesh.hpp>

namespace omegah2csg {

/// Flat arrays; `edge_faces` is CSR with `edge_faces_offsets` (nedges + 1).
struct Topology {
  Omega_h::LO nverts = 0, nedges = 0, nfaces = 0;
  Omega_h::Reals coords;   ///< 2 * nverts, (r, z)
  Omega_h::LOs edge_verts; ///< 2 * nedges
  Omega_h::LOs face_verts; ///< 3 * nfaces
  Omega_h::LOs face_edges; ///< 3 * nfaces
  Omega_h::LOs edge_faces_offsets;
  Omega_h::LOs edge_faces;
  Omega_h::LOs offset_face; ///< nfaces, empty if the tag is absent
  Omega_h::LOs is_on_wall;  ///< nverts, empty if the tag is absent
  bool has_wall_tags = false;
};

Topology build_topology(Omega_h::Mesh &mesh);

} // namespace omegah2csg

#endif
