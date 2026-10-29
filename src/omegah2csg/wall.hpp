#ifndef OMEGAH2CSG_WALL_HPP
#define OMEGAH2CSG_WALL_HPP
/// @file wall.hpp
/// @brief Wall data (L2). The wall is the interface between plasma faces
/// (offset_face 0) and boundary-layer faces (offset_face 1); it is not the
/// mesh boundary (edges with one face).

#include "omegah2csg/topology.hpp"

namespace omegah2csg {

struct Wall {
  Omega_h::LOs edges; ///< wall edges, ascending
  Omega_h::LOs faces; ///< 2 per wall edge: {plasma face, boundary-layer face}
};

/// Requires Topology::has_wall_tags. Throws GeometryError when the number of
/// wall edges differs from the number of `isOnWall` vertices.
Wall compute_wall(const Topology &t);

/// 2 per edge: {face on the + side, face on the - side}; -1 when missing.
Omega_h::LOs sort_edge_faces(const Topology &t,
                             Omega_h::Read<Omega_h::I8> face_sides);

/// Edges whose two faces are both plasma faces.
Omega_h::LO count_interior_edges(const Topology &t);

} // namespace omegah2csg

#endif
