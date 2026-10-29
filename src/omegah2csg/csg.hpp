#ifndef OMEGAH2CSG_CSG_HPP
#define OMEGAH2CSG_CSG_HPP
/// @file csg.hpp
/// @brief Edge surfaces, face sides and boundary edges (L2).

#include "omegah2csg/topology.hpp"

namespace omegah2csg {

struct Surfaces {
  Omega_h::Read<Omega_h::I8> type; ///< SurfaceType per edge
  Omega_h::Reals coeffs;           ///< kNumCoeffs per edge
};

struct Boundary {
  Omega_h::LOs edges;               ///< edges with one adjacent face
  Omega_h::Read<Omega_h::I8> sides; ///< side of that face (+1/-1)
};

Surfaces compute_surfaces(const Topology &t, Omega_h::Real tol);

/// kFaceEdges sides per face, aligned with Topology::face_edges.
Omega_h::Read<Omega_h::I8>
compute_face_sides(const Topology &t, const Surfaces &s, Omega_h::Real tol);

Boundary compute_boundary(const Topology &t,
                          Omega_h::Read<Omega_h::I8> face_sides);

} // namespace omegah2csg

#endif
