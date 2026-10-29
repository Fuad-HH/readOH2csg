#ifndef OMEGAH2CSG_CSG_GEOMETRY_HPP
#define OMEGAH2CSG_CSG_GEOMETRY_HPP
/// @file csg_geometry.hpp
/// @brief Result of compute_csg(): device arrays shared by both outputs.
/// Each mesh edge is one surface and each mesh triangle (face) one cell.

#include "omegah2csg/topology.hpp"

namespace omegah2csg {

struct CsgGeometry {
  Topology topology;
  Omega_h::Read<Omega_h::I8> surface_type;   ///< nedges, SurfaceType
  Omega_h::Reals surface_coeffs;             ///< kNumCoeffs * nedges
  Omega_h::LOs face_edges;                   ///< 3 * nfaces
  Omega_h::Read<Omega_h::I8> face_sides;     ///< 3 * nfaces, +1/-1
  Omega_h::LOs boundary_edges;               ///< nbdry, unsigned edge ids
  Omega_h::Read<Omega_h::I8> boundary_sides; ///< nbdry
  Omega_h::Reals cell_volume;                ///< nfaces
  Omega_h::Reals cell_centroid;              ///< 2 * nfaces
  Omega_h::Reals cell_bbox;                  ///< 4 * nfaces
  Omega_h::LOs edge_faces_sorted;            ///< 2 * nedges, {+ face, - face}
  Omega_h::LOs wall_edges;                   ///< nwall (wall tags only)
  Omega_h::LOs wall_faces;                   ///< 2 * nwall (wall tags only)
  Omega_h::LOs face_region_tag;              ///< nfaces = offset_face, if any
};

} // namespace omegah2csg

#endif
