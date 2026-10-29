#ifndef OMEGAH2CSG_DEGAS2_ASSEMBLE_HPP
#define OMEGAH2CSG_DEGAS2_ASSEMBLE_HPP
/// @file degas2_assemble.hpp
/// @brief Builds every DEGAS2 variable from a CsgGeometry (L3).

#include "omegah2csg/csg_geometry.hpp"
#include "omegah2csg/degas2_geometry.hpp"
#include "omegah2csg/options.hpp"

namespace omegah2csg {

/// Requires wall data. Plasma faces (offset_face 0) must be numbered before
/// boundary-layer faces, and each boundary-layer face may touch at most one
/// wall edge; otherwise throws GeometryError.
Degas2Geometry assemble_degas2(const CsgGeometry &g, const Degas2Options &o);

} // namespace omegah2csg

#endif
