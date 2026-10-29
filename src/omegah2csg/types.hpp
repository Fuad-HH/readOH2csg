#ifndef OMEGAH2CSG_TYPES_HPP
#define OMEGAH2CSG_TYPES_HPP
/// @file types.hpp
/// @brief Shared enums and array-layout constants (host-only header; the
/// int8 storage equals Omega_h::I8).

#include <cstdint>

namespace omegah2csg {

/// Surface of revolution produced by one mesh edge (values match 0.0.x).
enum class SurfaceType : std::int8_t { ZPlane = 1, ZCylinder = 2, ZCone = 3 };

/// Columns of `surface_coeffs` (6 per edge). The surface is
/// F(r, z) = A r^2 + B z^2 + C z + D = 0, r = distance from the z axis.
///   plane    z = z0      : {0, 0, 1, -z0, 0, 0}
///   cylinder r = r0      : {1, 0, 0, -r0^2, 0, 1}
///   cone     z = m r + c : {m^2, -1, 2c, -c^2, up(+1/-1), m}
constexpr int kCoeffA = 0, kCoeffB = 1, kCoeffC = 2, kCoeffD = 3,
              kCoeffUpFlag = 4, kCoeffSlope = 5, kNumCoeffs = 6;
constexpr int kFaceEdges = 3, kBBoxComps = 4, kCentroidComps = 2;

/// Boundary condition code stored per surface (OpenMC boundary types).
enum class BoundaryType : std::int8_t {
  Transmission = 0,
  Vacuum = 1,
  Reflective = 2,
  White = 3
};

} // namespace omegah2csg

#endif
