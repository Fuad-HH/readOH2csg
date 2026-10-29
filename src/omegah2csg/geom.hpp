#ifndef OMEGAH2CSG_GEOM_HPP
#define OMEGAH2CSG_GEOM_HPP
/// @file geom.hpp
/// @brief Per-entity geometry math (L0). Header-only, callable on host and
/// device. The mesh lives in the (r, z) half-plane; every edge is revolved
/// about the z axis into a plane, cylinder or cone (see types.hpp), and every
/// triangle becomes the intersection of one side of each of its 3 surfaces.

#include "omegah2csg/types.hpp"

#include <Kokkos_Core.hpp>
#include <Omega_h_vector.hpp>

namespace omegah2csg {

/// Vertical edges become cylinders, horizontal edges planes, others cones.
OMEGA_H_INLINE SurfaceType classify_edge(const Omega_h::Vector<2> &p0,
                                         const Omega_h::Vector<2> &p1,
                                         Omega_h::Real tol) {
  if (Kokkos::abs(p0[0] - p1[0]) < tol)
    return SurfaceType::ZCylinder;
  if (Kokkos::abs(p0[1] - p1[1]) < tol)
    return SurfaceType::ZPlane;
  return SurfaceType::ZCone;
}

OMEGA_H_INLINE Omega_h::Vector<kNumCoeffs>
surface_coeffs(const Omega_h::Vector<2> &p0, const Omega_h::Vector<2> &p1,
               SurfaceType type) {
  if (type == SurfaceType::ZCylinder)
    return {1.0, 0.0, 0.0, -p0[0] * p0[0], 0.0, 1.0};
  if (type == SurfaceType::ZPlane)
    return {0.0, 0.0, 1.0, -p0[1], 0.0, 0.0};
  // z = m r + c; up (+1) when the cone opens towards +z (OpenMC's `up`)
  const Omega_h::Real m = (p1[1] - p0[1]) / (p1[0] - p0[0]);
  const Omega_h::Real c = p0[1] - m * p0[0];
  return {m * m, -1.0, 2.0 * c, -c * c, (p0[1] > c) ? 1.0 : -1.0, m};
}

/// F(r, z) = A r^2 + B z^2 + C z + D
OMEGA_H_INLINE Omega_h::Real
evaluate_surface(const Omega_h::Vector<2> &p,
                 const Omega_h::Vector<kNumCoeffs> &c) {
  return c[kCoeffA] * p[0] * p[0] + c[kCoeffB] * p[1] * p[1] +
         c[kCoeffC] * p[1] + c[kCoeffD];
}

/// +1 above the line z = m r + c, -1 below, 0 on it.
OMEGA_H_INLINE Omega_h::I8 side_of_line(const Omega_h::Vector<2> &p,
                                        Omega_h::Real m, Omega_h::Real c,
                                        Omega_h::Real tol) {
  const Omega_h::Real d = p[1] - (m * p[0] + c);
  return d > tol ? 1 : (d < -tol ? -1 : 0);
}

/// Side (+1/-1) of an edge surface on which triangle (v0, v1, v2) lies.
/// Returns 0 when undecidable (degenerate triangle); callers report it.
OMEGA_H_INLINE Omega_h::I8 face_side(const Omega_h::Vector<2> &v0,
                                     const Omega_h::Vector<2> &v1,
                                     const Omega_h::Vector<2> &v2,
                                     const Omega_h::Vector<kNumCoeffs> &c,
                                     SurfaceType type, Omega_h::Real tol) {
  const Omega_h::Vector<2> v[3] = {v0, v1, v2};
  if (type == SurfaceType::ZCone) {
    // third vertex above/below the generating line, combined with the cone
    // direction, gives the side
    Omega_h::I8 line = 0;
    for (int i = 0; i < 3; ++i) {
      const Omega_h::I8 s =
          side_of_line(v[i], c[kCoeffSlope], c[kCoeffC] / 2.0, tol);
      if (s != 0)
        line = s;
    }
    if (line == 0)
      return 0;
    return (line + static_cast<Omega_h::I8>(c[kCoeffUpFlag]) == 0) ? 1 : -1;
  }
  for (int i = 0; i < 3; ++i) {
    const Omega_h::Real e = evaluate_surface(v[i], c);
    if (Kokkos::abs(e) > tol)
      return e > 0 ? 1 : -1;
  }
  return 0;
}

OMEGA_H_INLINE Omega_h::Real triangle_area(const Omega_h::Vector<2> &a,
                                           const Omega_h::Vector<2> &b,
                                           const Omega_h::Vector<2> &c) {
  return 0.5 * Kokkos::abs(Omega_h::cross(b - a, c - a));
}

OMEGA_H_INLINE Omega_h::Vector<2>
triangle_centroid(const Omega_h::Vector<2> &a, const Omega_h::Vector<2> &b,
                  const Omega_h::Vector<2> &c) {
  return (a + b + c) / 3.0;
}

/// Volume swept by revolving a triangle about z (Pappus): 2 pi r_bar A.
OMEGA_H_INLINE Omega_h::Real toroidal_volume(Omega_h::Real area,
                                             Omega_h::Real centroid_r) {
  return 2.0 * Kokkos::numbers::pi_v<Omega_h::Real> * area * centroid_r;
}

/// {rmin, zmin, rmax, zmax}
OMEGA_H_INLINE Omega_h::Vector<kBBoxComps>
triangle_bbox(const Omega_h::Vector<2> &a, const Omega_h::Vector<2> &b,
              const Omega_h::Vector<2> &c) {
  return {Kokkos::min(a[0], Kokkos::min(b[0], c[0])),
          Kokkos::min(a[1], Kokkos::min(b[1], c[1])),
          Kokkos::max(a[0], Kokkos::max(b[0], c[0])),
          Kokkos::max(a[1], Kokkos::max(b[1], c[1]))};
}

} // namespace omegah2csg

#endif
