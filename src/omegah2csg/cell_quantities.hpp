#ifndef OMEGAH2CSG_CELL_QUANTITIES_HPP
#define OMEGAH2CSG_CELL_QUANTITIES_HPP
/// @file cell_quantities.hpp
/// @brief Per-cell volume, centroid and bounding box (L2).

#include "omegah2csg/topology.hpp"

#include <Omega_h_vector.hpp>

namespace omegah2csg {

struct CellQuantities {
  Omega_h::Reals volume;   ///< nfaces, toroidal volume
  Omega_h::Reals centroid; ///< 2 * nfaces, (r, z)
  Omega_h::Reals bbox;     ///< 4 * nfaces, {rmin, zmin, rmax, zmax}
};

CellQuantities compute_cell_quantities(const Topology &t);

/// {rmin, zmin, rmax, zmax} over all cells.
Omega_h::Vector<4> universal_bounds(Omega_h::Reals bbox);

} // namespace omegah2csg

#endif
