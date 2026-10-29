#ifndef OMEGAH2CSG_DEGAS2_GEOMETRY_HPP
#define OMEGAH2CSG_DEGAS2_GEOMETRY_HPP
/// @file degas2_geometry.hpp
/// @brief DEGAS2 geometry: one array per geometry.nc variable.
///
/// DEGAS2 terms: a *zone* is a region with uniform plasma properties (here
/// one per plasma face, one per boundary-layer face next to the wall, plus
/// one catch-all zone); a *sector* is one side of a wall edge where
/// particles are scored; a *stratum* is a source of particles (one per wall
/// point). Large arrays live on the device; constants on the host.

#include "omegah2csg/host_array.hpp"

#include <Omega_h_array.hpp>

#include <map>
#include <variant>

namespace omegah2csg {

struct Degas2Variable {
  std::variant<Omega_h::LOs, Omega_h::Reals, std::vector<std::int32_t>,
               std::vector<double>, std::string>
      data;
  std::vector<std::size_t> shape;
};

struct Degas2Geometry {
  std::map<std::string, Degas2Variable> vars; ///< by DEGAS2 variable name
  // inputs of the aux text files
  Omega_h::Reals node_coords;
  Omega_h::LO nplasma = 0, nwall_points = 0;
};

/// Host copy of a variable (device data copied once when `want_data`).
HostVar host_var(const Degas2Variable &v, bool want_data);

} // namespace omegah2csg

#endif
