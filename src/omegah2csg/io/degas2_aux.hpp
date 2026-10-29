#ifndef OMEGAH2CSG_IO_DEGAS2_AUX_HPP
#define OMEGAH2CSG_IO_DEGAS2_AUX_HPP
/// @file degas2_aux.hpp
/// @brief DEGAS2 auxiliary text inputs (L4, host only): wallfile.txt (mesh
/// vertex coordinates), plasmafile.txt (placeholder plasma per plasma zone)
/// and sourcefile.txt (placeholder source per wall segment).

#include "omegah2csg/host_array.hpp"

namespace omegah2csg {

void write_degas2_aux(const std::string &dir, HostSpan<double> node_coords,
                      int nplasma, int nwall_points, bool force);

} // namespace omegah2csg

#endif
