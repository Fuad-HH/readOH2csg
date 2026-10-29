#ifndef OMEGAH2CSG_IO_DEGAS2_NETCDF_HPP
#define OMEGAH2CSG_IO_DEGAS2_NETCDF_HPP
/// @file degas2_netcdf.hpp
/// @brief DEGAS2 geometry.nc writer driven by degas2_schema (L4, host only).

#include "omegah2csg/host_array.hpp"
#include "omegah2csg/options.hpp"

namespace omegah2csg {

/// `get(name, load)` returns each schema variable (data only when `load`).
/// Dimensions come from the variable shapes. Auto format: CDF-2 (64-bit
/// offset, readable by every DEGAS2 build), CDF-5 when a variable exceeds
/// the CDF-2 limit. Variables are fetched and written one at a time.
void write_degas2_netcdf(
    const std::function<HostVar(const std::string &, bool)> &get,
    const std::string &path, NcFormat format, bool force);

} // namespace omegah2csg

#endif
