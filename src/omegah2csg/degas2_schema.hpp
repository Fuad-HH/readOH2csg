#ifndef OMEGAH2CSG_DEGAS2_SCHEMA_HPP
#define OMEGAH2CSG_DEGAS2_SCHEMA_HPP
/// @file degas2_schema.hpp
/// @brief Every DEGAS2 geometry.nc dimension and variable (host only).
/// Names, types and dimension order match DEGAS2's reader (netcdf.hweb).

#include "omegah2csg/host_array.hpp"

#include <map>

namespace omegah2csg {

struct Degas2Dim {
  const char *name;
  std::size_t fixed; ///< 0: size comes from the variables using it
};

struct Degas2VarSpec {
  const char *name;
  DType type;
  std::vector<const char *> dims; ///< empty for scalars
};

const std::vector<Degas2Dim> &degas2_dims();
const std::vector<Degas2VarSpec> &degas2_vars();

/// Dimension sizes inferred from variable shapes. Throws InputError when a
/// shape has the wrong rank, disagrees with another variable sharing a
/// dimension, or with a fixed dimension.
std::map<std::string, std::size_t> infer_degas2_dims(
    const std::function<std::vector<std::size_t>(const std::string &)>
        &shape_of);

} // namespace omegah2csg

#endif
