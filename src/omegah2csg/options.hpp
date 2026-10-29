#ifndef OMEGAH2CSG_OPTIONS_HPP
#define OMEGAH2CSG_OPTIONS_HPP
/// @file options.hpp
/// @brief Converter options: the single source of defaults and CLI flags.

#include "omegah2csg/types.hpp"

#include <functional>
#include <string>
#include <vector>

namespace omegah2csg {

enum class NcFormat { Auto, Cdf2, Cdf5, NetCdf4 };

struct CommonOptions {
  double tol = 1e-10; ///< absolute length tolerance, in mesh units
  int verbosity = 1;  ///< 0 quiet, 1 default, 2+ verbose
  bool timing = false;
  int debug_entities = 0;
  bool force = false;
  std::string output;
};

struct OpenMCOptions : CommonOptions {
  OpenMCOptions() { output = "geometry.xml"; }
  std::string material_ids_file; ///< whitespace-separated id per cell
  std::string boundary_type = "reflective";
  int cell_id_offset = 0;
  int surface_id_offset = 0;
};

struct Degas2Options : CommonOptions {
  Degas2Options() { output = "geometry.nc"; }
  bool aux_files = false;
  std::string aux_dir; ///< empty: directory of `output`
  double target_temperature_K = 300.0;
  double target_recyc_coef = 0.5;
  NcFormat nc_format = NcFormat::Auto;
};

/// One command-line flag bound to an options field.
struct Flag {
  std::string name, short_name, metavar, help, default_value;
  std::function<void(const std::string &)> set; ///< arg is "" for switches
};

std::vector<Flag> flags(OpenMCOptions &o);
std::vector<Flag> flags(Degas2Options &o);

void validate(const OpenMCOptions &o);
void validate(const Degas2Options &o);

BoundaryType parse_boundary_type(const std::string &name);
const char *boundary_type_name(BoundaryType type);
NcFormat parse_nc_format(const std::string &name);

} // namespace omegah2csg

#endif
