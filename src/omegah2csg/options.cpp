#include "omegah2csg/options.hpp"
#include "omegah2csg/errors.hpp"

#include <sstream>

namespace omegah2csg {

namespace {
template <class T> std::string to_text(const T &v) {
  std::ostringstream s;
  s << v;
  return s.str();
}

template <class T>
Flag value(const char *name, const char *short_name, const char *metavar,
           const char *help, T &field) {
  return {name,
          short_name,
          metavar,
          help,
          to_text(field),
          [&field, name](const std::string &arg) {
            std::istringstream s(arg);
            if (!(s >> field) || !s.eof())
              throw InputError(std::string("invalid value '") + arg + "' for " +
                               name);
          }};
}

Flag text(const char *name, const char *short_name, const char *metavar,
          const char *help, std::string &field) {
  return {name, short_name, metavar,
          help, field,      [&field](const std::string &arg) { field = arg; }};
}

Flag toggle(const char *name, const char *short_name, const char *help,
            std::function<void()> on) {
  return {name, short_name, "", help, "", [on](const std::string &) { on(); }};
}

std::vector<Flag> common_flags(CommonOptions &o) {
  return {text("--output", "-o", "FILE", "output file", o.output),
          value("--tol", "", "TOL", "absolute length tolerance in mesh units",
                o.tol),
          toggle("--verbose", "-v", "more progress output (repeatable)",
                 [&o] { ++o.verbosity; }),
          toggle("--quiet", "-q", "errors only", [&o] { o.verbosity = 0; }),
          toggle("--timing", "", "print per-stage wall time",
                 [&o] { o.timing = true; }),
          value("--debug", "", "N", "dump the first N entities of each stage",
                o.debug_entities),
          toggle("--force", "", "overwrite existing output",
                 [&o] { o.force = true; })};
}

void validate_common(const CommonOptions &o) {
  if (!(o.tol > 0.0 && o.tol < 1e-2))
    throw InputError("--tol must be in (0, 1e-2)");
  if (o.output.empty())
    throw InputError("--output must not be empty");
  if (o.debug_entities < 0)
    throw InputError("--debug must be >= 0");
}
} // namespace

std::vector<Flag> flags(OpenMCOptions &o) {
  auto f = common_flags(o);
  f.push_back(text("--material-ids", "", "FILE",
                   "file with one material id per cell (-1 = void)",
                   o.material_ids_file));
  f.push_back(text("--boundary-type", "", "TYPE",
                   "boundary of the outer surfaces: transmission, vacuum, "
                   "reflective or white",
                   o.boundary_type));
  f.push_back(value("--cell-id-offset", "", "N", "added to every cell id",
                    o.cell_id_offset));
  f.push_back(value("--surface-id-offset", "", "N", "added to every surface id",
                    o.surface_id_offset));
  return f;
}

std::vector<Flag> flags(Degas2Options &o) {
  auto f = common_flags(o);
  f.push_back(toggle("--aux-files", "-a",
                     "also write wallfile.txt, plasmafile.txt, sourcefile.txt",
                     [&o] { o.aux_files = true; }));
  f.push_back(text("--aux-dir", "", "DIR",
                   "directory for aux files (default: that of --output)",
                   o.aux_dir));
  f.push_back(value("--target-temperature", "", "K",
                    "target temperature in Kelvin", o.target_temperature_K));
  f.push_back(value("--target-recyc-coef", "", "R",
                    "target recycling coefficient", o.target_recyc_coef));
  f.push_back(
      {"--nc-format", "", "FMT", "auto, cdf2, cdf5 or netcdf4", "auto",
       [&o](const std::string &a) { o.nc_format = parse_nc_format(a); }});
  return f;
}

void validate(const OpenMCOptions &o) {
  validate_common(o);
  parse_boundary_type(o.boundary_type);
}

void validate(const Degas2Options &o) {
  validate_common(o);
  if (!(o.target_temperature_K > 0.0))
    throw InputError("--target-temperature must be > 0");
  if (!(o.target_recyc_coef >= 0.0 && o.target_recyc_coef <= 1.0))
    throw InputError("--target-recyc-coef must be in [0, 1]");
}

BoundaryType parse_boundary_type(const std::string &name) {
  for (auto t : {BoundaryType::Transmission, BoundaryType::Vacuum,
                 BoundaryType::Reflective, BoundaryType::White})
    if (name == boundary_type_name(t))
      return t;
  throw InputError("unknown boundary type '" + name + "'");
}

const char *boundary_type_name(BoundaryType type) {
  static const char *names[] = {"transmission", "vacuum", "reflective",
                                "white"};
  return names[static_cast<int>(type)];
}

NcFormat parse_nc_format(const std::string &name) {
  static const char *names[] = {"auto", "cdf2", "cdf5", "netcdf4"};
  for (int i = 0; i < 4; ++i)
    if (name == names[i])
      return static_cast<NcFormat>(i);
  throw InputError("unknown --nc-format '" + name + "'");
}

} // namespace omegah2csg
