#ifndef OMEGAH2CSG_ERRORS_HPP
#define OMEGAH2CSG_ERRORS_HPP
/// @file errors.hpp
/// @brief Exception types (host only; no Omega_h/Kokkos headers).

#include <stdexcept>
#include <string>

namespace omegah2csg {

/// Bad user input: path, missing tag, partitioned mesh, invalid options.
struct InputError : std::invalid_argument {
  explicit InputError(const std::string &what) : std::invalid_argument(what) {}
};

/// A geometry stage found an entity it cannot handle.
struct GeometryError : std::runtime_error {
  GeometryError(const std::string &stage, long entity, const std::string &why)
      : std::runtime_error(stage + ": " + why + " (first entity " +
                           std::to_string(entity) + ")"),
        stage(stage), entity(entity) {}
  std::string stage;
  long entity;
};

} // namespace omegah2csg

#endif
