#ifndef OMEGAH2CSG_DEVICE_ERRORS_HPP
#define OMEGAH2CSG_DEVICE_ERRORS_HPP
/// @file device_errors.hpp
/// @brief Error reporting from kernels without aborting the process.

#include "omegah2csg/errors.hpp"

#include "omegah2csg/transfer.hpp"

#include <limits>

namespace omegah2csg {

/// Kernels call report(entity); the host calls throw_if_any() after the
/// stage. Counts violations and keeps the smallest failing entity id.
struct DeviceErrors {
  // `=` (not braces): Write{...} would pick the initializer_list constructor
  Omega_h::Write<Omega_h::LO> count = Omega_h::Write<Omega_h::LO>(1, 0);
  Omega_h::Write<Omega_h::LO> first =
      Omega_h::Write<Omega_h::LO>(1, std::numeric_limits<Omega_h::LO>::max());

  OMEGA_H_DEVICE void report(Omega_h::LO entity) const {
    Kokkos::atomic_add(&count[0], 1);
    Kokkos::atomic_min(&first[0], entity);
  }

  void throw_if_any(const std::string &stage, const std::string &why) const {
    const auto n = to_host(Omega_h::Read<Omega_h::LO>(count))[0];
    if (n == 0)
      return;
    throw GeometryError(stage, to_host(Omega_h::Read<Omega_h::LO>(first))[0],
                        why + " [" + std::to_string(n) + " entities]");
  }
};

} // namespace omegah2csg

#endif
