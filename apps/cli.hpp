#ifndef OMEGAH2CSG_APPS_CLI_HPP
#define OMEGAH2CSG_APPS_CLI_HPP
/// @file cli.hpp
/// @brief Shared driver for the native executables.
///
/// Exit codes: 0 success, 1 usage error, 2 input or geometry error,
/// 3 other (I/O) error. `--kokkos-*` and `--osh-*` flags are handled by
/// Omega_h::Library.

#include <Omega_h_library.hpp>

#include <string>

namespace omegah2csg {

template <class Options>
int run_cli(int argc, char **argv, const std::string &about,
            void (*convert)(Omega_h::Library &, const std::string &,
                            const Options &),
            bool need_wall_tags);

} // namespace omegah2csg

#endif
