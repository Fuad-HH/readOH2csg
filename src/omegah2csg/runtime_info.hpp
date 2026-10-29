#ifndef OMEGAH2CSG_RUNTIME_INFO_HPP
#define OMEGAH2CSG_RUNTIME_INFO_HPP
/// @file runtime_info.hpp
/// @brief Build/runtime configuration for --version -v and show_config().

#include <map>
#include <string>

namespace omegah2csg {

/// Versions, Kokkos execution space and library paths.
std::map<std::string, std::string> runtime_info();

/// Absolute path of the shared library defining `symbol` ("" if unknown).
std::string library_path_of(const void *symbol);

} // namespace omegah2csg

#endif
