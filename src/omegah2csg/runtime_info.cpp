#include "omegah2csg/runtime_info.hpp"

#include <Omega_h_config.h>
#include <Omega_h_fence.hpp>
#include <Omega_h_kokkos.hpp>

#include <dlfcn.h>

#include <filesystem>

namespace omegah2csg {

std::string library_path_of(const void *symbol) {
  Dl_info info;
  if (!dladdr(symbol, &info) || !info.dli_fname)
    return "";
  std::error_code ec;
  const auto p = std::filesystem::canonical(info.dli_fname, ec);
  return ec ? info.dli_fname : p.string();
}

std::map<std::string, std::string> runtime_info() {
  return {
      {"omegah2csg", OMEGAH2CSG_VERSION},
      {"omega_h", OMEGA_H_SEMVER},
      {"kokkos", std::to_string(KOKKOS_VERSION)},
      {"execution_space", Omega_h::ExecSpace::name()},
      {"concurrency", std::to_string(Omega_h::ExecSpace().concurrency())},
#ifdef OMEGA_H_USE_MPI
      {"mpi", "on"},
#else
      {"mpi", "off"},
#endif
      {"libomega_h",
       library_path_of(reinterpret_cast<const void *>(&Omega_h::fence))},
      {"libkokkoscore", library_path_of(reinterpret_cast<const void *>(
                            &Kokkos::is_initialized))},
  };
}

} // namespace omegah2csg
