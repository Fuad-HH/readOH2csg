#ifndef OMEGAH2CSG_TRANSFER_HPP
#define OMEGAH2CSG_TRANSFER_HPP
/// @file transfer.hpp
/// @brief The only device -> host copy point for bulk arrays.

#include <Kokkos_Core.hpp>
#include <Omega_h_array.hpp>

#include <vector>

namespace omegah2csg {

/// One deep_copy from `src` into the host buffer `dst` (src.size() items).
template <class T> void copy_to_host(Omega_h::Read<T> src, T *dst) {
  Kokkos::View<T *, typename decltype(src.view())::array_layout,
               Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      host(dst, src.size());
  Kokkos::deep_copy(host, src.view());
}

template <class T> std::vector<T> to_host(Omega_h::Read<T> src) {
  std::vector<T> out(src.size());
  copy_to_host(src, out.data());
  return out;
}

} // namespace omegah2csg

#endif
