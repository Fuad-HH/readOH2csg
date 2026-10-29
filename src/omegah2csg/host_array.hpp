#ifndef OMEGAH2CSG_HOST_ARRAY_HPP
#define OMEGAH2CSG_HOST_ARRAY_HPP
/// @file host_array.hpp
/// @brief Host-side array types for writers (no Omega_h/Kokkos headers).

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace omegah2csg {

/// Non-owning view of contiguous host data.
template <class T> struct HostSpan {
  const T *data = nullptr;
  std::size_t size = 0;
  const T &operator[](std::size_t i) const { return data[i]; }
};

enum class DType { I32, F64, Char };

/// A typed, shaped host array; `data` keeps its owner alive.
struct HostVar {
  DType type = DType::I32;
  std::vector<std::size_t> shape;
  std::shared_ptr<const void> data;
  std::size_t size() const {
    return std::accumulate(shape.begin(), shape.end(), std::size_t{1},
                           std::multiplies<>());
  }
};

} // namespace omegah2csg

#endif
