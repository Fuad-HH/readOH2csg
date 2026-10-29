#ifndef OMEGAH2CSG_BINDINGS_HPP
#define OMEGAH2CSG_BINDINGS_HPP
/// @file bindings.hpp
/// @brief Shared declarations of the omegah2csg._core module.

#include "omegah2csg/csg_geometry.hpp"
#include "omegah2csg/degas2_geometry.hpp"
#include "omegah2csg/host_array.hpp"
#include "omegah2csg/transfer.hpp"

#include <Omega_h_library.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <map>

namespace omegah2csg {

/// Process-wide Library, created on first use and released by Py_AtExit.
Omega_h::Library &library();
void initialize(const std::vector<std::string> &kokkos_args);

/// Raises TypeError unless `mesh` comes from the same libomega_h as _core.
void require_same_libomega_h(const Omega_h::Mesh &mesh);

/// CsgGeometry plus the host (NumPy) copies Python has requested or set.
struct PyCsg {
  CsgGeometry g;
  std::map<std::string, pybind11::array> cache;
  pybind11::array_t<std::int32_t> cell_material;   ///< -1 = void
  pybind11::array_t<std::int8_t> surface_boundary; ///< BoundaryType
};

/// Degas2Geometry plus host copies requested or replaced from Python.
struct PyDegas2 {
  Degas2Geometry g;
  std::map<std::string, pybind11::array> cache;
};

/// One device -> host copy into a new C-contiguous array (`cols` > 1: 2D).
template <class T>
pybind11::array_t<T> to_numpy(Omega_h::Read<T> a, pybind11::ssize_t cols) {
  const pybind11::ssize_t n = a.size();
  pybind11::array_t<T> out(cols > 1
                               ? std::vector<pybind11::ssize_t>{n / cols, cols}
                               : std::vector<pybind11::ssize_t>{n});
  copy_to_host(a, out.mutable_data());
  return out;
}

template <class T, int Flags>
HostSpan<T> span_of(const pybind11::array_t<T, Flags> &a) {
  return {a.data(), static_cast<std::size_t>(a.size())};
}

void bind_options(pybind11::module_ &m);
void bind_results(pybind11::module_ &m);
void bind_pipeline(pybind11::module_ &m);

} // namespace omegah2csg

#endif
