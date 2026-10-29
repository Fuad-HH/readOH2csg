#include "bindings.hpp"

#include "omegah2csg/pipeline.hpp"
#include "omegah2csg/runtime_info.hpp"

#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

namespace omegah2csg {

namespace {
template <class T> pybind11::array_t<T> numpy_copy(const std::vector<T> &v) {
  return pybind11::array_t<T>(static_cast<pybind11::ssize_t>(v.size()),
                              v.data());
}

PyCsg wrap(CsgGeometry g, const OpenMCOptions &o) {
  PyCsg s;
  s.cell_material = numpy_copy(read_material_ids(o, g.topology.nfaces));
  s.surface_boundary = numpy_copy(default_surface_boundary(g, o));
  s.g = std::move(g);
  return s;
}

/// Runs `f` on a mesh read from `path` with the GIL released.
template <class F> auto from_path(const std::filesystem::path &path, F f) {
  auto &lib = library();
  pybind11::gil_scoped_release release;
  return f(lib, path.string());
}
} // namespace

void bind_pipeline(pybind11::module_ &m) {
  const auto source = pybind11::arg("source");
  m.def(
      "compute_csg",
      [](const std::filesystem::path &p, const OpenMCOptions &o) {
        validate(o);
        return wrap(from_path(p,
                              [&](Omega_h::Library &lib, const std::string &s) {
                                return compute_csg(lib, s, o);
                              }),
                    o);
      },
      source, pybind11::arg("options") = OpenMCOptions(),
      "Compute CSG surfaces and cells from an .osh path.");
  m.def(
      "compute_csg",
      [](Omega_h::Mesh &mesh, const OpenMCOptions &o) {
        require_same_libomega_h(mesh);
        validate(o);
        return wrap(compute_csg(mesh, o), o);
      },
      source, pybind11::arg("options") = OpenMCOptions(),
      "Compute CSG surfaces and cells from a PyOmega_h mesh.");
  m.def(
      "compute_degas2",
      [](const std::filesystem::path &p, const Degas2Options &o) {
        validate(o);
        return PyDegas2{
            from_path(p, [&](Omega_h::Library &lib, const std::string &s) {
              return compute_degas2(lib, s, o);
            })};
      },
      source, pybind11::arg("options") = Degas2Options(),
      "Compute every DEGAS2 geometry variable from an .osh path.");
  m.def(
      "compute_degas2",
      [](Omega_h::Mesh &mesh, const Degas2Options &o) {
        require_same_libomega_h(mesh);
        validate(o);
        return PyDegas2{compute_degas2(mesh, o)};
      },
      source, pybind11::arg("options") = Degas2Options(),
      "Compute every DEGAS2 geometry variable from a PyOmega_h mesh.");
  m.def(
      "convert_to_openmc",
      [](const std::filesystem::path &p, const OpenMCOptions &o) {
        from_path(p, [&](Omega_h::Library &lib, const std::string &s) {
          convert_to_openmc(lib, s, o);
          return 0;
        });
      },
      source, pybind11::arg("options") = OpenMCOptions(),
      "Mesh path -> options.output (geometry.xml), without Python objects.");
  m.def(
      "convert_to_degas2",
      [](const std::filesystem::path &p, const Degas2Options &o) {
        from_path(p, [&](Omega_h::Library &lib, const std::string &s) {
          convert_to_degas2(lib, s, o);
          return 0;
        });
      },
      source, pybind11::arg("options") = Degas2Options(),
      "Mesh path -> options.output (geometry.nc) and aux files.");
  m.def("initialize", &initialize, pybind11::arg("kokkos_args"),
        "Start Kokkos/MPI with arguments (before any other call).");
  m.def(
      "show_config",
      [] {
        library();
        return runtime_info();
      },
      "Versions, Kokkos execution space and library paths.");
}

} // namespace omegah2csg
