#include "bindings.hpp"

#include "omegah2csg/degas2_schema.hpp"
#include "omegah2csg/errors.hpp"
#include "omegah2csg/io/openmc_xml.hpp"
#include "omegah2csg/pipeline.hpp"
#include "omegah2csg/types.hpp"
#include "omegah2csg/wall.hpp"

#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include <cstring>

namespace omegah2csg {

namespace {
constexpr int kContiguous =
    pybind11::array::c_style | pybind11::array::forcecast;

/// Host copy of a CsgGeometry array: cached (possibly user-edited) or new.
template <class T>
pybind11::array_t<T, kContiguous> cached(PyCsg &s, const char *name,
                                         Omega_h::Read<T> CsgGeometry::*member,
                                         int cols) {
  auto it = s.cache.find(name);
  if (it == s.cache.end())
    it = s.cache.emplace(name, to_numpy(s.g.*member, cols)).first;
  return it->second.cast<pybind11::array_t<T, kContiguous>>();
}

template <class T>
void array_property(pybind11::class_<PyCsg> &c, const char *name,
                    Omega_h::Read<T> CsgGeometry::*member, int cols) {
  c.def_property(
      name,
      [name, member, cols](PyCsg &s) {
        return pybind11::array(s.cache[name] = cached(s, name, member, cols));
      },
      [name](PyCsg &s, const pybind11::array_t<T, kContiguous> &a) {
        s.cache[name] = a;
      });
}

pybind11::dtype dtype_of(DType t) {
  return t == DType::I32   ? pybind11::dtype::of<std::int32_t>()
         : t == DType::F64 ? pybind11::dtype::of<double>()
                           : pybind11::dtype("S1");
}

const Degas2VarSpec &spec_of(const std::string &name) {
  for (const auto &v : degas2_vars())
    if (name == v.name)
      return v;
  throw pybind11::key_error("unknown DEGAS2 variable '" + name + "'");
}

pybind11::array degas2_item(PyDegas2 &s, const std::string &name) {
  if (auto it = s.cache.find(name); it != s.cache.end())
    return it->second;
  const auto v = host_var(s.g.vars.at(spec_of(name).name), true);
  pybind11::array a(dtype_of(v.type), std::vector<pybind11::ssize_t>(
                                          v.shape.begin(), v.shape.end()));
  std::memcpy(a.mutable_data(), v.data.get(), a.nbytes());
  return s.cache[name] = a;
}

/// Variable source for the writer: Python copies first, then device data.
HostVar degas2_var(PyDegas2 &s, const std::string &name, bool load) {
  const auto it = s.cache.find(name);
  if (it == s.cache.end())
    return host_var(s.g.vars.at(name), load);
  const pybind11::array a = it->second;
  HostVar v;
  v.type = spec_of(name).type;
  v.shape.assign(a.shape(), a.shape() + a.ndim());
  if (load) // keep the NumPy array alive while the writer uses it
    v.data = std::shared_ptr<const void>(a.data(), [a](const void *) {});
  return v;
}

void write_xml(PyCsg &s, const std::filesystem::path &path,
               const OpenMCOptions &o) {
  const auto type = cached(s, "surface_type", &CsgGeometry::surface_type, 1);
  const auto coeffs =
      cached(s, "surface_coeffs", &CsgGeometry::surface_coeffs, kNumCoeffs);
  const auto edges = cached(s, "face_edges", &CsgGeometry::face_edges, 3);
  const auto sides = cached(s, "face_sides", &CsgGeometry::face_sides, 3);
  const auto boundary =
      s.surface_boundary.cast<pybind11::array_t<std::int8_t, kContiguous>>();
  const auto material =
      s.cell_material.cast<pybind11::array_t<std::int32_t, kContiguous>>();
  OpenMCTables t;
  t.surface_type = span_of(type);
  t.surface_coeffs = span_of(coeffs);
  t.face_edges = span_of(edges);
  t.face_sides = span_of(sides);
  t.surface_boundary = span_of(boundary);
  t.cell_material = span_of(material);
  t.cell_id_offset = o.cell_id_offset;
  t.surface_id_offset = o.surface_id_offset;
  write_openmc_xml(t, path.string(), o.force);
}

void write_nc(PyDegas2 &s, const std::filesystem::path &path, Degas2Options o) {
  o.output = path.string();
  validate(o);
  write_degas2(s.g, o, [&s](const std::string &name, bool load) {
    return degas2_var(s, name, load);
  });
}
} // namespace

void bind_results(pybind11::module_ &m) {
  pybind11::class_<PyCsg> csg(m, "CsgGeometry", R"(
Result of compute_csg(): one surface per mesh edge, one cell per mesh face.
Array properties are NumPy copies made on first access; edit them in place
or assign new arrays. surface_coeffs columns: A, B, C, D, up flag, slope of
F(r, z) = A r^2 + B z^2 + C z + D.)");
  array_property(csg, "surface_type", &CsgGeometry::surface_type, 1);
  array_property(csg, "surface_coeffs", &CsgGeometry::surface_coeffs,
                 kNumCoeffs);
  array_property(csg, "face_edges", &CsgGeometry::face_edges, 3);
  array_property(csg, "face_sides", &CsgGeometry::face_sides, 3);
  array_property(csg, "boundary_edges", &CsgGeometry::boundary_edges, 1);
  array_property(csg, "boundary_sides", &CsgGeometry::boundary_sides, 1);
  array_property(csg, "cell_volume", &CsgGeometry::cell_volume, 1);
  array_property(csg, "cell_centroid", &CsgGeometry::cell_centroid, 2);
  array_property(csg, "cell_bbox", &CsgGeometry::cell_bbox, 4);
  array_property(csg, "edge_faces_sorted", &CsgGeometry::edge_faces_sorted, 2);
  array_property(csg, "wall_edges", &CsgGeometry::wall_edges, 1);
  array_property(csg, "wall_faces", &CsgGeometry::wall_faces, 2);
  array_property(csg, "face_region_tag", &CsgGeometry::face_region_tag, 1);
  csg.def_readwrite("cell_material", &PyCsg::cell_material,
                    "material id per cell (-1 = void)")
      .def_readwrite("surface_boundary", &PyCsg::surface_boundary,
                     "BoundaryType code per surface")
      .def_property_readonly("nverts",
                             [](const PyCsg &s) { return s.g.topology.nverts; })
      .def_property_readonly("nedges",
                             [](const PyCsg &s) { return s.g.topology.nedges; })
      .def_property_readonly("nfaces",
                             [](const PyCsg &s) { return s.g.topology.nfaces; })
      .def_property_readonly(
          "node_coordinates",
          [](const PyCsg &s) { return to_numpy(s.g.topology.coords, 2); })
      .def_property_readonly("interior_edge_count", [](const PyCsg &s) {
        return count_interior_edges(s.g.topology);
      });

  pybind11::class_<PyDegas2>(m, "Degas2Geometry", R"(
Result of compute_degas2(): one array per DEGAS2 geometry.nc variable,
accessed by name (geom["zone_volume"]). Arrays are NumPy copies made on
first access; edit them in place or assign new arrays (shapes are checked
when written).)")
      .def("__getitem__", &degas2_item)
      .def("__setitem__",
           [](PyDegas2 &s, const std::string &name, pybind11::object value) {
             s.cache[name] =
                 pybind11::array(pybind11::module_::import("numpy").attr(
                     "ascontiguousarray")(value, dtype_of(spec_of(name).type)));
           })
      .def("keys", [](const PyDegas2 &) {
        std::vector<std::string> names;
        for (const auto &v : degas2_vars())
          names.push_back(v.name);
        return names;
      });

  m.def("write_openmc_xml", &write_xml, pybind11::arg("geometry"),
        pybind11::arg("path"), pybind11::arg("options") = OpenMCOptions(),
        "Write geometry.xml from a (possibly edited) CsgGeometry.");
  m.def("write_degas2", &write_nc, pybind11::arg("geometry"),
        pybind11::arg("path"), pybind11::arg("options") = Degas2Options(),
        "Write geometry.nc (+ aux files) from a (possibly edited) "
        "Degas2Geometry.");
}

} // namespace omegah2csg
