#include "omegah2csg/pipeline.hpp"
#include "omegah2csg/cell_quantities.hpp"
#include "omegah2csg/csg.hpp"
#include "omegah2csg/degas2_assemble.hpp"
#include "omegah2csg/errors.hpp"
#include "omegah2csg/io/degas2_aux.hpp"
#include "omegah2csg/io/degas2_netcdf.hpp"
#include "omegah2csg/mesh_input.hpp"
#include "omegah2csg/profiling.hpp"
#include "omegah2csg/transfer.hpp"
#include "omegah2csg/wall.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>

namespace omegah2csg {

namespace {
template <class T> HostSpan<T> span(const std::vector<T> &v) {
  return {v.data(), v.size()};
}

template <class T> void print(const char *name, Omega_h::Read<T> a, int n) {
  const auto h = to_host(a);
  std::printf("%s:", name);
  for (std::size_t i = 0; i < h.size() && int(i) < n; ++i)
    std::printf(" %g", double(h[i]));
  std::printf("\n");
}
} // namespace

CsgGeometry compute_csg(Omega_h::Mesh &mesh, const CommonOptions &o) {
  check_required_tags(mesh, false);
  CsgGeometry g;
  {
    StageTimer t("topology");
    g.topology = build_topology(mesh);
  }
  const Topology &t = g.topology;
  {
    StageTimer s("surfaces");
    const auto surfaces = compute_surfaces(t, o.tol);
    g.surface_type = surfaces.type;
    g.surface_coeffs = surfaces.coeffs;
    g.face_edges = t.face_edges;
    g.face_sides = compute_face_sides(t, surfaces, o.tol);
  }
  {
    StageTimer s("boundary");
    const auto b = compute_boundary(t, g.face_sides);
    g.boundary_edges = b.edges;
    g.boundary_sides = b.sides;
    g.edge_faces_sorted = sort_edge_faces(t, g.face_sides);
  }
  {
    StageTimer s("cell_quantities");
    const auto q = compute_cell_quantities(t);
    g.cell_volume = q.volume;
    g.cell_centroid = q.centroid;
    g.cell_bbox = q.bbox;
  }
  if (t.has_wall_tags) {
    StageTimer s("wall");
    const auto w = compute_wall(t);
    g.wall_edges = w.edges;
    g.wall_faces = w.faces;
    g.face_region_tag = t.offset_face;
  }
  if (o.debug_entities > 0)
    dump_csg(g, o.debug_entities);
  return g;
}

CsgGeometry compute_csg(Omega_h::Library &lib, const std::string &path,
                        const CommonOptions &o) {
  Omega_h::Mesh mesh = [&] {
    StageTimer t("read");
    return read_mesh(lib, path);
  }();
  return compute_csg(mesh, o);
}

Degas2Geometry compute_degas2(Omega_h::Mesh &mesh, const Degas2Options &o) {
  check_required_tags(mesh, true);
  const auto g = compute_csg(mesh, o);
  StageTimer t("degas2_assemble");
  return assemble_degas2(g, o);
}

Degas2Geometry compute_degas2(Omega_h::Library &lib, const std::string &path,
                              const Degas2Options &o) {
  Omega_h::Mesh mesh = [&] {
    StageTimer t("read");
    return read_mesh(lib, path);
  }();
  return compute_degas2(mesh, o);
}

std::vector<std::int32_t> read_material_ids(const OpenMCOptions &o,
                                            Omega_h::LO nfaces) {
  std::vector<std::int32_t> ids(nfaces, -1);
  if (o.material_ids_file.empty())
    return ids;
  std::ifstream in(o.material_ids_file);
  if (!in)
    throw InputError("cannot read " + o.material_ids_file);
  for (auto &id : ids)
    if (!(in >> id))
      throw InputError(o.material_ids_file + " must contain " +
                       std::to_string(nfaces) + " material ids");
  return ids;
}

std::vector<std::int8_t> default_surface_boundary(const CsgGeometry &g,
                                                  const OpenMCOptions &o) {
  std::vector<std::int8_t> b(g.topology.nedges, 0);
  const auto type =
      static_cast<std::int8_t>(parse_boundary_type(o.boundary_type));
  for (const auto e : to_host(g.boundary_edges))
    b[e] = type;
  return b;
}

void write_openmc_xml(const CsgGeometry &g, const OpenMCOptions &o) {
  StageTimer t("write_openmc_xml");
  const auto type = to_host(g.surface_type);
  const auto coeffs = to_host(g.surface_coeffs);
  const auto edges = to_host(g.face_edges);
  const auto sides = to_host(g.face_sides);
  const auto boundary = default_surface_boundary(g, o);
  const auto materials = read_material_ids(o, g.topology.nfaces);
  OpenMCTables tables;
  tables.surface_type = span(type);
  tables.surface_coeffs = span(coeffs);
  tables.face_edges = span(edges);
  tables.face_sides = span(sides);
  tables.surface_boundary = span(boundary);
  tables.cell_material = span(materials);
  tables.cell_id_offset = o.cell_id_offset;
  tables.surface_id_offset = o.surface_id_offset;
  write_openmc_xml(tables, o.output, o.force);
}

void write_degas2(
    const Degas2Geometry &g, const Degas2Options &o,
    const std::function<HostVar(const std::string &, bool)> &get) {
  StageTimer t("write_degas2");
  const auto from_geometry = [&](const std::string &name, bool load) {
    const auto it = g.vars.find(name);
    if (it == g.vars.end())
      throw InputError("DEGAS2 variable '" + name + "' is missing");
    return host_var(it->second, load);
  };
  if (get)
    write_degas2_netcdf(get, o.output, o.nc_format, o.force);
  else
    write_degas2_netcdf(from_geometry, o.output, o.nc_format, o.force);
  if (o.aux_files) {
    const auto xy = to_host(g.node_coords);
    const auto dir =
        o.aux_dir.empty()
            ? std::filesystem::absolute(o.output).parent_path().string()
            : o.aux_dir;
    write_degas2_aux(dir, span(xy), g.nplasma, g.nwall_points, o.force);
  }
}

void convert_to_openmc(Omega_h::Library &lib, const std::string &path,
                       const OpenMCOptions &o) {
  validate(o);
  write_openmc_xml(compute_csg(lib, path, o), o);
}

void convert_to_degas2(Omega_h::Library &lib, const std::string &path,
                       const Degas2Options &o) {
  validate(o);
  write_degas2(compute_degas2(lib, path, o), o);
}

void dump_csg(const CsgGeometry &g, int n) {
  print("surface_type", g.surface_type, n);
  print("surface_coeffs", g.surface_coeffs, kNumCoeffs * n);
  print("face_edges", g.face_edges, 3 * n);
  print("face_sides", g.face_sides, 3 * n);
  print("boundary_edges", g.boundary_edges, n);
  print("cell_volume", g.cell_volume, n);
  print("wall_edges", g.wall_edges, n);
}

} // namespace omegah2csg
