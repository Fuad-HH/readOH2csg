#include "omegah2csg/io/openmc_xml.hpp"
#include "omegah2csg/errors.hpp"
#include "omegah2csg/io/atomic_file.hpp"
#include "omegah2csg/options.hpp"
#include "omegah2csg/types.hpp"

#include <pugixml.hpp>

#include <charconv>
#include <cmath>

namespace omegah2csg {

namespace {
std::string num(double v) {
  char buf[32];
  return std::string(buf, std::to_chars(buf, buf + sizeof buf, v).ptr);
}

std::string half(long id, int sign) {
  return (sign < 0 ? "-" : "") + std::to_string(id);
}

void add_surface(pugi::xml_node root, long id, const char *type,
                 const std::string &coeffs, std::int8_t boundary) {
  auto s = root.append_child("surface");
  s.append_attribute("id") = id;
  s.append_attribute("type") = type;
  s.append_attribute("coeffs") = coeffs.c_str();
  if (boundary != static_cast<std::int8_t>(BoundaryType::Transmission))
    s.append_attribute("boundary") =
        boundary_type_name(static_cast<BoundaryType>(boundary));
}
} // namespace

void validate(const OpenMCTables &t) {
  const std::size_t ne = t.surface_type.size, nf = t.face_edges.size / 3;
  if (t.surface_coeffs.size != kNumCoeffs * ne || t.face_edges.size != 3 * nf ||
      t.face_sides.size != 3 * nf || t.surface_boundary.size != ne ||
      t.cell_material.size != nf)
    throw InputError("OpenMC tables have inconsistent sizes");
  for (std::size_t i = 0; i < ne; ++i)
    if (t.surface_type[i] < 1 || t.surface_type[i] > 3 ||
        t.surface_boundary[i] < 0 || t.surface_boundary[i] > 3)
      throw InputError("invalid surface type/boundary at surface " +
                       std::to_string(i));
  for (std::size_t i = 0; i < 3 * nf; ++i)
    if (t.face_edges[i] < 0 || std::size_t(t.face_edges[i]) >= ne ||
        (t.face_sides[i] != 1 && t.face_sides[i] != -1))
      throw InputError("invalid face edge/side at face " +
                       std::to_string(i / 3));
}

void write_openmc_xml(const OpenMCTables &t, const std::string &path,
                      bool force) {
  validate(t);
  const long ne = static_cast<long>(t.surface_type.size);
  const long nf = static_cast<long>(t.face_edges.size / 3);
  if (ne * 2 + t.surface_id_offset >= INT32_MAX ||
      nf + t.cell_id_offset >= INT32_MAX)
    throw InputError("OpenMC ids exceed 32-bit range");
  pugi::xml_document doc;
  auto decl = doc.append_child(pugi::node_declaration);
  decl.append_attribute("version") = "1.0";
  decl.append_attribute("encoding") = "utf-8";
  auto root = doc.append_child("geometry");

  const auto surface_id = [&](long e) { return e + 1 + t.surface_id_offset; };
  const auto plane_id = [&](long e) {
    return ne + e + 1 + t.surface_id_offset;
  };
  const auto region_of = [&](long e, int side) -> std::string {
    if (static_cast<SurfaceType>(t.surface_type[e]) != SurfaceType::ZCone)
      return half(surface_id(e), side);
    const int up = t.surface_coeffs[kNumCoeffs * e + kCoeffUpFlag] > 0 ? 1 : -1;
    const auto cone = surface_id(e), plane = plane_id(e);
    if (side < 0)
      return "(" + half(cone, -1) + " " + half(plane, up) + ")";
    return "(" + half(cone, 1) + " | " + half(plane, -up) + ")";
  };

  for (long f = 0; f < nf; ++f) {
    std::string region;
    for (int j = 0; j < 3; ++j)
      region += (j ? " " : "") +
                region_of(t.face_edges[3 * f + j], t.face_sides[3 * f + j]);
    auto c = root.append_child("cell");
    c.append_attribute("id") = f + 1 + t.cell_id_offset;
    const auto m = t.cell_material[f];
    c.append_attribute("material") = m < 0 ? "void" : std::to_string(m).c_str();
    c.append_attribute("region") = region.c_str();
  }
  for (long e = 0; e < ne; ++e) {
    const double *c = &t.surface_coeffs[kNumCoeffs * e];
    const auto b = t.surface_boundary[e];
    switch (static_cast<SurfaceType>(t.surface_type[e])) {
    case SurfaceType::ZPlane:
      add_surface(root, surface_id(e), "z-plane", num(-c[kCoeffD]), b);
      break;
    case SurfaceType::ZCylinder:
      add_surface(root, surface_id(e), "z-cylinder",
                  "0.0 0.0 " + num(std::sqrt(-c[kCoeffD])), b);
      break;
    case SurfaceType::ZCone:
      add_surface(root, surface_id(e), "z-cone",
                  "0.0 0.0 " + num(c[kCoeffC] / 2.0) + " " +
                      num(1.0 / std::abs(c[kCoeffA])),
                  b);
      // like OpenMC, the disambiguation plane never carries the boundary
      add_surface(root, plane_id(e), "z-plane", num(c[kCoeffC] / 2.0),
                  static_cast<std::int8_t>(BoundaryType::Transmission));
      break;
    }
  }
  AtomicOutput out(path, force);
  if (!doc.save_file(out.tmp().c_str(), "  "))
    throw std::runtime_error("cannot write " + out.tmp());
  out.commit();
}

} // namespace omegah2csg
