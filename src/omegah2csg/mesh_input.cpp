#include "omegah2csg/mesh_input.hpp"
#include "omegah2csg/errors.hpp"

#include <Omega_h_file.hpp>

#include <filesystem>
#include <fstream>

namespace omegah2csg {

namespace {
int read_int_file(const std::filesystem::path &file) {
  std::ifstream in(file);
  int value = -1;
  if (!(in >> value))
    throw InputError("cannot read " + file.string());
  return value;
}

void require_tag(Omega_h::Mesh &mesh, int dim, const std::string &name) {
  if (!mesh.has_tag(dim, name) ||
      mesh.get_tagbase(dim, name)->type() != OMEGA_H_I32)
    throw InputError("mesh is missing the I32 tag '" + name +
                     "' on dimension " + std::to_string(dim));
}
} // namespace

void check_mesh_path(const std::string &path) {
  const std::filesystem::path p(path);
  if (!std::filesystem::is_directory(p))
    throw InputError("mesh directory not found: " + path);
  if (const int n = read_int_file(p / "nparts"); n != 1)
    throw InputError(path + " has " + std::to_string(n) +
                     " parts; only single-part meshes are supported");
  if (const int v = read_int_file(p / "version");
      v > Omega_h::binary::latest_version)
    throw InputError(path + " has format version " + std::to_string(v) +
                     ", newer than this omega-h supports");
}

Omega_h::Mesh read_mesh(Omega_h::Library &lib, const std::string &path) {
  check_mesh_path(path);
  return Omega_h::binary::read(path, lib.self());
}

void check_required_tags(Omega_h::Mesh &mesh, bool need_wall_tags) {
  if (mesh.dim() != 2)
    throw InputError("mesh must be 2D, got dimension " +
                     std::to_string(mesh.dim()));
  if (need_wall_tags) {
    require_tag(mesh, Omega_h::FACE, "offset_face");
    require_tag(mesh, Omega_h::VERT, "isOnWall");
  }
}

void check_single_rank(Omega_h::Library &lib) {
  if (lib.world()->size() > 1)
    throw InputError("omegah2csg is serial; run it on one MPI rank");
}

} // namespace omegah2csg
