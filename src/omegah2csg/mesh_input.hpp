#ifndef OMEGAH2CSG_MESH_INPUT_HPP
#define OMEGAH2CSG_MESH_INPUT_HPP
/// @file mesh_input.hpp
/// @brief Validation before omega-h is called (L1). An `.osh` mesh is a
/// directory with `nparts`, `version` and one binary file per part.

#include <Omega_h_library.hpp>
#include <Omega_h_mesh.hpp>

#include <string>

namespace omegah2csg {

/// Throws InputError unless `path` is a single-part `.osh` directory with a
/// supported format version.
void check_mesh_path(const std::string &path);

/// check_mesh_path(), then read the mesh on the self communicator.
Omega_h::Mesh read_mesh(Omega_h::Library &lib, const std::string &path);

/// Throws InputError unless the mesh is 2D and, when `need_wall_tags`, has
/// the I32 tags `offset_face` (faces: 0 plasma, 1 boundary layer) and
/// `isOnWall` (vertices: 1 on the plasma/boundary-layer interface).
void check_required_tags(Omega_h::Mesh &mesh, bool need_wall_tags);

/// Throws InputError when running on more than one MPI rank.
void check_single_rank(Omega_h::Library &lib);

} // namespace omegah2csg

#endif
