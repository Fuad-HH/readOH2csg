#ifndef OMEGAH2CSG_PIPELINE_HPP
#define OMEGAH2CSG_PIPELINE_HPP
/// @file pipeline.hpp
/// @brief The only place that chains stages (L5).

#include "omegah2csg/csg_geometry.hpp"
#include "omegah2csg/degas2_geometry.hpp"
#include "omegah2csg/io/openmc_xml.hpp"
#include "omegah2csg/options.hpp"

#include <Omega_h_library.hpp>

namespace omegah2csg {

CsgGeometry compute_csg(Omega_h::Mesh &mesh, const CommonOptions &o);
CsgGeometry compute_csg(Omega_h::Library &lib, const std::string &path,
                        const CommonOptions &o);

Degas2Geometry compute_degas2(Omega_h::Mesh &mesh, const Degas2Options &o);
Degas2Geometry compute_degas2(Omega_h::Library &lib, const std::string &path,
                              const Degas2Options &o);

/// Per-cell material ids: -1 (void) or read from o.material_ids_file.
std::vector<std::int32_t> read_material_ids(const OpenMCOptions &o,
                                            Omega_h::LO nfaces);

/// Per-surface boundary codes: o.boundary_type on mesh-boundary edges.
std::vector<std::int8_t> default_surface_boundary(const CsgGeometry &g,
                                                  const OpenMCOptions &o);

void write_openmc_xml(const CsgGeometry &g, const OpenMCOptions &o);

/// Writes geometry.nc (+ aux files). `get` overrides where variables come
/// from (e.g. arrays edited in Python); by default they come from `g`.
void write_degas2(
    const Degas2Geometry &g, const Degas2Options &o,
    const std::function<HostVar(const std::string &, bool)> &get = {});

void convert_to_openmc(Omega_h::Library &lib, const std::string &path,
                       const OpenMCOptions &o);
void convert_to_degas2(Omega_h::Library &lib, const std::string &path,
                       const Degas2Options &o);

/// Prints the first `n` entries of each array (debug output on the host).
void dump_csg(const CsgGeometry &g, int n);

} // namespace omegah2csg

#endif
