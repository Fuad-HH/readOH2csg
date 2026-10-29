#ifndef OMEGAH2CSG_IO_OPENMC_XML_HPP
#define OMEGAH2CSG_IO_OPENMC_XML_HPP
/// @file openmc_xml.hpp
/// @brief OpenMC geometry.xml writer (L4, host only).
///
/// IDs: surface of edge e = e + 1 + surface_id_offset; the extra plane of a
/// cone edge = nedges + e + 1 + surface_id_offset; cell of face f =
/// f + 1 + cell_id_offset. A cone edge is OpenMC's ZConeOneSided: a z-cone
/// plus a z-plane through its apex; its - side is (-cone & ±plane) and its
/// + side the complement (+cone | ∓plane), with ± given by the up flag. The
/// plane never carries a boundary condition (as in OpenMC).

#include "omegah2csg/host_array.hpp"

namespace omegah2csg {

struct OpenMCTables {
  HostSpan<std::int8_t> surface_type;     ///< nedges, SurfaceType
  HostSpan<double> surface_coeffs;        ///< 6 * nedges
  HostSpan<std::int32_t> face_edges;      ///< 3 * nfaces
  HostSpan<std::int8_t> face_sides;       ///< 3 * nfaces
  HostSpan<std::int8_t> surface_boundary; ///< nedges, BoundaryType
  HostSpan<std::int32_t> cell_material;   ///< nfaces, -1 = void
  int cell_id_offset = 0, surface_id_offset = 0;
};

/// Throws InputError on wrong sizes, out-of-range edges or invalid codes.
void validate(const OpenMCTables &t);

void write_openmc_xml(const OpenMCTables &t, const std::string &path,
                      bool force);

} // namespace omegah2csg

#endif
