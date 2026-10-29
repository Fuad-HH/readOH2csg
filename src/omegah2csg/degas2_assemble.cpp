#include "omegah2csg/degas2_assemble.hpp"
#include "omegah2csg/cell_quantities.hpp"
#include "omegah2csg/errors.hpp"
#include "omegah2csg/types.hpp"

#include <Omega_h_array_ops.hpp>
#include <Omega_h_for.hpp>
#include <Omega_h_int_scan.hpp>

namespace omegah2csg {

namespace {
constexpr Omega_h::LO kIntUnused = 2000000000;
constexpr double kDblUnused = 2.0e30;
constexpr double kBoltzmann = 1.380649e-23; // J/K
constexpr double kTwoPi = 6.28318530717959; // DEGAS2's value

/// Device array of n values f(i).
template <class T, class F> Omega_h::Read<T> fill(Omega_h::LO n, F f) {
  Omega_h::Write<T> w(n);
  auto k = OMEGA_H_LAMBDA(Omega_h::LO i) { w[i] = f(i); };
  Omega_h::parallel_for(n, k, "omegah2csg::degas2::assemble");
  return w;
}

template <class T>
void put(Degas2Geometry &g, const char *name, T data,
         std::vector<std::size_t> shape) {
  g.vars[name] = Degas2Variable{std::move(data), std::move(shape)};
}

void scalar(Degas2Geometry &g, const char *name, std::int32_t v) {
  put(g, name, std::vector<std::int32_t>{v}, {});
}

/// Rows of fixed width, space padded (netCDF char arrays).
std::string rows(std::initializer_list<const char *> text, std::size_t width) {
  std::string out;
  for (const char *t : text)
    out += std::string(t).append(width - std::string(t).size(), ' ');
  return out;
}

std::size_t sz(Omega_h::LO n) { return static_cast<std::size_t>(n); }
} // namespace

Degas2Geometry assemble_degas2(const CsgGeometry &g, const Degas2Options &o) {
  const Topology &t = g.topology;
  const Omega_h::LO ntri = t.nfaces, nedge = t.nedges;
  const Omega_h::LO nbdry = g.boundary_edges.size();
  const Omega_h::LO nwp = g.wall_edges.size();
  const Omega_h::LO nplasma = ntri - Omega_h::get_sum(t.offset_face);
  const Omega_h::LO nz = nplasma + nwp, nsect = 2 * nwp;
  const auto tag = t.offset_face;
  const auto coords = t.coords;
  const auto ev = t.edge_verts;
  const auto fe = t.face_edges;
  const auto fs = g.face_sides;
  const auto be = g.boundary_edges;
  const auto bs = g.boundary_sides;
  const auto sorted = g.edge_faces_sorted;
  const auto we = g.wall_edges;
  const auto wf = g.wall_faces;
  const auto coeffs = g.surface_coeffs;
  const auto ctr = g.cell_centroid;
  const auto box = g.cell_bbox;
  const auto vol = g.cell_volume;

  // plasma faces first: zone i+1; wall-adjacent boundary-layer faces get
  // the next zones; every other face shares the last zone
  if (Omega_h::get_sum(fill<Omega_h::I8>(
          ntri, OMEGA_H_LAMBDA(Omega_h::LO i) {
            return (i < nplasma) != (tag[i] == 0);
          })))
    throw GeometryError("degas2", nplasma,
                        "plasma faces (offset_face 0) must be numbered first");
  Omega_h::Write<Omega_h::LO> zone_w(ntri);
  auto zone_init = OMEGA_H_LAMBDA(Omega_h::LO i) {
    zone_w[i] = i < nplasma ? i + 1 : nz + 1;
  };
  Omega_h::parallel_for(ntri, zone_init, "omegah2csg::degas2::zone_init");
  auto zone_wall = OMEGA_H_LAMBDA(Omega_h::LO k) {
    zone_w[wf[2 * k + 1]] = nplasma + 1 + k;
  };
  Omega_h::parallel_for(nwp, zone_wall, "omegah2csg::degas2::zone_wall");
  const Omega_h::LOs zone = zone_w;
  if (Omega_h::get_sum(fill<Omega_h::I8>(
          ntri, OMEGA_H_LAMBDA(Omega_h::LO i) {
            return zone[i] > nplasma && zone[i] <= nz;
          })) != nwp)
    throw GeometryError(
        "degas2", nwp, "a boundary-layer face touches more than one wall edge");
  if (Omega_h::get_sum(fill<Omega_h::I8>(
          nwp, OMEGA_H_LAMBDA(Omega_h::LO i) {
            const Omega_h::LO pos = sorted[2 * we[i]];
            return wf[2 * i] != pos && wf[2 * i + 1] != pos;
          })))
    throw GeometryError("degas2", 0, "wall edge with no face on its + side");

  Degas2Geometry d;
  d.node_coords = t.coords;
  d.nplasma = nplasma;
  d.nwall_points = nwp;

  scalar(d, "ncells", ntri);
  scalar(d, "nsurfaces", nedge);
  scalar(d, "nboundaries", nbdry + 3 * ntri);
  scalar(d, "nneighbors", 2 * nedge);
  scalar(d, "ntransforms", 0);
  scalar(d, "geometry_symmetry", 2);
  const auto ub = universal_bounds(box);
  put(d, "universal_cell_min", std::vector<double>{ub[0], 0.0, ub[1]}, {3});
  put(d, "universal_cell_max", std::vector<double>{ub[2], kTwoPi, ub[3]}, {3});
  put(d, "universal_cell_vol", std::vector<double>{Omega_h::get_sum(vol)}, {});

  // ---- cells, boundaries, surfaces ----
  put(d, "cells",
      fill<Omega_h::LO>(
          4 * (ntri + 1),
          OMEGA_H_LAMBDA(Omega_h::LO i) {
            const Omega_h::LO r = i / 4, c = i % 4;
            if (r == 0)
              return c == 0 ? 1 : (c == 3 ? 0 : nbdry);
            return c == 0 ? 1 + nbdry + 3 * (r - 1)
                          : (c == 3 ? zone[r - 1] : 3);
          }),
      {sz(ntri + 1), 4});
  put(d, "boundaries",
      fill<Omega_h::LO>(
          nbdry + 3 * ntri,
          OMEGA_H_LAMBDA(Omega_h::LO i) {
            if (i < nbdry)
              return bs[i] * (be[i] + 1);
            return fs[i - nbdry] * (fe[i - nbdry] + 1);
          }),
      {sz(nbdry + 3 * ntri)});
  put(d, "surfaces",
      fill<Omega_h::LO>(
          4 * nedge,
          OMEGA_H_LAMBDA(Omega_h::LO i) {
            const Omega_h::LO e = i / 4, c = i % 4;
            const Omega_h::LO sides[4] = {
                1 + 2 * e, 2 * e, sorted[2 * e + 1] != -1, sorted[2 * e] != -1};
            return sides[c];
          }),
      {sz(nedge), 2, 2});
  put(d, "surfaces_tx_ind", Omega_h::LOs(4 * nedge, 0), {sz(nedge), 2, 2});
  put(d, "surfaces_tx_mx", std::vector<double>(12, kDblUnused), {1, 4, 3});
  Omega_h::Write<Omega_h::I8> is_wall(nedge, 0);
  auto mark = OMEGA_H_LAMBDA(Omega_h::LO i) { is_wall[we[i]] = 1; };
  Omega_h::parallel_for(nwp, mark, "omegah2csg::degas2::mark_wall");
  const auto wall_before =
      Omega_h::offset_scan(Omega_h::Read<Omega_h::I8>(is_wall));
  put(d, "surface_sectors",
      fill<Omega_h::LO>(
          4 * nedge,
          OMEGA_H_LAMBDA(Omega_h::LO i) {
            const Omega_h::LO e = i / 4, c = i % 4, w = is_wall[e];
            const Omega_h::LO ptr = 1 + 2 * wall_before[e];
            return c == 0 ? ptr : (c == 1 ? ptr + w : w);
          }),
      {sz(nedge), 2, 2});
  put(d, "neighbors",
      fill<Omega_h::LO>(
          2 * nedge + 1,
          OMEGA_H_LAMBDA(Omega_h::LO i) {
            if (i == 2 * nedge)
              return 0;
            return sorted[i] == -1 ? -1 : sorted[i] + 1;
          }),
      {sz(2 * nedge + 1)});
  put(d, "surface_coeffs",
      fill<Omega_h::Real>(
          11 * nedge,
          OMEGA_H_LAMBDA(Omega_h::LO i) {
            // DEGAS2 slot -> our column (-1: zero)
            const int col[11] = {kCoeffD, -1, -1, kCoeffC, kCoeffA,     kCoeffA,
                                 kCoeffB, -1, -1, -1,      kCoeffUpFlag};
            const int c = col[i % 11];
            return c < 0 ? 0.0 : coeffs[kNumCoeffs * (i / 11) + c];
          }),
      {sz(nedge), 11});
  put(d, "surface_points",
      fill<Omega_h::Real>(
          6 * nedge,
          OMEGA_H_LAMBDA(Omega_h::LO i) {
            const Omega_h::LO v = ev[2 * (i / 6) + (i % 6) / 3];
            const int c = i % 3;
            return c == 1 ? 0.0 : coords[2 * v + c / 2];
          }),
      {sz(nedge), 2, 3});

  // ---- zones: rows 0..nz-1 map to a face, row nz is the catch-all zone ----
  const Omega_h::Real rest_volume = Omega_h::get_sum(fill<Omega_h::Real>(
      ntri, OMEGA_H_LAMBDA(Omega_h::LO i) {
        return zone[i] == nz + 1 ? vol[i] : 0.0;
      }));
  auto face_of = OMEGA_H_LAMBDA(Omega_h::LO r) {
    return r < nplasma ? r : (r < nz ? wf[2 * (r - nplasma) + 1] : ntri - 1);
  };
  scalar(d, "zn_num", nz + 1);
  put(d, "zone_type_num",
      std::vector<std::int32_t>{0, nplasma, nz - nplasma + 1, 0}, {4});
  put(d, "zone_type",
      fill<Omega_h::LO>(
          nz + 1,
          OMEGA_H_LAMBDA(Omega_h::LO r) { return r < nplasma ? 2 : 3; }),
      {sz(nz + 1)});
  put(d, "zone_index",
      fill<Omega_h::LO>(
          4 * (nz + 1),
          OMEGA_H_LAMBDA(Omega_h::LO i) { return i % 4 == 3 ? i / 4 + 1 : 0; }),
      {sz(nz + 1), 4});
  put(d, "zone_index_min", std::vector<std::int32_t>(4, 0), {4});
  put(d, "zone_index_max", std::vector<std::int32_t>(4, 0), {4});
  put(d, "zone_pointer",
      fill<Omega_h::LO>(
          nz + 1,
          OMEGA_H_LAMBDA(Omega_h::LO r) {
            return r < nplasma ? r + 1 : r - nplasma + 1;
          }),
      {sz(nz + 1)});
  put(d, "zone_volume",
      fill<Omega_h::Real>(
          nz + 1,
          OMEGA_H_LAMBDA(Omega_h::LO r) {
            return r < nz ? vol[face_of(r)] : rest_volume;
          }),
      {sz(nz + 1)});
  put(d, "zone_center",
      fill<Omega_h::Real>(
          3 * (nz + 1),
          OMEGA_H_LAMBDA(Omega_h::LO i) {
            const int c = i % 3;
            return c == 1 ? 0.0 : ctr[2 * face_of(i / 3) + c / 2];
          }),
      {sz(nz + 1), 3});
  for (int side = 0; side < 2; ++side) // zone_min, zone_max
    put(d, side == 0 ? "zone_min" : "zone_max",
        fill<Omega_h::Real>(
            3 * (nz + 1),
            OMEGA_H_LAMBDA(Omega_h::LO i) {
              const Omega_h::LO r = i / 3;
              const int c = i % 3;
              if (c == 1)
                return 0.0;
              return r < nz ? box[4 * face_of(r) + 2 * side + c / 2]
                            : ub[2 * side + c / 2];
            }),
        {sz(nz + 1), 3});

  // ---- sectors: row 0 unused, rows 2i+1 / 2i+2 are the two sides of wall
  // edge i (first: plasma face side, second: boundary-layer side) ----
  auto plasma_is_pos = OMEGA_H_LAMBDA(Omega_h::LO i) {
    return wf[2 * i] == sorted[2 * we[i]];
  };
  scalar(d, "nsectors", nsect);
  put(d, "strata",
      fill<Omega_h::LO>(
          nsect + 1,
          OMEGA_H_LAMBDA(Omega_h::LO r) {
            return r == 0 ? kIntUnused : nplasma + 1 + (r - 1) / 2;
          }),
      {sz(nsect + 1)});
  put(d, "sector_strata_segment",
      fill<Omega_h::LO>(
          nsect + 1,
          OMEGA_H_LAMBDA(Omega_h::LO r) { return r == 0 ? kIntUnused : 0; }),
      {sz(nsect + 1)});
  put(d, "sectors",
      fill<Omega_h::LO>(
          nsect + 1,
          OMEGA_H_LAMBDA(Omega_h::LO r) {
            if (r == 0)
              return kIntUnused;
            const Omega_h::LO i = (r - 1) / 2, second = (r - 1) % 2;
            return 2 * i + 1 + (plasma_is_pos(i) ? 1 - second : second);
          }),
      {sz(nsect + 1)});
  put(d, "sector_zone",
      fill<Omega_h::LO>(
          nsect + 1,
          OMEGA_H_LAMBDA(Omega_h::LO r) {
            return r == 0 ? kIntUnused : zone[wf[r - 1]];
          }),
      {sz(nsect + 1)});
  put(d, "sector_surface",
      fill<Omega_h::LO>(
          nsect + 1,
          OMEGA_H_LAMBDA(Omega_h::LO r) {
            if (r == 0)
              return kIntUnused;
            const Omega_h::LO i = (r - 1) / 2, second = (r - 1) % 2;
            const bool plus = plasma_is_pos(i) == (second == 0);
            return plus ? we[i] + 1 : -(we[i] + 1);
          }),
      {sz(nsect + 1)});
  put(d, "sector_points",
      fill<Omega_h::Real>(
          6 * (nsect + 1),
          OMEGA_H_LAMBDA(Omega_h::LO i) {
            const Omega_h::LO r = i / 6;
            const int c = i % 3;
            if (r == 0)
              return kDblUnused;
            if (c == 1)
              return 0.0;
            const Omega_h::LO v = ev[2 * we[(r - 1) / 2] + (i % 6) / 3];
            return coords[2 * v + c / 2];
          }),
      {sz(nsect + 1), 2, 3});
  put(d, "sector_type_pointer",
      fill<Omega_h::LO>(
          17 * (nsect + 1),
          OMEGA_H_LAMBDA(Omega_h::LO i) {
            const Omega_h::LO r = i / 17, c = i % 17;
            if (r == 0)
              return kIntUnused / 2;
            const Omega_h::LO w = (r - 1) / 2;
            const bool hit =
                (r - 1) % 2 == 0 ? c == 1 : (c == 2 || (c >= 5 && c <= 7));
            return hit ? w + 1 : kIntUnused;
          }),
      {sz(nsect + 1), 17});

  // ---- sector groups and target settings ----
  scalar(d, "sc_vacuum_num", 0);
  put(d, "vacuum_sector", std::vector<std::int32_t>{kIntUnused}, {1});
  scalar(d, "sc_plasma_num", nwp);
  put(d, "plasma_sector",
      fill<Omega_h::LO>(
          nwp + 1,
          OMEGA_H_LAMBDA(Omega_h::LO r) {
            return r == 0 ? kIntUnused : 2 * r - 1;
          }),
      {sz(nwp + 1)});
  scalar(d, "sc_target_num", nwp);
  put(d, "target_sector",
      fill<Omega_h::LO>(
          nwp + 1, OMEGA_H_LAMBDA(
                       Omega_h::LO r) { return r == 0 ? kIntUnused : 2 * r; }),
      {sz(nwp + 1)});
  put(d, "target_material",
      fill<Omega_h::LO>(
          nwp + 1,
          OMEGA_H_LAMBDA(Omega_h::LO r) { return r == 0 ? kIntUnused : 6; }),
      {sz(nwp + 1)});
  const double temperature_J = o.target_temperature_K * kBoltzmann;
  const double recyc = o.target_recyc_coef;
  put(d, "target_temperature",
      fill<Omega_h::Real>(
          nwp + 1,
          OMEGA_H_LAMBDA(Omega_h::LO r) {
            return r == 0 ? kDblUnused : temperature_J;
          }),
      {sz(nwp + 1)});
  put(d, "target_recyc_coef",
      fill<Omega_h::Real>(
          nwp + 1, OMEGA_H_LAMBDA(
                       Omega_h::LO r) { return r == 0 ? kDblUnused : recyc; }),
      {sz(nwp + 1)});
  scalar(d, "sc_wall_num", 0);
  put(d, "wall_sector", std::vector<std::int32_t>{kIntUnused}, {1});
  put(d, "wall_material", std::vector<std::int32_t>{kIntUnused}, {1});
  put(d, "wall_temperature", std::vector<double>{kDblUnused}, {1});
  put(d, "wall_recyc_coef", std::vector<double>{kDblUnused}, {1});
  scalar(d, "sc_exit_num", 0);
  put(d, "exit_sector", std::vector<std::int32_t>{kIntUnused}, {1});

  // ---- diagnostics (wall and target counts / energy / angle spectra) ----
  scalar(d, "sc_diagnostic_grps", 3);
  scalar(d, "sc_diag_max_bins", 4);
  put(d, "diagnostic_grp_name",
      rows({"UNUSED", "Wall and Target Counts",
            "Wall and Target Energy Spectrum",
            "Wall and Target Angle Spectrum"},
           40),
      {4, 40});
  put(d, "diagnostic_num_sectors",
      std::vector<std::int32_t>{kIntUnused, nbdry, nbdry, nbdry}, {4});
  put(d, "diagnostic_var", std::vector<std::int32_t>{kIntUnused, 0, 1, 2}, {4});
  put(d, "diagnostic_tab_index", std::vector<std::int32_t>{kIntUnused, 0, 4, 4},
      {4});
  put(d, "diagnostic_min",
      std::vector<double>{kDblUnused, 0, -45.5803383244769, 0.174532925199433},
      {4});
  put(d, "diagnostic_delta",
      std::vector<double>{kDblUnused, 0, 2.30258509299405, 0.349065850398866},
      {4});
  put(d, "diagnostic_spacing", std::vector<std::int32_t>{kIntUnused, 0, 2, 1},
      {4});
  put(d, "diagnostic_grp_base",
      std::vector<std::int32_t>{kIntUnused, 0, nbdry, 2 * nbdry}, {4});
  scalar(d, "sc_diag_size", 3 * nwp);
  put(d, "diagnostic_sector_tab",
      fill<Omega_h::LO>(
          3 * nwp, OMEGA_H_LAMBDA(Omega_h::LO i) { return 2 * (i % nwp) + 2; }),
      {sz(3 * nwp)});

  // ---- detectors: none ----
  scalar(d, "de_grps", 0);
  scalar(d, "de_max_bins", 0);
  scalar(d, "de_zone_frags_dim", 100);
  scalar(d, "de_zone_frags_size", 0);
  put(d, "detector_name", rows({"UNUSED"}, 100), {1, 100});
  for (const char *name :
       {"detector_num_views", "detector_var", "detector_tab_index",
        "detector_spacing", "de_view_algorithm", "de_view_base", "de_view_tab"})
    put(d, name, std::vector<std::int32_t>{kIntUnused}, {1});
  for (const char *name :
       {"detector_min", "detector_delta", "de_view_halfwidth"})
    put(d, name, std::vector<double>{kDblUnused}, {1});
  for (const char *name : {"de_zone_frags_start", "de_zone_frags_num",
                           "de_zone_frags_min_zn", "de_zone_frags_max_zn"})
    put(d, name, std::vector<std::int32_t>{0}, {1});
  scalar(d, "detector_total_views", 0);
  put(d, "de_view_points", std::vector<double>(6, kDblUnused), {1, 2, 3});
  put(d, "de_zone_frags", std::vector<double>(100, 0.0), {100});
  std::vector<std::int32_t> frag_zones(100, 4);
  frag_zones[0] = kIntUnused;
  put(d, "de_zone_frags_zones", frag_zones, {100});
  scalar(d, "de_view_size", 1);
  return d;
}

} // namespace omegah2csg
