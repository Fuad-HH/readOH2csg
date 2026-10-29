#include "omegah2csg/degas2_schema.hpp"
#include "omegah2csg/errors.hpp"

namespace omegah2csg {

const std::vector<Degas2Dim> &degas2_dims() {
  static const std::vector<Degas2Dim> dims = {{"vector", 3},
                                              {"string", 300},
                                              {"cell_info_ind", 4},
                                              {"cell_ind", 0},
                                              {"surface_ind", 0},
                                              {"boundary_ind", 0},
                                              {"neighbor_ind", 0},
                                              {"neg_pos", 2},
                                              {"surface_info_ind", 2},
                                              {"surface_tx_ind", 2},
                                              {"tx_ind_1", 3},
                                              {"tx_ind_2", 4},
                                              {"transform_ind", 1},
                                              {"coeff_ind", 11},
                                              {"zone_type_ind", 4},
                                              {"zone_index_ind", 4},
                                              {"zone_ind", 0},
                                              {"sector_ind", 0},
                                              {"sector_neg_pos_ind", 2},
                                              {"sector_type_ind", 17},
                                              {"vacuum_ind", 1},
                                              {"plasma_ind", 0},
                                              {"target_ind", 0},
                                              {"wall_ind", 1},
                                              {"exit_ind", 1},
                                              {"sc_diag_name_string", 40},
                                              {"diag_grp_ind", 4},
                                              {"sc_diag_ind", 0},
                                              {"de_symbol_string", 24},
                                              {"de_name_string", 100},
                                              {"de_grp_ind", 1},
                                              {"de_zone_frags_ind", 100},
                                              {"de_tot_view_ind", 1},
                                              {"de_start_end_ind", 2},
                                              {"de_view_ind", 1}};
  return dims;
}

const std::vector<Degas2VarSpec> &degas2_vars() {
  const DType I = DType::I32, F = DType::F64, C = DType::Char;
  static const std::vector<Degas2VarSpec> vars = {
      {"ncells", I, {}},
      {"nsurfaces", I, {}},
      {"nboundaries", I, {}},
      {"nneighbors", I, {}},
      {"ntransforms", I, {}},
      {"geometry_symmetry", I, {}},
      {"universal_cell_min", F, {"vector"}},
      {"universal_cell_max", F, {"vector"}},
      {"universal_cell_vol", F, {}},
      {"cells", I, {"cell_ind", "cell_info_ind"}},
      {"surfaces", I, {"surface_ind", "surface_info_ind", "neg_pos"}},
      {"surfaces_tx_ind", I, {"surface_ind", "surface_tx_ind", "neg_pos"}},
      {"surfaces_tx_mx", F, {"transform_ind", "tx_ind_2", "tx_ind_1"}},
      {"surface_sectors", I, {"surface_ind", "surface_info_ind", "neg_pos"}},
      {"boundaries", I, {"boundary_ind"}},
      {"neighbors", I, {"neighbor_ind"}},
      {"surface_coeffs", F, {"surface_ind", "coeff_ind"}},
      {"surface_points", F, {"surface_ind", "neg_pos", "vector"}},
      {"zn_num", I, {}},
      {"zone_type_num", I, {"zone_type_ind"}},
      {"zone_type", I, {"zone_ind"}},
      {"zone_index", I, {"zone_ind", "zone_index_ind"}},
      {"zone_index_min", I, {"zone_index_ind"}},
      {"zone_index_max", I, {"zone_index_ind"}},
      {"zone_pointer", I, {"zone_ind"}},
      {"zone_volume", F, {"zone_ind"}},
      {"zone_center", F, {"zone_ind", "vector"}},
      {"zone_min", F, {"zone_ind", "vector"}},
      {"zone_max", F, {"zone_ind", "vector"}},
      {"nsectors", I, {}},
      {"strata", I, {"sector_ind"}},
      {"sector_strata_segment", I, {"sector_ind"}},
      {"sectors", I, {"sector_ind"}},
      {"sector_zone", I, {"sector_ind"}},
      {"sector_surface", I, {"sector_ind"}},
      {"sector_points", F, {"sector_ind", "sector_neg_pos_ind", "vector"}},
      {"sector_type_pointer", I, {"sector_ind", "sector_type_ind"}},
      {"sc_vacuum_num", I, {}},
      {"vacuum_sector", I, {"vacuum_ind"}},
      {"sc_plasma_num", I, {}},
      {"plasma_sector", I, {"plasma_ind"}},
      {"sc_target_num", I, {}},
      {"target_sector", I, {"target_ind"}},
      {"target_material", I, {"target_ind"}},
      {"target_temperature", F, {"target_ind"}},
      {"target_recyc_coef", F, {"target_ind"}},
      {"sc_wall_num", I, {}},
      {"wall_sector", I, {"wall_ind"}},
      {"wall_material", I, {"wall_ind"}},
      {"wall_temperature", F, {"wall_ind"}},
      {"wall_recyc_coef", F, {"wall_ind"}},
      {"sc_exit_num", I, {}},
      {"exit_sector", I, {"exit_ind"}},
      {"sc_diagnostic_grps", I, {}},
      {"sc_diag_max_bins", I, {}},
      {"diagnostic_grp_name", C, {"diag_grp_ind", "sc_diag_name_string"}},
      {"diagnostic_num_sectors", I, {"diag_grp_ind"}},
      {"diagnostic_var", I, {"diag_grp_ind"}},
      {"diagnostic_tab_index", I, {"diag_grp_ind"}},
      {"diagnostic_min", F, {"diag_grp_ind"}},
      {"diagnostic_delta", F, {"diag_grp_ind"}},
      {"diagnostic_spacing", I, {"diag_grp_ind"}},
      {"diagnostic_grp_base", I, {"diag_grp_ind"}},
      {"sc_diag_size", I, {}},
      {"diagnostic_sector_tab", I, {"sc_diag_ind"}},
      {"de_grps", I, {}},
      {"de_max_bins", I, {}},
      {"de_zone_frags_dim", I, {}},
      {"de_zone_frags_size", I, {}},
      {"detector_name", C, {"de_grp_ind", "de_name_string"}},
      {"detector_num_views", I, {"de_grp_ind"}},
      {"detector_var", I, {"de_grp_ind"}},
      {"detector_tab_index", I, {"de_grp_ind"}},
      {"detector_min", F, {"de_grp_ind"}},
      {"detector_delta", F, {"de_grp_ind"}},
      {"detector_spacing", I, {"de_grp_ind"}},
      {"detector_total_views", I, {}},
      {"de_view_points", F, {"de_tot_view_ind", "de_start_end_ind", "vector"}},
      {"de_view_algorithm", I, {"de_tot_view_ind"}},
      {"de_view_halfwidth", F, {"de_tot_view_ind"}},
      {"de_zone_frags", F, {"de_zone_frags_ind"}},
      {"de_zone_frags_start", I, {"de_tot_view_ind"}},
      {"de_zone_frags_num", I, {"de_tot_view_ind"}},
      {"de_zone_frags_zones", I, {"de_zone_frags_ind"}},
      {"de_zone_frags_min_zn", I, {"de_tot_view_ind"}},
      {"de_zone_frags_max_zn", I, {"de_tot_view_ind"}},
      {"de_view_base", I, {"de_grp_ind"}},
      {"de_view_size", I, {}},
      {"de_view_tab", I, {"de_view_ind"}}};
  return vars;
}

std::map<std::string, std::size_t> infer_degas2_dims(
    const std::function<std::vector<std::size_t>(const std::string &)>
        &shape_of) {
  std::map<std::string, std::size_t> sizes;
  for (const auto &d : degas2_dims())
    if (d.fixed)
      sizes[d.name] = d.fixed;
  for (const auto &v : degas2_vars()) {
    const auto shape = shape_of(v.name);
    if (shape.size() != v.dims.size())
      throw InputError(std::string("DEGAS2 variable '") + v.name + "' has " +
                       std::to_string(shape.size()) + " dimensions, expected " +
                       std::to_string(v.dims.size()));
    for (std::size_t i = 0; i < shape.size(); ++i) {
      const auto [it, added] = sizes.emplace(v.dims[i], shape[i]);
      if (!added && it->second != shape[i])
        throw InputError(std::string("DEGAS2 variable '") + v.name +
                         "': dimension '" + v.dims[i] + "' is " +
                         std::to_string(shape[i]) + " but other data says " +
                         std::to_string(it->second));
    }
  }
  return sizes;
}

} // namespace omegah2csg
