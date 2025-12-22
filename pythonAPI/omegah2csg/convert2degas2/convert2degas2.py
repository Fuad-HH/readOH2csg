from ctypes import c_bool

import netCDF4

from ..openmcGeometry import get_all_geometry_info
from ..OmegaHMesh import OmegaHMesh
from .. import _dll, OmegaHMeshPointer, kokkos_runtime

_dll.capi_is_mesh_bounded_by_box.restype = c_bool
_dll.capi_is_mesh_bounded_by_box.argtypes = [OmegaHMeshPointer]

def is_bounded_by_box(mesh: OmegaHMesh):
    if not kokkos_runtime.is_running():
        raise RuntimeError("Kokkos not running...")

    try:
        return _dll.capi_is_mesh_bounded_by_box(mesh.mesh)
    except Exception as exception:
        raise RuntimeError(f"Error finding box: {exception}")



def convert2degas2(mesh_filename, netcdf_filename='geometry.nc', tol=1e-10):
    assert netcdf_filename.endswith('.nc'), "Degas2 mesh name should end with .nc but given {}".format(netcdf_filename)
    with OmegaHMesh(mesh_filename) as mesh:
        assert is_bounded_by_box(mesh), "Degas2 requires mesh to be bounded by box. Use addBoxAround tool from tomms."
        [edge_coefficients, boundary_edge_ids, face2edge_map] = get_all_geometry_info(mesh, tol=tol)

    # in  notebook, ncells add number of wall twice
    ncells = face2edge_map.shape[0]
    # notebook has 2*nwall extra
    Nsurf_tot = edge_coefficients.shape[0] + ncells



    # TODO NETCDF_CLASSIC is limited to 2GB
    root_g = netCDF4.Dataset(netcdf_filename, mode='w', format='NETCDF4_CLASSIC')
    vector = root_g.createDimension("vector", 3)
    string = root_g.createDimension("string", 300)
    cell_info_ind = root_g.createDimension("cell_info_ind", 4)
    cell_ind = root_g.createDimension("cell_ind", ncells + 1)
    surface_ind = root_g.createDimension("surface_ind", Nsurf_tot)
    boundary_ind = root_g.createDimension("boundary_ind", nboundaries)
    neighbor_ind = root_g.createDimension("neighbor_ind", nneighbors + 1)
    neg_pos = root_g.createDimension("neg_pos", 2)
    surface_info_ind = root_g.createDimension("surface_info_ind", 2)
    surface_tx_ind = root_g.createDimension("surface_tx_ind", 2)
    tx_ind_1 = root_g.createDimension("tx_ind_1", 3)
    tx_ind_2 = root_g.createDimension("tx_ind_2", 4)
    transform_ind = root_g.createDimension("transform_ind", 1)
    coeff_ind = root_g.createDimension("coeff_ind", 10)
    zone_type_ind = root_g.createDimension("zone_type_ind", 4)
    zone_index_ind = root_g.createDimension("zone_index_ind", 4)
    zone_ind = root_g.createDimension("zone_ind", Nplasma + 1)
    sector_ind = root_g.createDimension("sector_ind", 2 * Nwall + 1)
    sector_neg_pos_ind = root_g.createDimension("sector_neg_pos_ind", 2)
    sector_type_ind = root_g.createDimension("sector_type_ind", 17)
    vacuum_ind = root_g.createDimension("vacuum_ind", 1)
    plasma_ind = root_g.createDimension("plasma_ind", Nwall + 1)
    target_ind = root_g.createDimension("target_ind", Nwall + 1)
    wall_ind = root_g.createDimension("wall_ind", 1)
    exit_ind = root_g.createDimension("exit_ind", 1)
    sc_diag_name_string = root_g.createDimension("sc_diag_name_string", 40)
    diag_grp_ind = root_g.createDimension("diag_grp_ind", 4)
    sc_diag_ind = root_g.createDimension("sc_diag_ind", 3 * Nwall)
    de_symbol_string = root_g.createDimension("de_symbol_string", 24)
    de_name_string = root_g.createDimension("de_name_string", 100)
    de_grp_ind = root_g.createDimension("de_grp_ind", 1)
    de_zone_frags_ind = root_g.createDimension("de_zone_frags_ind", 100)
    de_tot_view_ind = root_g.createDimension("de_tot_view_ind", 1)
    de_start_end_ind = root_g.createDimension("de_start_end_ind", 2)
    de_view_ind = root_g.createDimension("de_view_ind", 1)

    ncells_var = root_g.createVariable("ncells", "i4")
    nsurfaces_var = root_g.createVariable("nsurfaces", "i4")
    nboundaries_var = root_g.createVariable("nboundaries", "i4")
    nneighbors_var = root_g.createVariable("nneighbors", "i4")
    ntransforms_var = root_g.createVariable("ntransforms", "i4")
    geometry_symmetry_var = root_g.createVariable("geometry_symmetry", "i4")
    universal_cell_min_var = root_g.createVariable("universal_cell_min", "f8", ("vector",))
    universal_cell_max_var = root_g.createVariable("universal_cell_max", "f8", ("vector",))
    universal_cell_vol_var = root_g.createVariable("universal_cell_vol", "f8")
    cells_var = root_g.createVariable("cells", "i4", ("cell_ind", "cell_info_ind",))
    surfaces_var = root_g.createVariable("surfaces", "i4", ("surface_ind", "surface_info_ind", "neg_pos",))
    surfaces_tx_ind_var = root_g.createVariable("surfaces_tx_ind", "i4", ("surface_ind", "surface_tx_ind", "neg_pos",))
    surfaces_tx_mx_var = root_g.createVariable("surfaces_tx_mx", "f8", ("transform_ind", "tx_ind_2", "tx_ind_1",))
    surface_sectors_var = root_g.createVariable("surface_sectors", "i4", ("surface_ind", "surface_info_ind", "neg_pos",))
    boundaries_var = root_g.createVariable("boundaries", "i4", ("boundary_ind",))
    neighbors_var = root_g.createVariable("neighbors", "i4", ("neighbor_ind",))
    surface_coeffs_var = root_g.createVariable("surface_coeffs", "f8", ("surface_ind", "coeff_ind",))
    surface_points_var = root_g.createVariable("surface_points", "f8", ("surface_ind", "neg_pos", "vector",))
    zn_num_var = root_g.createVariable("zn_num", "i4")
    zone_type_num_var = root_g.createVariable("zone_type_num", "i4", ("zone_type_ind",))
    zone_type_var = root_g.createVariable("zone_type", "i4", ("zone_ind",))
    zone_index_var = root_g.createVariable("zone_index", "i4", ("zone_ind", "zone_index_ind",))
    zone_index_min_var = root_g.createVariable("zone_index_min", "i4", ("zone_index_ind",))
    zone_index_max_var = root_g.createVariable("zone_index_max", "i4", ("zone_index_ind",))
    zone_pointer_var = root_g.createVariable("zone_pointer", "i4", ("zone_ind",))
    zone_volume_var = root_g.createVariable("zone_volume", "f8", ("zone_ind",))
    zone_center_var = root_g.createVariable("zone_center", "f8", ("zone_ind", "vector",))
    zone_min_var = root_g.createVariable("zone_min", "f8", ("zone_ind", "vector",))
    zone_max_var = root_g.createVariable("zone_max", "f8", ("zone_ind", "vector",))
    nsectors_var = root_g.createVariable("nsectors", "i4")
    strata_var = root_g.createVariable("strata", "i4", ("sector_ind",))
    sector_strata_segment_var = root_g.createVariable("sector_strata_segment", "i4", ("sector_ind",))
    sectors_var = root_g.createVariable("sectors", "i4", ("sector_ind",))
    sector_zone_var = root_g.createVariable("sector_zone", "i4", ("sector_ind",))
    sector_surface_var = root_g.createVariable("sector_surface", "i4", ("sector_ind",))
    sector_points_var = root_g.createVariable("sector_points", "f8", ("sector_ind", "sector_neg_pos_ind", "vector",))
    sector_type_pointer_var = root_g.createVariable("sector_type_pointer", "i4", ("sector_ind", "sector_type_ind",))
    sc_vacuum_num_var = root_g.createVariable("sc_vacuum_num", "i4")
    vacuum_sector_var = root_g.createVariable("vacuum_sector", "i4", ("vacuum_ind",))
    sc_plasma_num_var = root_g.createVariable("sc_plasma_num", "i4")
    plasma_sector_var = root_g.createVariable("plasma_sector", "i4", ("plasma_ind",))
    sc_target_num_var = root_g.createVariable("sc_target_num", "i4")
    target_sector_var = root_g.createVariable("target_sector", "i4", ("target_ind",))
    target_material_var = root_g.createVariable("target_material", "i4", ("target_ind",))
    target_temperature_var = root_g.createVariable("target_temperature", "f8", ("target_ind",))
    target_recyc_coef_var = root_g.createVariable("target_recyc_coef", "f8", ("target_ind",))
    sc_wall_num_var = root_g.createVariable("sc_wall_num", "i4")
    wall_sector_var = root_g.createVariable("wall_sector", "i4", ("wall_ind",))
    wall_material_var = root_g.createVariable("wall_material", "i4", ("wall_ind",))
    wall_temperature_var = root_g.createVariable("wall_temperature", "f8", ("wall_ind",))
    wall_recyc_coef_var = root_g.createVariable("wall_recyc_coef", "f8", ("wall_ind",))
    sc_exit_num_var = root_g.createVariable("sc_exit_num", "i4")
    exit_sector_var = root_g.createVariable("exit_sector", "i4", ("exit_ind",))
    sc_diagnostic_grps_var = root_g.createVariable("sc_diagnostic_grps", "i4")
    sc_diag_max_bins_var = root_g.createVariable("sc_diag_max_bins", "i4")
    diagnostic_grp_name_var = root_g.createVariable("diagnostic_grp_name", "c", ("diag_grp_ind", "sc_diag_name_string",))
    diagnostic_num_sectors_var = root_g.createVariable("diagnostic_num_sectors", "i4", ("diag_grp_ind",))
    diagnostic_var_var = root_g.createVariable("diagnostic_var", "i4", ("diag_grp_ind",))
    diagnostic_tab_index_var = root_g.createVariable("diagnostic_tab_index", "i4", ("diag_grp_ind",))
    diagnostic_min_var = root_g.createVariable("diagnostic_min", "f8", ("diag_grp_ind",))
    diagnostic_delta_var = root_g.createVariable("diagnostic_delta", "f8", ("diag_grp_ind",))
    diagnostic_spacing_var = root_g.createVariable("diagnostic_spacing", "i4", ("diag_grp_ind",))
    diagnostic_grp_base_var = root_g.createVariable("diagnostic_grp_base", "i4", ("diag_grp_ind",))
    sc_diag_size_var = root_g.createVariable("sc_diag_size", "i4")
    diagnostic_sector_tab_var = root_g.createVariable("diagnostic_sector_tab", "i4", ("sc_diag_ind",))
    de_grps_var = root_g.createVariable("de_grps", "i4")
    de_max_bins_var = root_g.createVariable("de_max_bins", "i4")
    de_zone_frags_dim_var = root_g.createVariable("de_zone_frags_dim", "i4")
    de_zone_frags_size_var = root_g.createVariable("de_zone_frags_size", "i4")
    detector_name_var = root_g.createVariable("detector_name", "c", ("de_grp_ind", "de_name_string",))
    detector_num_views_var = root_g.createVariable("detector_num_views", "i4", ("de_grp_ind",))
    detector_var_var = root_g.createVariable("detector_var", "i4", ("de_grp_ind",))
    detector_tab_index_var = root_g.createVariable("detector_tab_index", "i4", ("de_grp_ind",))
    detector_min_var = root_g.createVariable("detector_min", "f8", ("de_grp_ind",))
    detector_delta_var = root_g.createVariable("detector_delta", "f8", ("de_grp_ind",))
    detector_spacing_var = root_g.createVariable("detector_spacing", "i4", ("de_grp_ind",))
    detector_total_views_var = root_g.createVariable("detector_total_views", "i4")
    de_view_points_var = root_g.createVariable("de_view_points", "f8", ("de_tot_view_ind", "de_start_end_ind", "vector",))
    de_view_algorithm_var = root_g.createVariable("de_view_algorithm", "i4", ("de_tot_view_ind",))
    de_view_halfwidth_var = root_g.createVariable("de_view_halfwidth", "f8", ("de_tot_view_ind",))
    de_zone_frags_var = root_g.createVariable("de_zone_frags", "f8", ("de_zone_frags_ind",))
    de_zone_frags_start_var = root_g.createVariable("de_zone_frags_start", "i4", ("de_tot_view_ind",))
    de_zone_frags_num_var = root_g.createVariable("de_zone_frags_num", "i4", ("de_tot_view_ind",))
    de_zone_frags_zones_var = root_g.createVariable("de_zone_frags_zones", "i4", ("de_zone_frags_ind",))
    de_zone_frags_min_zn_var = root_g.createVariable("de_zone_frags_min_zn", "i4", ("de_tot_view_ind",))
    de_zone_frags_max_zn_var = root_g.createVariable("de_zone_frags_max_zn", "i4", ("de_tot_view_ind",))
    de_view_base_var = root_g.createVariable("de_view_base", "i4", ("de_grp_ind",))
    de_view_size_var = root_g.createVariable("de_view_size", "i4")
    de_view_tab_var = root_g.createVariable("de_view_tab", "i4", ("de_view_ind",))

    root_g.close()

