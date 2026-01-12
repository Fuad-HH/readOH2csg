import netCDF4
import numpy as np

from ..openmcGeometry import get_all_geometry_info
from ..OmegaHMesh import OmegaHMesh

INT_UNUSED = 2000000000
DBL_UNUSED = 2.0e30
STR_UNUSED = "UNUSED                                                                                              "


def sort_edge_to_face_map(edge_to_face_map, face2edge_map) -> np.ndarray:
    nedges = edge_to_face_map.shape[0]

    for i in range(0, nedges):
        first_face = edge_to_face_map[i, 0]
        for j in range(0, 3):
            edge = face2edge_map[first_face, 2*j]
            if edge == i:
                sign = face2edge_map[first_face, 2*j+1]
                if sign == -1: # then swap
                    edge_to_face_map[i,0], edge_to_face_map[i,1] = edge_to_face_map[i,1], edge_to_face_map[i,0]
                    continue
                if sign == 1:
                    continue
                else:
                    raise RuntimeError(f"Sign cannon be {sign}")

    return edge_to_face_map

def convert2degas2(mesh_filename, netcdf_filename='geometry.nc', tol=1e-10):
    assert netcdf_filename.endswith('.nc'), "Degas2 mesh name should end with .nc but given {}".format(netcdf_filename)
    with OmegaHMesh(mesh_filename) as mesh:
        assert mesh.has_boundary_layer, "Degas2 requires mesh to have a boundary layer. Use addBonudaryLayer tool from tomms."
        [edge_coefficients, boundary_edge_ids, face2edge_map] = get_all_geometry_info(mesh, tol=tol)
        boundary_face_flag = mesh.get_boundary_face_flag()
        num_node = mesh.num_entities(0)
        cell_bounding_boxes = mesh.get_cell_bounding_boxes()
        tri_volumes = mesh.get_cell_volumes()
        centroids = mesh.get_cell_centroids()
        edge_coordinates = mesh.get_edge_coordinates()
        edge_to_face_map = mesh.get_edge_to_face_map()
        edge_to_face_map_sorted = sort_edge_to_face_map(edge_to_face_map, face2edge_map)

    Ntri = face2edge_map.shape[0] # ncells
    num_boundary_face = boundary_face_flag.sum()
    Nplasma = Ntri - num_boundary_face
    Nedge = edge_coefficients.shape[0]
    Nsurf_tot = Nedge + 2*Ntri # each triangle has two cut surfaces nSurf_tot, nsurfaces
    Nwall = boundary_edge_ids.shape[0]
    # universal cells has Nwall boundaries and each triangle has 3 sides and 2 cut surfaces
    nboundaries = Nwall + 5*Ntri
    nneighbors = Nedge*2
    nsectors = 2 * Nwall

    # FIXME NETCDF_CLASSIC is limited to 2GB
    root_g = netCDF4.Dataset(netcdf_filename, mode='w', format='NETCDF4_CLASSIC')


    # *********************************************************************************************** #
    # ---------------------------------- Dimensions ------------------------------------------------- #
    # *********************************************************************************************** #

    vector = root_g.createDimension("vector", 3)
    string = root_g.createDimension("string", 300)
    cell_info_ind = root_g.createDimension("cell_info_ind", 4)
    cell_ind = root_g.createDimension("cell_ind", Ntri + 1)
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

    # *********************************************************************************************** #
    # ---------------------------------- Variables -------------------------------------------------- #
    # *********************************************************************************************** #

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
    zone_type_var = root_g.createVariable("zone_type", "i4", ("zone_ind",)) # ask
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


    # *********************************************************************************************** #
    # ---------------------------------- Fill Up ---------------------------------------------------- #
    # *********************************************************************************************** #

    ncells_var[:] = Ntri
    nsurfaces_var[:] = Nsurf_tot
    nboundaries_var[:] = [nboundaries]
    nneighbors_var[:] = nneighbors
    ntransforms_var[:] = 0 # hardcoded to zero
    geometry_symmetry_var[:] = 2 # hardcoded to two

    assert cell_bounding_boxes.size == Ntri * 4
    universal_cell_min = [np.min(cell_bounding_boxes[0::4]), 0.0, np.min(cell_bounding_boxes[1::4])]
    universal_cell_max = [np.max(cell_bounding_boxes[2::4]), 6.28318530717959, np.max(cell_bounding_boxes[3::4])]
    universal_cell_min_var[:] = universal_cell_min
    universal_cell_max_var[:] = universal_cell_max

    zone_min = np.zeros((Nplasma + 1, 3))
    zone_min[0:Nplasma, 0] = cell_bounding_boxes[0:Nplasma*4:4]
    zone_min[0:Nplasma, 2] = cell_bounding_boxes[1:Nplasma*4:4]
    zone_min[Nplasma, :] = [universal_cell_min[0], 0.0, universal_cell_min[2]]
    zone_min_var[:] = zone_min

    zone_max = np.zeros((Nplasma + 1, 3))
    zone_max[0:Nplasma, 0] = cell_bounding_boxes[2:Nplasma*4:4]
    zone_max[0:Nplasma, 2] = cell_bounding_boxes[3:Nplasma*4:4]
    zone_max[Nplasma, :] = [universal_cell_max[0], 0.0, universal_cell_max[2]]
    zone_max_var[:] = zone_max

    total_volume = tri_volumes.sum()
    universal_cell_vol_var[:] = total_volume

    zone_volume = np.empty(Nplasma+1)
    zone_volume[0:Nplasma] = tri_volumes[0:Nplasma]
    zone_volume[Nplasma] = total_volume
    zone_volume_var[:] = zone_volume

    # ------------------------- Cells -------------------------------------- #
    # in note ncells = Ntri = Nplasma + 2*Nwall; ncells = ncells = Ntri+2*Nwall
    cells = np.zeros([Ntri + 1, 4], dtype=int) # ask
    cells[0, 0:4] = [1, Nwall, Nwall, 0]
    # todo: fix this indexing
    cells[1:Ntri + 1, 0] = 1 + Nwall + 5 * np.array(range(0, Ntri), dtype=int)
    cells[1:Ntri + 1, 1] = 3
    cells[1:Ntri + 1, 2] = 5
    cells[1:Nplasma + 1, 3] = np.array(range(1, Nplasma + 1), dtype=int)
    cells[Nplasma + 1:, 3] = Nplasma + 1
    ncells_var[:] = Ntri
    cells_var[:] = cells

    # ------------------------ Surfaces ------------------------------------ #
    surfaces = np.zeros([Nsurf_tot, 2, 2], dtype=int)

    zone_center = np.zeros([Nplasma + 1, 3])
    zone_center[0:Nplasma, 0] = centroids[0:Nplasma*2:2]
    zone_center[0:Nplasma, 1] = centroids[1:Nplasma*2:2]
    # these two are slightly different from notebook
    zone_center[Nplasma, 0] = np.average(centroids[0:Nplasma*2:2])
    zone_center[Nplasma, 2] = np.average(centroids[1:Nplasma*2:2])
    zone_center_var[:] = zone_center

    surface_points = np.zeros([Nsurf_tot, 2, 3])
    surface_points[0:Nedge, :, 0] = edge_coordinates[0:Nedge*4:2].reshape((Nedge, 2))
    surface_points[0:Nedge, :, 2] = edge_coordinates[1:Nedge*4:2].reshape((Nedge, 2))
    surface_points_var[:] = surface_points

    # ------------------------- Boundaries ------------------------------------ #
    boundaries = np.zeros(nboundaries, dtype=int)
    for i in range(0, Ntri):
        bdy_start = Nwall + i*5
        cut_start = Nedge + 2*i

        boundaries[bdy_start:bdy_start + 3] = [face2edge_map[i,0]*face2edge_map[i,1],
                                               face2edge_map[i,2]*face2edge_map[i,3],
                                               face2edge_map[i,4]*face2edge_map[i,5]]
        # todo check this: we may have different indices
        boundaries[bdy_start + 3:bdy_start + 5] = [cut_start + 1, cut_start + 2]

    # fill edges of the universal cell
    boundaries[0:Nwall] = boundary_edge_ids

    boundaries_var[:] = boundaries

    # ----------------------- Neighbors (Edge Adjacency Info) ----------------- #
    neighbors = np.zeros(nneighbors + 1, dtype=int)
    # ask


    # -------------------------- Sector ---------------------------------------- #
    sector_type_pointer = INT_UNUSED * np.ones([nsectors + 1, 17], dtype=int)
    sc_diag_size = 3 * Nwall # ask
    diagnostic_sector_tab = np.zeros(sc_diag_size, dtype=int)

    sector_type_pointer[0, :] = INT_UNUSED / 2
    # fixme after surface, face, etc.
    for i in range(0, Nwall):
        sector_type_pointer[2*i+1,1] = i+1
        sector_type_pointer[2*i+2,2] = i+1
        sector_type_pointer[2*i+2,5:8] = i+1

        diagnostic_sector_tab[i] = 2 * i + 2
        diagnostic_sector_tab[Nwall + i] = 2 * i + 2
        diagnostic_sector_tab[2 * Nwall + i] = 2 * i + 2

    sector_type_pointer_var[:] = sector_type_pointer
    diagnostic_sector_tab_var[:] = diagnostic_sector_tab

    # ----------------------- Unmodified Variables ----------------------------- #
    de_grps = 0
    de_view_size = 1
    de_grps = 0
    de_max_bins = 0
    de_zone_fragment_dim = 100
    de_zone_frags_size = 0
    de_zone_frags_ind = 100
    detector_total_views = 0
    zn_num = Nplasma + 1
    sc_vacuum_num = 0
    sc_plasma_num = Nwall
    sc_target_num = Nwall
    sc_wall_num = 0
    sc_exit_num = 0
    sc_diagnostic_grps = 3
    sc_diag_max_bins = 4

    zn_num_var[:] = zn_num
    vacuum_sector = np.zeros(sc_vacuum_num + 1, dtype=int)
    vacuum_sector[0] = INT_UNUSED
    de_view_tab = INT_UNUSED * np.ones(de_view_size, dtype=int)
    de_view_tab_var[:] = de_view_tab
    de_view_size_var[:] = de_view_size
    de_view_base = INT_UNUSED * np.ones(de_grps + 1, dtype=int)
    de_view_base_var[:] = de_view_base
    sc_plasma_num_var[:] = sc_plasma_num
    sc_target_num_var[:] = sc_target_num
    sc_wall_num_var[:] = sc_wall_num
    sc_exit_num_var[:] = sc_exit_num
    sc_diagnostic_grps_var[:] = sc_diagnostic_grps
    sc_diag_max_bins_var[:] = sc_diag_max_bins
    sc_vacuum_num_var[:] = sc_vacuum_num

    #target_temperature = np.zeros(sc_target_num+1) # ask
    Twall = 300 * 1.380649e-23
    target_temperature = Twall*np.ones(sc_target_num+1)
    target_temperature[0]=DBL_UNUSED
    target_temperature_var[:] = target_temperature

    wall_recyc_coef = np.zeros(sc_wall_num + 1)
    wall_recyc_coef[0] = DBL_UNUSED
    wall_recyc_coef_var[:] = wall_recyc_coef

    exit_sector = np.zeros(sc_exit_num + 1, dtype=int)
    exit_sector[0] = INT_UNUSED
    exit_sector_var[:] = exit_sector

    vacuum_sector = np.zeros(sc_vacuum_num + 1, dtype=int)
    wall_sector = np.zeros(sc_wall_num + 1, dtype=int)
    wall_material = np.zeros(sc_wall_num + 1, dtype=int)
    wall_temperature = np.zeros(sc_wall_num + 1)
    wall_recyc_coef = np.zeros(sc_wall_num + 1)
    vacuum_sector[0] = INT_UNUSED
    wall_sector[0] = INT_UNUSED
    wall_material[0] = INT_UNUSED
    wall_temperature[0] = DBL_UNUSED
    wall_recyc_coef[0] = DBL_UNUSED
    vacuum_sector_var[:] = vacuum_sector
    wall_sector_var[:] = wall_sector
    wall_material_var[:] = wall_material
    wall_temperature_var[:] = wall_temperature
    wall_recyc_coef_var[:] = wall_recyc_coef

    recyc_coef = 0.99
    target_recyc_coef = recyc_coef * np.ones(sc_target_num + 1)
    target_recyc_coef[0] = DBL_UNUSED
    target_recyc_coef_var[:] = target_recyc_coef

    target_material = 4 * np.ones(sc_target_num + 1, dtype=int)
    target_material[0] = INT_UNUSED
    target_material_var[:] = target_material

    zone_type_num = np.zeros(4, dtype=int)
    zone_type_num[1] = Nplasma
    zone_type_num[2] = 1
    zone_type_num_var[:] = zone_type_num

    surfaces_tx_ind = np.zeros([Nsurf_tot, 2, 2], dtype=int)
    surfaces_tx_mx = DBL_UNUSED * np.ones([1, 4, 3])
    surfaces_tx_ind_var[:] = surfaces_tx_ind
    surfaces_tx_mx_var[:] = surfaces_tx_mx

    zone_index = np.zeros([Nplasma + 1, 4], dtype=int)
    zone_index[:, 3] = np.array(range(1, Nplasma + 2))
    zone_index_var[:] = zone_index

    zone_index_min = np.zeros(4, dtype=int)
    zone_index_max = np.zeros(4, dtype=int)
    zone_index_min_var[:] = zone_index_min
    zone_index_max_var[:] = zone_index_max

    nsectors_var[:] = nsectors
    strata = (Nplasma + 1) * np.ones(nsectors + 1, dtype=int)
    strata[0] = INT_UNUSED
    strata_var[:] = strata

    detector_name = [STR_UNUSED] * (de_grps + 1)
    detector_num_views = INT_UNUSED * np.ones(shape=de_grps + 1, dtype=int)
    detector_var = INT_UNUSED * np.ones(de_grps + 1, dtype=int)
    detector_tab_index = INT_UNUSED * np.ones(de_grps + 1, dtype=int)
    detector_min = DBL_UNUSED * np.ones(de_grps + 1)
    detector_delta = DBL_UNUSED * np.ones(de_grps + 1)
    detector_spacing = INT_UNUSED * np.ones(de_grps + 1, dtype=int)
    de_view_points = DBL_UNUSED * np.ones([detector_total_views + 1, 2, 3])
    de_view_algorithm = INT_UNUSED * np.ones(detector_total_views + 1, dtype=int)
    de_view_halfwidth = DBL_UNUSED * np.ones(detector_total_views + 1)
    de_zone_frags = np.zeros(de_zone_frags_ind)
    de_zone_frags_start = np.zeros(detector_total_views + 1, dtype=int)
    de_zone_frags_num = np.zeros(detector_total_views + 1, dtype=int)
    de_zone_frags_zones = np.zeros(de_zone_frags_ind)
    de_zone_frags_zones[:] = 4
    de_zone_frags_zones[0] = INT_UNUSED
    de_zone_frags_min_zn = np.zeros(detector_total_views + 1, dtype=int)
    de_zone_frags_max_zn = np.zeros(detector_total_views + 1, dtype=int)
    detector_num_views_var[:] = detector_num_views
    detector_var_var[:] = detector_var
    detector_tab_index_var[:] = detector_tab_index
    detector_min_var[:] = detector_min
    detector_delta_var[:] = detector_delta
    detector_spacing_var[:] = detector_spacing
    detector_total_views_var[:] = detector_total_views
    de_view_points_var[:] = de_view_points
    de_view_algorithm_var[:] = de_view_algorithm
    de_view_halfwidth_var[:] = de_view_halfwidth
    de_zone_frags_var[:] = de_zone_frags
    de_zone_frags_start_var[:] = de_zone_frags_start
    de_zone_frags_num_var[:] = de_zone_frags_num
    de_zone_frags_zones_var[:] = de_zone_frags_zones
    de_zone_frags_min_zn_var[:] = de_zone_frags_min_zn
    de_zone_frags_max_zn_var[:] = de_zone_frags_max_zn
    diagnostic_num_sectors = np.array([INT_UNUSED, Nwall, Nwall, Nwall])
    diagnostic_var = np.array([INT_UNUSED, 0, 1, 2])
    diagnostic_tab_index = np.array([INT_UNUSED, 0, 4, 4])
    diagnostic_spacing = np.array([INT_UNUSED, 0, 2, 1])
    diagnostic_grp_base = np.array([INT_UNUSED, 0, Nwall, 2 * Nwall])
    diagnostic_min = np.array([DBL_UNUSED, 0, -45.5803383244769, 0.174532925199433])
    diagnostic_delta = np.array([DBL_UNUSED, 0, 2.30258509299405, 0.349065850398866])
    diagnostic_num_sectors_var[:] = diagnostic_num_sectors
    diagnostic_var_var[:] = diagnostic_var
    diagnostic_tab_index_var[:] = diagnostic_tab_index
    diagnostic_min_var[:] = diagnostic_min
    diagnostic_delta_var[:] = diagnostic_delta
    diagnostic_spacing_var[:] = diagnostic_spacing
    diagnostic_grp_base_var[:] = diagnostic_grp_base
    sc_diag_size = 3 * Nwall
    sc_diag_size_var[:] = sc_diag_size
    de_grps_var[:] = de_grps
    de_max_bins_var[:] = de_max_bins
    de_zone_frags_dim = 100
    de_zone_frags_dim_var[:] = de_zone_frags_dim
    de_zone_frags_size_var[:] = de_zone_frags_size

    diagnostic_grp_name_var[0, :] = "UNUSED                                  "
    diagnostic_grp_name_var[1, :] = "Wall and Target Counts                  "
    diagnostic_grp_name_var[2, :] = "Wall and Target Energy Spectrum         "
    diagnostic_grp_name_var[3, :] = "Wall and Target Angle Spectrum          "
    detector_name_var[
        :] = "UNUSED                                                                                              "

    root_g.sync()
    root_g.close()

