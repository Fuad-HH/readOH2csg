from pathlib import Path

import numpy as np

from omegah2csg import OmegaHMesh
from omegah2csg import convert2degas2
import netCDF4 as nc

parent_dir = Path(__file__).resolve().parent
with_boundary_layer_file = parent_dir / 'assets/field_following_ltx_wboundary_layer.osh'
dg2_generated_geometry_filename = parent_dir / 'assets/ltx-field-following-geometry.nc'


def test_is_bounded_by_box():
    with_box_mesh_file = parent_dir / 'assets/unstructured_wbbox_ltx.osh'
    without_box_mesh_file = parent_dir / 'assets/field_following_ltx.osh'

    with OmegaHMesh(with_box_mesh_file) as mesh:
        assert mesh.is_bounded_by_box(), "This mesh is bounded by box but found not"

    with OmegaHMesh(without_box_mesh_file) as mesh:
        assert not mesh.is_bounded_by_box(), "This mesh is not bounded by box but found a box"


def test_has_boundary_layer():
    with OmegaHMesh(with_boundary_layer_file) as mesh:
        assert mesh.has_boundary_layer(), "This mesh wrapped with a boundary layer but found false"

def test_get_integer_tag_array():
    with OmegaHMesh(with_boundary_layer_file) as mesh:
        offset_face_tag = mesh.get_integer_tag_array(2, "offset_face")
        num_offset_faces = offset_face_tag.sum()
        assert num_offset_faces == 429, "For this mesh, there are 429 offset faces"
        print(f"Number of offset faces {num_offset_faces}")

def test_get_cell_volumes():
    with OmegaHMesh(with_boundary_layer_file) as mesh:
        boundary_face_flag = mesh.get_boundary_face_flag()
        num_boundary_faces = boundary_face_flag.sum()
        num_inside_faces = mesh.num_entities(2) - num_boundary_faces
        volumes = mesh.get_cell_volumes()

    with nc.Dataset(dg2_generated_geometry_filename, "r", format="NETCDF4_CLASSIC") as ref_geometry:
        nc_volumes = ref_geometry["zone_volume"][:]

    total_volume = np.sum(volumes[0:num_inside_faces])
    nc_total_volume = np.sum(nc_volumes[0:num_inside_faces])
    print(f"Total volumes: Native={total_volume}, NetCDF={nc_total_volume}")
    assert np.isclose(total_volume, nc_total_volume, rtol=1e-5)

    for i in range(0, num_inside_faces):
        assert np.isclose(nc_volumes[i], volumes[i], rtol = 1e-3), f"{nc_volumes[i]} ≠ {volumes[i]} for face {i}"

def test_get_cell_bounding_boxes():
    with OmegaHMesh(with_boundary_layer_file) as mesh:
        boundary_face_flag = mesh.get_boundary_face_flag()
        num_boundary_faces = boundary_face_flag.sum()
        num_inside_faces = mesh.num_entities(2) - num_boundary_faces
        bboxes = mesh.get_cell_bounding_boxes()

    with nc.Dataset(dg2_generated_geometry_filename, "r", format="NETCDF4_CLASSIC") as ref_geometry:
        bbox_mins = ref_geometry["zone_min"][:]
        bbox_maxes = ref_geometry["zone_max"][:]

    for i in range(0, num_inside_faces):
        # xmin
        coord_tol = 1e-5
        assert np.isclose(bboxes[i*4+0], bbox_mins[i,0], atol=coord_tol), f"Bounding box mismatch for face {i}"
        # ymin
        assert np.isclose(bboxes[i*4+1], bbox_mins[i,2], atol=coord_tol), f"Bounding box mismatch for face {i}"
        # xmax
        assert np.isclose(bboxes[i*4+2], bbox_maxes[i,0], atol=coord_tol), f"Bounding box mismatch for face {i}"
        # ymax
        assert np.isclose(bboxes[i*4+3], bbox_maxes[i,2], atol=coord_tol), f"Bounding box mismatch for face {i}"

def test_get_cell_centroids():
    with OmegaHMesh(with_boundary_layer_file) as mesh:
        boundary_face_flag = mesh.get_boundary_face_flag()
        num_boundary_faces = boundary_face_flag.sum()
        num_inside_faces = mesh.num_entities(2) - num_boundary_faces
        centroids = mesh.get_cell_centroids()

    with nc.Dataset(dg2_generated_geometry_filename, "r", format="NETCDF4_CLASSIC") as ref_geometry:
        nc_centroids = ref_geometry["zone_center"][:]

    for i in range(0, num_inside_faces):
        coord_tol = 1e-5
        assert np.isclose(centroids[i*2+0], nc_centroids[i, 0], atol=coord_tol), "Centroid x mismatch for face {i}"
        assert np.isclose(centroids[i*2+1], nc_centroids[i, 2], atol=coord_tol), "Centroid y mismatch for face {i}"

def test_get_number_of_edges_inside_wall():
    with OmegaHMesh(with_boundary_layer_file) as mesh:
        number_of_edges_inside_wall = mesh.get_number_of_edges_inside_wall()
        #assert number_of_edges_inside_wall == 28917 - get_num_of_boundary_edges(mesh)


def test_convert2degas2():
    with nc.Dataset(dg2_generated_geometry_filename, "r", format="NETCDF4_CLASSIC") as ref_geometry:
        surface_sectors_works = ref_geometry["surface_sectors"]
        sectors_works = ref_geometry["sectors"]
        sector_zone_works = ref_geometry["sector_zone"]
        sector_surface_works = ref_geometry["sector_surface"]
        surfaece_sectors_works = ref_geometry["surface_sectors"]
        surfidx = abs(sector_surface_works[2])
        assert surfidx == 24599, "suridx must be 24599 for this geometry"

def test_get_wall_adjacent_triangles():
    with OmegaHMesh(with_boundary_layer_file) as mesh:
        wall_adjacent_triangles = mesh.get_wall_adjacent_triangles()
        assert len(wall_adjacent_triangles) > 0

        plasma_side = wall_adjacent_triangles[0::2]
        target_side = wall_adjacent_triangles[1::2]
        print("Wall adjacent triangles:")
        print("Plasma side:", plasma_side.tolist())
        print("Target side:", target_side.tolist())

    #convert2degas2(with_boundary_layer_file)

