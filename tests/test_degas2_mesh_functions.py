from pathlib import Path
import numpy as np

from omegah2csg import OmegaHMesh
from omegah2csg import convert2degas2
import netCDF4 as nc

parent_dir = Path(__file__).resolve().parent
omega_h_mesh_to_convert = parent_dir / "assets/simple_degas2_case/tagged-dg2mesh.osh"
gold_nc_mesh = parent_dir / "assets/simple_degas2_case/gold-geometry.nc"
dg2d_converted_mesh = (
    parent_dir / "assets/simple_degas2_case/definegeometry2d-geometry.nc"
)


def test_is_bounded_by_box():
    with OmegaHMesh(omega_h_mesh_to_convert) as mesh:
        assert mesh.is_bounded_by_box(), "This mesh is bounded by box but found not"


def test_has_boundary_layer():
    with OmegaHMesh(omega_h_mesh_to_convert) as mesh:
        assert mesh.has_boundary_layer(), (
            "This mesh wrapped with a boundary layer but found false"
        )


def test_get_integer_tag_array():
    with OmegaHMesh(omega_h_mesh_to_convert) as mesh:
        offset_face_tag = mesh.get_integer_tag_array(2, "offset_face")
        num_offset_faces = offset_face_tag.sum()
        print(f"Number of offset faces {num_offset_faces}")
        assert num_offset_faces == 28, (
            "For this mesh, there are 28 offset faces."
            "Check picture in assets/simple_degas2_case/README.md"
        )


def test_get_cell_volumes():
    with OmegaHMesh(omega_h_mesh_to_convert) as mesh:
        boundary_face_flag = mesh.get_boundary_face_flag()
        num_boundary_faces = boundary_face_flag.sum()
        num_inside_faces = mesh.num_entities(2) - num_boundary_faces
        oh_cell_volumes = mesh.get_cell_volumes()
        oh_total_volume = np.sum(oh_cell_volumes)

    with nc.Dataset(gold_nc_mesh, "r", format="NETCDF4_CLASSIC") as mesh:
        gold_nc_zone_volumes = mesh["zone_volume"][:]
        gold_nc_total_volume = mesh["universal_cell_vol"][:]
        assert np.isclose(oh_total_volume, gold_nc_total_volume, rtol=1e-5), (
            f"Total volume mismatch. {oh_total_volume=} , {gold_nc_total_volume=}"
        )

    with nc.Dataset(dg2d_converted_mesh, "r", format="NETCDF4_CLASSIC") as mesh:
        dg2d_total_volume = mesh["universal_cell_vol"][:]
        dg2d_nc_zone_volumes = mesh["zone_volume"][:]
        assert np.isclose(oh_total_volume, dg2d_total_volume, rtol=1e-5), (
            f"Total volume mismatch. {oh_total_volume=} , {dg2d_total_volume=}"
        )

    for i in range(0, num_inside_faces):
        assert np.isclose(gold_nc_zone_volumes[i], oh_cell_volumes[i], rtol=1e-5), (
            f"{gold_nc_zone_volumes[i]} ≠ {oh_cell_volumes[i]} for face {i}"
        )
        assert np.isclose(dg2d_nc_zone_volumes[i], oh_cell_volumes[i], rtol=1e-5), (
            f"{dg2d_nc_zone_volumes[i]} ≠ {oh_cell_volumes[i]} for face {i}"
        )


def test_get_cell_bounding_boxes():
    with OmegaHMesh(omega_h_mesh_to_convert) as mesh:
        boundary_face_flag = mesh.get_boundary_face_flag()
        num_boundary_faces = boundary_face_flag.sum()
        num_inside_faces = mesh.num_entities(2) - num_boundary_faces
        oh_cell_bboxes = mesh.get_cell_bounding_boxes()

    with nc.Dataset(gold_nc_mesh, "r", format="NETCDF4_CLASSIC") as mesh:
        gold_nc_bbox_mins = mesh["zone_min"][:]
        gold_nc_bbox_maxs = mesh["zone_max"][:]

    with nc.Dataset(dg2d_converted_mesh, "r", format="NETCDF4_CLASSIC") as mesh:
        dg2d_nc_bbox_mins = mesh["zone_min"][:]
        dg2d_nc_bbox_maxs = mesh["zone_max"][:]

    for i in range(0, num_inside_faces):
        # xmin
        coord_tol = 1e-5
        assert np.isclose(
            oh_cell_bboxes[i * 4 + 0], gold_nc_bbox_mins[i, 0], atol=coord_tol
        ), f"Bounding box mismatch for face {i}"
        assert np.isclose(
            oh_cell_bboxes[i * 4 + 0], dg2d_nc_bbox_mins[i, 0], atol=coord_tol
        ), f"Bounding box mismatch for face {i}"
        # ymin
        assert np.isclose(
            oh_cell_bboxes[i * 4 + 1], gold_nc_bbox_mins[i, 2], atol=coord_tol
        ), f"Bounding box mismatch for face {i}"
        assert np.isclose(
            oh_cell_bboxes[i * 4 + 1], dg2d_nc_bbox_mins[i, 2], atol=coord_tol
        ), f"Bounding box mismatch for face {i}"
        # xmax
        assert np.isclose(
            oh_cell_bboxes[i * 4 + 2], gold_nc_bbox_maxs[i, 0], atol=coord_tol
        ), f"Bounding box mismatch for face {i}"
        assert np.isclose(
            oh_cell_bboxes[i * 4 + 2], dg2d_nc_bbox_maxs[i, 0], atol=coord_tol
        ), f"Bounding box mismatch for face {i}"
        # ymax
        assert np.isclose(
            oh_cell_bboxes[i * 4 + 3], gold_nc_bbox_maxs[i, 2], atol=coord_tol
        ), f"Bounding box mismatch for face {i}"
        assert np.isclose(
            oh_cell_bboxes[i * 4 + 3], dg2d_nc_bbox_maxs[i, 2], atol=coord_tol
        ), f"Bounding box mismatch for face {i}"


def test_get_cell_centroids():
    with OmegaHMesh(omega_h_mesh_to_convert) as mesh:
        boundary_face_flag = mesh.get_boundary_face_flag()
        num_boundary_faces = boundary_face_flag.sum()
        num_inside_faces = mesh.num_entities(2) - num_boundary_faces
        oh_cell_centroids = mesh.get_cell_centroids()

    with nc.Dataset(gold_nc_mesh, "r", format="NETCDF4_CLASSIC") as mesh:
        gold_zone_centroids = mesh["zone_center"][:]

    with nc.Dataset(dg2d_converted_mesh, "r", format="NETCDF4_CLASSIC") as mesh:
        dg2d_zone_centroids = mesh["zone_center"][:]

    for i in range(0, num_inside_faces):
        coord_tol = 1e-5
        assert np.isclose(
            oh_cell_centroids[i * 2 + 0], gold_zone_centroids[i, 0], atol=coord_tol
        ), f"Centroid x mismatch for face {i}"
        assert np.isclose(
            oh_cell_centroids[i * 2 + 0], dg2d_zone_centroids[i, 0], atol=coord_tol
        ), f"Centroid x mismatch for face {i}"

        assert np.isclose(
            oh_cell_centroids[i * 2 + 1], gold_zone_centroids[i, 2], atol=coord_tol
        ), f"Centroid y mismatch for face {i}"
        assert np.isclose(
            oh_cell_centroids[i * 2 + 1], dg2d_zone_centroids[i, 2], atol=coord_tol
        ), f"Centroid y mismatch for face {i}"


def test_get_number_of_edges_inside_wall():
    with OmegaHMesh(omega_h_mesh_to_convert) as mesh:
        number_of_edges_inside_wall = mesh.get_number_of_edges_inside_wall()
        assert number_of_edges_inside_wall == 14, (
            f"Number of edges_inside_wall (expected 14) = {number_of_edges_inside_wall}"
        )


def test_convert2degas2():
    convert2degas2(omega_h_mesh_to_convert, create_aux_files=False)


def test_small_netcdf_write():
    """
    The tests if the netcdf installation (python) is working on the current platform.
    """
    root_g = nc.Dataset("test.nc", mode="w", format="NETCDF4")
    vector = root_g.createDimension("vector", 3)
    integer_scalar_var = root_g.createVariable("int_scalar", "i4", ())

    vector_var = root_g.createVariable("vector_var", "f8", ("vector",))
    print(f"{vector_var.shape=}")
    some_vector = np.array([0.0, 0.0, 0.0])
    print(f"{some_vector.shape=}")
    vector_var[:] = some_vector
    integer_scalar_var[:] = 5
    root_g.close()


def test_get_wall_adjacent_triangles():
    with OmegaHMesh(omega_h_mesh_to_convert) as mesh:
        wall_adjacent_triangles = mesh.get_wall_adjacent_triangles()
        assert len(wall_adjacent_triangles) > 0

        plasma_side = wall_adjacent_triangles[0::2]
        target_side = wall_adjacent_triangles[1::2]
        print("Wall adjacent triangles:")
        print("Plasma side:", plasma_side.tolist())
        print("Target side:", target_side.tolist())

    ref_plasma_side = np.array([0, 2, 3, 4, 10, 10, 11, 11], dtype=int)
    ref_target_side = np.array([12, 15, 17, 18, 21, 22, 24, 27], dtype=int)

    # sort plasma_side and target_side for comparison
    plasma_side = np.sort(plasma_side)
    target_side = np.sort(target_side)

    assert np.all(ref_plasma_side == plasma_side), (
        f"Plasma side cell id mismatch {ref_plasma_side=} != {plasma_side=}"
    )
    assert np.all(ref_target_side == target_side), (
        f"Target side cell id mismatch {ref_target_side=} != {target_side=}"
    )
