from pathlib import Path
from omegah2csg import is_bounded_by_box
from omegah2csg import OmegaHMesh

parent_dir = Path(__file__).resolve().parent

def test_is_bounded_by_box():
    with_box_mesh_file = parent_dir / 'assets/unstructured_wbbox_ltx.osh'
    without_box_mesh_file = parent_dir / 'assets/field_following_ltx.osh'

    with OmegaHMesh(with_box_mesh_file) as mesh:
        assert is_bounded_by_box(mesh), "This mesh is bounded by box but found not"

    with OmegaHMesh(without_box_mesh_file) as mesh:
        assert not is_bounded_by_box(mesh), "This mesh is not bounded by box but found a box"


