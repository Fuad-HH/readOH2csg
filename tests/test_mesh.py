from omegah2csg import OmegaHMesh
from pathlib import Path

parent_directory = Path(__file__).resolve().parent

def test_mesh_loading():
    with OmegaHMesh(parent_directory/f'assets/6elem.osh') as mesh:
        assert mesh.dim == 2
        assert mesh.num_entities(mesh.dim) == 6

def test_multiple_mesh_loading():
    with OmegaHMesh(parent_directory/f'assets/16elem.osh') as mesh:
        assert mesh.dim == 2
        assert mesh.num_entities(mesh.dim) == 16

