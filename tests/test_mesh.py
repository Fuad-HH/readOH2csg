from omegah2csg import OmegaHMesh
from pathlib import Path

parent_directory = Path(__file__).resolve().parent


def test_mesh_loading():
    with OmegaHMesh(parent_directory / "assets/6elem.osh") as mesh:
        assert mesh.dim == 2
        assert mesh.num_entities(mesh.dim) == 6


def test_multiple_mesh_loading():
    with OmegaHMesh(parent_directory / "assets/16elem.osh") as mesh:
        assert mesh.dim == 2
        assert mesh.num_entities(mesh.dim) == 16


def test_get_node_coordinates():
    with OmegaHMesh(parent_directory / "assets/6elem.osh") as mesh:
        node_coordinates = mesh.get_node_coordinates()
        n_nodes = mesh.num_entities(0)
        assert node_coordinates.shape[0] == n_nodes * 2

        print("\nNode coordinates:")
        for i in range(n_nodes):
            print(
                f"Node {i}: {float(node_coordinates[i * 2]), float(node_coordinates[i * 2 + 1])}"
            )
