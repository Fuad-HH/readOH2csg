from ..openmcGeometry import create_openmc_geometry
from ..OmegaHMesh import OmegaHMesh
import openmc

def convert2openmc(filename):
    assert filename.endswith(".osh")

    with OmegaHMesh(filename) as mesh:
        universe = create_openmc_geometry(mesh)
        geom = openmc.Geometry(universe)
        geom.export_to_xml()
