from ..openmcGeometry import create_openmc_geometry
from ..OmegaHMesh import OmegaHMesh
import openmc

def convert2openmc(filename, tol):
    assert filename.endswith(".osh")
    assert (tol < 1e-6) and (tol > 0.0)

    with OmegaHMesh(filename) as mesh:
        universe = create_openmc_geometry(mesh=mesh, tol=tol)
        geom = openmc.Geometry(universe)
        geom.export_to_xml()
