"""OpenMC path, checked against the saved 0.0.x results for the 6-element mesh."""

import numpy as np
import pytest

import omegah2csg as o2c

openmc = pytest.importorskip("openmc")


def read_edge_coefficients(path):
    """Saved 0.0.x file: `id c0..c4` per edge, then boundary-edge count and ids."""
    lines = path.read_text().splitlines()
    coeffs = np.array([[float(x) for x in ln.split()[1:]] for ln in lines[:-2]])
    signed_boundary = np.array(lines[-1].split(), dtype=int)
    return coeffs, signed_boundary


def read_face_connectivity(path):
    """Saved 0.0.x file: `face e0 s0 e1 s1 e2 s2` per face."""
    return np.loadtxt(path, dtype=int)[:, 1:]


def test_6elem_matches_saved_results(assets):
    g = o2c.compute_csg(assets / "6elem.osh")
    coeffs, signed_boundary = read_edge_coefficients(
        assets / "6elem_mesh_files/edge-coefficients.dat"
    )
    conn = read_face_connectivity(
        assets / "6elem_mesh_files/face-edge-connectivity.dat"
    )

    assert (g.nedges, g.nfaces) == (13, 6)
    np.testing.assert_allclose(g.surface_coeffs[:, :5], coeffs)
    np.testing.assert_array_equal(g.face_edges, conn[:, 0::2])
    np.testing.assert_array_equal(g.face_sides, conn[:, 1::2])
    # 0.0.x stored edge_id * side, which loses the sign of edge 0
    np.testing.assert_array_equal(g.boundary_edges * g.boundary_sides, signed_boundary)


@pytest.mark.parametrize(
    ("mesh", "tol", "counts"),
    [("6elem.osh", 1e-10, (0, 3, 10)), ("16elem.osh", 1e-4, (2, 4, 24))],
)
def test_surface_type_counts(assets, mesh, tol, counts):
    options = o2c.OpenMCOptions()
    options.tol = tol
    g = o2c.compute_csg(assets / mesh, options)
    found = tuple(
        int(np.sum(g.surface_type == int(t)))
        for t in (
            o2c.SurfaceType.ZPlane,
            o2c.SurfaceType.ZCylinder,
            o2c.SurfaceType.ZCone,
        )
    )
    assert found == counts


@pytest.mark.filterwarnings("ignore::openmc.IDWarning")  # same ids loaded twice
def test_xml_and_python_builder_agree(assets, tmp_path):
    g = o2c.compute_csg(assets / "6elem.osh")
    o2c.write_openmc_xml(g, tmp_path / "geometry.xml")
    from_xml = openmc.Geometry.from_xml(
        tmp_path / "geometry.xml", materials=openmc.Materials()
    )
    built = o2c.to_openmc_universe(g)
    xml_cells = {c.id: str(c.region) for c in from_xml.get_all_cells().values()}
    py_cells = {c.id: str(c.region) for c in built.cells.values()}
    assert xml_cells == py_cells
    reflective = {
        s.id
        for s in from_xml.get_all_surfaces().values()
        if s.boundary_type == "reflective"
    }
    assert reflective == set(g.boundary_edges + 1)


def test_edits_reach_the_xml(assets, tmp_path):
    g = o2c.compute_csg(assets / "6elem.osh")
    g.cell_material[:] = 7
    o2c.write_openmc_xml(g, tmp_path / "geometry.xml")
    assert 'material="7"' in (tmp_path / "geometry.xml").read_text()
    g.face_edges = np.full((g.nfaces, 3), 999)
    with pytest.raises(ValueError, match="face"):
        o2c.write_openmc_xml(g, tmp_path / "bad.xml")
