"""DEGAS2 path, checked against the saved gold geometry files."""

import numpy as np
import pytest

import omegah2csg as o2c

nc = pytest.importorskip("netCDF4")


@pytest.fixture(scope="module")
def gold(assets):
    case = assets / "simple_degas2_case"
    with (
        nc.Dataset(case / "gold-geometry.nc") as a,
        nc.Dataset(case / "definegeometry2d-geometry.nc") as b,
    ):
        yield [{k: a[k][:] for k in a.variables}, {k: b[k][:] for k in b.variables}]


def test_csg_wall_data(degas2_mesh):
    g = o2c.compute_csg(degas2_mesh)
    assert g.interior_edge_count == 14
    assert np.sort(g.wall_faces[:, 0]).tolist() == [0, 2, 3, 4, 10, 10, 11, 11]
    assert np.sort(g.wall_faces[:, 1]).tolist() == [12, 15, 17, 18, 21, 22, 24, 27]


def test_zones_match_gold(degas2_mesh, gold):
    d = o2c.compute_degas2(degas2_mesh)
    nplasma = int(d["zone_type_num"][1])
    for ref in gold:
        for name in ("zone_volume", "zone_min", "zone_max", "zone_center"):
            np.testing.assert_allclose(
                d[name][:nplasma], ref[name][:nplasma], atol=1e-5
            )
        np.testing.assert_allclose(
            d["universal_cell_vol"], ref["universal_cell_vol"], rtol=1e-5
        )


def test_written_file_matches_gold(degas2_mesh, gold, tmp_path):
    options = o2c.Degas2Options()
    options.target_recyc_coef = float(gold[0]["target_recyc_coef"][1])
    o2c.convert_to_degas2(degas2_mesh, _with_output(options, tmp_path / "g.nc"))
    with nc.Dataset(tmp_path / "g.nc") as out:
        assert out.data_model == "NETCDF3_64BIT_OFFSET"
        assert set(out.variables) == set(gold[0])
        for name, ref in gold[0].items():
            if ref.dtype.kind != "S":
                np.testing.assert_allclose(out[name][:], ref, err_msg=name)


def test_edited_arrays_are_written(degas2_mesh, tmp_path):
    d = o2c.compute_degas2(degas2_mesh)
    d["target_temperature"][1:] = 1.0
    o2c.write_degas2(d, tmp_path / "g.nc")
    with nc.Dataset(tmp_path / "g.nc") as out:
        assert np.all(out["target_temperature"][1:] == 1.0)
    d["zone_volume"] = np.zeros(3)
    with pytest.raises(ValueError, match="zone_ind"):
        o2c.write_degas2(d, tmp_path / "bad.nc")


def test_aux_files(degas2_mesh, tmp_path):
    options = _with_output(o2c.Degas2Options(), tmp_path / "g.nc")
    options.aux_files = True
    o2c.convert_to_degas2(degas2_mesh, options)
    for name in ("wallfile.txt", "plasmafile.txt", "sourcefile.txt"):
        assert (tmp_path / name).stat().st_size > 0


def _with_output(options, path):
    options.output = str(path)
    return options
