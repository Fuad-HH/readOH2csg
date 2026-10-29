"""Native executables: exit codes and outputs."""

import subprocess

import pytest

from omegah2csg import _cli


def _exe(name):
    try:
        return str(_cli._find(name))
    except SystemExit:
        pytest.skip(f"{name} not installed")


def _run(*args):
    return subprocess.run(args, capture_output=True, text=True, check=False)


def test_openmc_cli(assets, tmp_path):
    exe = _exe("convert2openmc")
    out = tmp_path / "geometry.xml"
    assert _run(exe, str(assets / "6elem.osh"), "-o", str(out)).returncode == 0
    assert out.is_file()
    assert _run(exe, str(assets / "6elem.osh"), "-o", str(out)).returncode == 2
    assert (
        _run(exe, str(assets / "6elem.osh"), "-o", str(out), "--force").returncode == 0
    )
    assert _run(exe, "--no-such-flag").returncode == 1
    assert _run(exe, str(tmp_path / "missing.osh")).returncode == 2


def test_degas2_cli(degas2_mesh, assets, tmp_path):
    exe = _exe("convert2degas2")
    aux = tmp_path / "aux"
    aux.mkdir()
    args = [str(degas2_mesh), "-o", str(tmp_path / "g.nc"), "-a", "--aux-dir", str(aux)]
    assert _run(exe, *args).returncode == 0
    assert (aux / "wallfile.txt").is_file()
    assert _run(exe, str(assets / "6elem.osh"), "--check").returncode == 2  # no tags
