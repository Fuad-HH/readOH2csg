import numpy as np
import openmc
from omegah2csg import get_line_equation
from omegah2csg import create_openmc_surface
from omegah2csg import is_horizontal, is_vertical, is_cone

def test_is_horizontal_vertical_cone():
    p1 = (1,1)
    p2 = (2,3)

    assert not is_horizontal(p1, p2)
    assert not is_vertical(p1, p2)
    assert is_cone(p1, p2)

    p1 = (1, 1)
    p3 = (1.0, -10.0)
    assert not is_horizontal(p1, p3)
    assert is_vertical(p1, p3)
    assert not is_cone(p1, p3)

    p2 = (2, 3)
    p4 = (-1.0, 3)
    assert is_horizontal(p2, p4)
    assert not is_vertical(p2, p4)
    assert not is_cone(p2, p4)

def test_get_line_equation():
    p1 = (1,1)
    p2 = (2,2)

    m,c,up = get_line_equation(p1,p2)
    assert np.isclose(m,1)
    assert np.isclose(c,0)
    assert up == True

def test_create_surface_cone():
    p1 = (1,1)
    p2 = (2,2)

    cone = create_openmc_surface(p1,p2, 1e-6)
    assert type(cone) == openmc.model.ZConeOneSided
    disambiguation_surface = cone.plane
    assert np.isclose(disambiguation_surface.z0, 0.0)
    assert cone.up == True

    p1 = (1,1)
    p2 = (2,3)

    cone = create_openmc_surface(p1,p2, 1e-6)
    assert type(cone) == openmc.model.ZConeOneSided
    disambiguation_surface = cone.plane
    assert np.isclose(disambiguation_surface.z0, -1.0)
    assert cone.up == True


    p1 = (1,1)
    p2 = (2,-1)

    cone = create_openmc_surface(p1,p2, 1e-6)
    assert type(cone) == openmc.model.ZConeOneSided
    disambiguation_surface = cone.plane
    assert np.isclose(disambiguation_surface.z0, 3.0)
    assert cone.up == False
