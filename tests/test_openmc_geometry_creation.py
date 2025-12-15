import numpy as np
import openmc

import pandas as pd
from omegah2csg import OmegaHMesh
from omegah2csg import get_edge_coefficients
from omegah2csg import get_boundary_edge_ids
from omegah2csg import get_face_connectivity
from omegah2csg import get_all_geometry_info
from omegah2csg import create_openmc_geometry
from omegah2csg import read_edge_coefficients_from_file
from omegah2csg import read_face_connectivity_from_file

from pathlib import Path


parent_directory = Path(__file__).resolve().parent

def test_read_from_file():
    edge_coefficient_filename = parent_directory / 'assets/6elem_mesh_files/edge-coefficients.dat'
    [edge_coefficients_file, boundary_edge_ids_file] = read_edge_coefficients_from_file(edge_coefficient_filename)

    face_connectivity_filename = parent_directory / 'assets/6elem_mesh_files/face-edge-connectivity.dat'
    face_connectivity_file = read_face_connectivity_from_file(face_connectivity_filename)

    with OmegaHMesh(parent_directory / f'assets/6elem.osh') as mesh:
        [edge_coefficients_api, boundary_edge_ids_api, face_connectivity_api] = get_all_geometry_info(mesh)

    # compare boundary edges
    assert boundary_edge_ids_api.ndim == 1
    assert boundary_edge_ids_file.ndim == 1
    # 8 boundary edges
    assert boundary_edge_ids_api.shape[0] == boundary_edge_ids_file.shape[0] == 8

    # edge ids are the same
    for i in range(boundary_edge_ids_api.shape[0]):
        assert boundary_edge_ids_api[i] == boundary_edge_ids_file[i]

    assert edge_coefficients_api.ndim == edge_coefficients_file.ndim == 2
    assert edge_coefficients_api.shape[0] == edge_coefficients_file.shape[0] == 13
    assert edge_coefficients_api.shape[1] == edge_coefficients_file.shape[1] == 6

    # first 5 coefficients are the same
    for i in range(edge_coefficients_api.shape[0]):
        for j in range(edge_coefficients_file.shape[1]-1):
            assert np.isclose(edge_coefficients_api[i, j], edge_coefficients_file[i, j])

    # check face connectivity
    assert face_connectivity_file.shape[0] == face_connectivity_api.shape[0] == 6
    assert face_connectivity_file.shape[1] == face_connectivity_api.shape[1] == 6

    for i in range(face_connectivity_file.shape[0]):
        for j in range(face_connectivity_file.shape[1]):
            assert face_connectivity_api[i, j] == face_connectivity_file[i, j]



def test_all_gemetry_info():
    with OmegaHMesh(parent_directory / f'assets/6elem.osh') as mesh:
        [edge_coefficients, boundary_edge_ids, face_connctivity] = get_all_geometry_info(mesh)
        print(pd.DataFrame(edge_coefficients))
        print(pd.DataFrame(boundary_edge_ids))
        print(pd.DataFrame(face_connctivity))

    assert edge_coefficients.shape[0] == 13
    assert edge_coefficients.shape[1] == 6

    assert boundary_edge_ids.shape[0] == 8

    assert face_connctivity.shape[0] == 6
    assert face_connctivity.shape[1] == 6

def test_create_openmc_universe():
    with OmegaHMesh(parent_directory / f'assets/6elem.osh') as mesh:
        universe = create_openmc_geometry(mesh)
        geom = openmc.Geometry(universe)
        geom.export_to_xml()

def test_edge_and_face_coefficients():
    with OmegaHMesh(parent_directory / f'assets/6elem.osh') as mesh:
        edge_coefficients = get_edge_coefficients(mesh)
        boundary_edge_ids = get_boundary_edge_ids(mesh)
        n_faces = mesh.num_entities(2)
        n_edges = mesh.num_entities(1)

        print("\n\nEdge coefficients:")
        print(pd.DataFrame(edge_coefficients))

        print("\n\nFace connectivity:")
        face_connectivity_edge_given = get_face_connectivity(mesh, edge_coefficients)
        print(pd.DataFrame(face_connectivity_edge_given))
        face_connectivity_edge_not_given = get_face_connectivity(mesh)

    top_bottom_flag = edge_coefficients[:, 5]
    edges = []
    intersections = edge_coefficients[:, 3]
    m2 = edge_coefficients[:, 1]
    z2 = edge_coefficients[:, 2]
    neg_c = edge_coefficients[:, 4]

    for i in range(len(intersections)):

        if abs(m2[i]) < 1e-10:  # zplane
            #print("Zplane - id ", int(data[i][0]))
            edges.append(openmc.ZPlane(z0=-neg_c[i]))
        elif abs(z2[i] + 1) > 1e-10:  # not a cone
            #print("Quad - id ", int(data[i][0]))
            edges.append(openmc.Quadric(a=m2[i], b=m2[i], c=z2[i], j=intersections[i], k=neg_c[i]))
        else:  # cone
            #print("Cone - id ", int(data[i][0]), " flag ", top_bottom_flag[i])
            edges.append(openmc.model.ZConeOneSided(z0=intersections[i], r2=1.0 / abs(m2[i]),
                                                    up=True if top_bottom_flag[i] == 1 else False))

    assert edge_coefficients.shape[0] == 13
    assert n_edges == 13
    assert edge_coefficients.shape[1] == 6
    assert boundary_edge_ids.shape[0] == 8

    print(f"\n\nBoundary edge ids: {boundary_edge_ids}")

    # assert that both face connectivities are the same
    assert face_connectivity_edge_given.shape[0] == n_faces
    assert face_connectivity_edge_given.shape[1] == 6

    assert face_connectivity_edge_given.shape[0] == face_connectivity_edge_not_given.shape[0]
    assert face_connectivity_edge_given.shape[1] == face_connectivity_edge_not_given.shape[1]

    for i in range (face_connectivity_edge_given.shape[0]):
        for j in range(face_connectivity_edge_given.shape[1]):
            assert face_connectivity_edge_given[i, j] == face_connectivity_edge_not_given[i, j]


