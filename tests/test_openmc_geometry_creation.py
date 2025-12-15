import openmc

import pandas as pd
from omegah2csg import OmegaHMesh
from omegah2csg import get_edge_coefficients
from omegah2csg import get_boundary_edge_ids


from pathlib import Path



def test_edge_and_face_coefficients():
    parent_directory = Path(__file__).resolve().parent

    with OmegaHMesh(parent_directory / f'assets/6elem.osh') as mesh:
        edge_coefficients = get_edge_coefficients(mesh)
        boundary_edge_ids = get_boundary_edge_ids(mesh)
        assert mesh.num_entities(1) == 13
        print(pd.DataFrame(edge_coefficients))

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
    assert edge_coefficients.shape[1] == 6
    assert boundary_edge_ids.shape[0] == 8

