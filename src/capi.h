/**
 *@file capi.h
 *@brief  C API header for Omega_h2csg
 */

#ifndef OMEGAH2CSG_CAPI_H
#define OMEGAH2CSG_CAPI_H

#include <Omega_h_adj.hpp>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct OmegaHLibrary {
  void *pointer;
};

typedef struct OmegaHLibrary OmegaHLibrary;

struct OmegaHMesh {
  void *pointer;
};

typedef struct OmegaHMesh OmegaHMesh;

OmegaHLibrary create_omegah_library();
void destroy_omegah_library(OmegaHLibrary lib);

OmegaHMesh create_omegah_mesh(OmegaHLibrary lib, const char *filename);
void destroy_omegah_mesh(OmegaHMesh mesh);

void print_mesh_info(OmegaHMesh mesh);
int get_num_entities(OmegaHMesh mesh, int dim);
int get_dim(OmegaHMesh mesh);

void kokkos_initialize();
void kokkos_finalize();

void capi_compute_edge_coefficients(OmegaHMesh oh_mesh, int size,
                                    double coefficients[], bool print_debug,
                                    double tol = 1e-6);

int capi_get_number_of_boundary_edges(OmegaHMesh oh_mesh);
void capi_get_boundary_edge_ids(OmegaHMesh oh_mesh, int size, int edge_ids[]);

void capi_get_face_connectivity(OmegaHMesh oh_mesh, int edge_size,
                                double edge_coefficients[], int face_size,
                                int face_connectivity[], bool print_debug,
                                double tol = 1e-6);

void capi_get_all_geometry_info(OmegaHMesh oh_mesh, int n_edges, int n_faces,
                                double edge_coefficients[],
                                int boundary_edges[], int face_connectivity[],
                                bool print_debug, double tol = 1e-6);

bool capi_is_mesh_bounded_by_box(OmegaHMesh oh_mesh);
bool capi_has_boundary_layer(OmegaHMesh oh_mesh);
bool capi_get_mesh_int_tag_array(OmegaHMesh oh_mesh, int dim, const char *name,
                                 int *tag_aray, int size);
void capi_get_cell_bounding_boxes(OmegaHMesh oh_mesh, double *bbox, int size);
void capi_get_cell_volumes(OmegaHMesh oh_mesh, double *volumes, int size);
void capi_get_cell_centroids(OmegaHMesh oh_mesh, double *centroids, int size);
void capi_get_edge_coordinates(OmegaHMesh oh_mesh, double *edge_coords,
                               int size);
// used for test
int capi_get_number_of_edges_inside_wall(OmegaHMesh oh_mesh);

#ifdef __cplusplus
}
#endif

#endif // OMEGAH2CSG_CAPI_H
