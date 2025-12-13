/**
 *@file capi.h
 *@brief  C API header for Omega_h2csg
 */

#ifndef OMEGAH2CSG_CAPI_H
#define OMEGAH2CSG_CAPI_H
#include <Omega_h_adj.hpp>

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

#ifdef __cplusplus
}
#endif

#endif // OMEGAH2CSG_CAPI_H
