/**
 *@file capi.cpp
 *@brief C API for the library
 *@details This contains only the functions that are needed for C API
 *exclusively. The rest of the functions are in compute_surface.cpp.
 */

#include "capi.h"
#include "compute_surface.h"

#include <Omega_h_file.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_mesh.hpp>
#include <cassert>

extern "C" OmegaHLibrary create_omegah_library() {
  const auto lib = new Omega_h::Library();
  return static_cast<OmegaHLibrary>(lib);
}

extern "C" void destroy_omegah_library(OmegaHLibrary lib) {
  const auto library = static_cast<Omega_h::Library *>(lib.pointer);
  delete library;
}

extern "C" OmegaHMesh create_omegah_mesh(OmegaHLibrary lib,
                                         const char *filename) {
  assert(lib.pointer != nullptr);
  auto *library = static_cast<Omega_h::Library *>(lib.pointer);
  const auto mesh = new Omega_h::Mesh(library);
  Omega_h::binary::read(filename, library->world(), mesh);
  return static_cast<OmegaHMesh>(mesh);
}

extern "C" void destroy_omegah_mesh(OmegaHMesh mesh) {
  const auto omega_h_mesh = static_cast<Omega_h::Mesh *>(mesh.pointer);
  delete omega_h_mesh;
}

extern "C" void print_mesh_info(OmegaHMesh mesh) {
  const auto omega_h_mesh = static_cast<Omega_h::Mesh *>(mesh.pointer);
  // Print some basic information about the mesh
  // Number of vertices, edges, faces, and elements
  const int num_vertices = omega_h_mesh->nverts();
  const int num_edges = omega_h_mesh->nedges();
  const int num_faces = omega_h_mesh->nfaces();
  const int num_elements = omega_h_mesh->nelems();
  printf("Mesh Information:\n");
  printf("\tNumber of vertices: %d\n", num_vertices);
  printf("\tNumber of edges: %d\n", num_edges);
  printf("\tNumber of faces: %d\n", num_faces);
  printf("\tNumber of elements: %d\n", num_elements);
  printf("\n");
}

extern "C" int get_num_entities(OmegaHMesh mesh, int dim) {
  assert(mesh.pointer != nullptr);
  const auto omega_h_mesh = static_cast<Omega_h::Mesh *>(mesh.pointer);
  return omega_h_mesh->nents(dim);
}

extern "C" int get_dim(OmegaHMesh mesh) {
  assert(mesh.pointer != nullptr);
  const auto omega_h_mesh = static_cast<Omega_h::Mesh *>(mesh.pointer);
  return omega_h_mesh->dim();
}

extern "C" void kokkos_initialize() {
  if (!Kokkos::is_initialized()) {
    Kokkos::initialize();
  }
}

extern "C" void kokkos_finalize() {
  if (Kokkos::is_initialized()) {
    if (!Kokkos::is_finalized()) {
      Kokkos::Tools::finalize();
    }
  }
}

extern "C" void capi_compute_edge_coefficients(OmegaHMesh oh_mesh, int size,
                                               double coefficients[],
                                               const bool print_debug) {
  auto mesh = static_cast<Omega_h::Mesh *>(oh_mesh.pointer);
  const auto n_edges = mesh->nedges();
  auto edge_coefficients_view = Kokkos::View<double *[6]>(
      "edge_coefficients_view", n_edges);

  compute_edge_coefficients(*mesh, edge_coefficients_view, print_debug);
  auto host_edge_coefficients_view = Kokkos::create_mirror_view(
      edge_coefficients_view);
  Kokkos::deep_copy(host_edge_coefficients_view, edge_coefficients_view);

  if (size != host_edge_coefficients_view.size()) {
    throw std::runtime_error(
        "Error: size of coefficients array does not match number of edges * 6");
  }

  // TODO Use the pointer as host copy
  for (int edge = 0; edge < host_edge_coefficients_view.extent(0); ++edge) {
    for (int i = 0; i < 6; ++i) {
      coefficients[edge * 6 + i] = host_edge_coefficients_view(edge, i);
    }
  }
}