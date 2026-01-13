/**
 *@file capi.cpp
 *@brief C API for the library
 *@details This contains only the functions that are needed for C API
 *exclusively. The rest of the functions are in compute_surface.cpp.
 */

#include "capi.h"
#include "compute_surface.h"

#include <Omega_h_array_ops.hpp>
#include <Omega_h_file.hpp>
#include <Omega_h_for.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_mark.hpp>
#include <Omega_h_mesh.hpp>

#include <cassert>

extern "C" void capi_get_all_geometry_info(OmegaHMesh oh_mesh, int n_edges,
                                           int n_faces,
                                           double edge_coefficients[],
                                           int boundary_edges[],
                                           int face_connectivity[],
                                           bool print_debug, const double tol) {
  auto mesh = reinterpret_cast<Omega_h::Mesh *>(oh_mesh.pointer);
  if (n_edges != mesh->nedges()) {
    throw std::runtime_error("Error: size of edge_coefficients array does not "
                             "match number of edges.");
  }
  if (n_faces != mesh->nfaces()) {
    throw std::runtime_error("Error: size of face_connectivity array does not "
                             "match number of faces.");
  }

  // compute edge coefficients
  auto edge_coefficients_view =
      Kokkos::View<double *[6]>("edge_coefficients", mesh->nedges());
  compute_edge_coefficients(*mesh, edge_coefficients_view, print_debug, tol);

  // get boundary edge ids
  Omega_h::LOs boundary_edge_ids = get_boundary_edge_ids(*mesh);

  // compute face connectivity
  Kokkos::View<int *[6]> face_connectivity_view = calculate_face_connectivity(
      *mesh, edge_coefficients_view, print_debug, tol);

  // insert sign with boundary edge ids facing inward
  Omega_h::LOs boundary_edges_ids_with_sign =
      insert_inward_sign_with_boundary_edges(*mesh, boundary_edge_ids,
                                             face_connectivity_view);

  // copy edge coefficients to output array
  auto host_edge_coefficients =
      Kokkos::create_mirror_view(edge_coefficients_view);
  Kokkos::deep_copy(host_edge_coefficients, edge_coefficients_view);
  for (int edge = 0; edge < mesh->nedges(); ++edge) {
    for (int i = 0; i < 6; ++i) {
      edge_coefficients[edge * 6 + i] = host_edge_coefficients(edge, i);
    }
  }

  // copy boundary edge ids to output array
  auto host_boundary_edge_ids_with_sign =
      Omega_h::HostRead<Omega_h::LO>(boundary_edges_ids_with_sign);
  for (int i = 0; i < boundary_edges_ids_with_sign.size(); ++i) {
    boundary_edges[i] = host_boundary_edge_ids_with_sign[i];
  }

  // copy face connectivity to output array
  auto host_face_connectivity =
      Kokkos::create_mirror_view(face_connectivity_view);
  Kokkos::deep_copy(host_face_connectivity, face_connectivity_view);
  for (int face = 0; face < mesh->nfaces(); ++face) {
    for (int i = 0; i < 6; ++i) {
      face_connectivity[face * 6 + i] = host_face_connectivity(face, i);
    }
  }
}

extern "C" void capi_get_face_connectivity(OmegaHMesh oh_mesh, int edge_size,
                                           double edge_coefficients[],
                                           int face_size,
                                           int face_connectivity[],
                                           bool print_debug, const double tol) {

  auto mesh = reinterpret_cast<Omega_h::Mesh *>(oh_mesh.pointer);
  const auto n_faces = mesh->nfaces();
  const auto n_edges = mesh->nedges();
  // copy edge coefficients to Kokkos view
  if (edge_size != mesh->nedges() * 6) {
    throw std::runtime_error(
        "Error: size of edge_coefficients array does not match number of edges "
        "* 6");
  }
  if (face_size != mesh->nfaces() * 6) {
    throw std::runtime_error(
        "Error: size of face_connectivity array does not match number of faces "
        "* 6");
  }

  Kokkos::View<double *[6], Kokkos::DefaultExecutionSpace>
      edge_coefficients_view("edge_coefficients_view", mesh->nedges());
  auto host_edge_efficients =
      Kokkos::create_mirror_view(edge_coefficients_view);

  // TODO remove this by using unmanaged view
  for (int edge = 0; edge < n_edges; ++edge) {
    for (int i = 0; i < 6; ++i) {
      host_edge_efficients(edge, i) = edge_coefficients[edge * 6 + i];
    }
  }
  Kokkos::deep_copy(edge_coefficients_view, host_edge_efficients);

  auto connectivity = calculate_face_connectivity(*mesh, edge_coefficients_view,
                                                  print_debug, tol);

  auto host_connectivity = Kokkos::create_mirror_view(connectivity);
  Kokkos::deep_copy(host_connectivity, connectivity);

  // copy to output array
  for (int face = 0; face < n_faces; ++face) {
    for (int i = 0; i < 6; ++i) {
      face_connectivity[face * 6 + i] = host_connectivity(face, i);
    }
  }
}

extern "C" int capi_get_number_of_boundary_edges(OmegaHMesh oh_mesh) {
  auto mesh = reinterpret_cast<Omega_h::Mesh *>(oh_mesh.pointer);
  const auto exposed_side_marks = Omega_h::mark_exposed_sides(mesh);
  int num_boundary_edges = Omega_h::get_sum(exposed_side_marks);

  return num_boundary_edges;
}

extern "C" void capi_get_boundary_edge_ids(OmegaHMesh oh_mesh, const int size,
                                           int edge_ids[]) {
  auto mesh = reinterpret_cast<Omega_h::Mesh *>(oh_mesh.pointer);
  const auto exposed_sides = get_boundary_edge_ids(*mesh);
  if (size != exposed_sides.size()) {
    throw std::runtime_error("Error: size of edge_ids array does not match "
                             "number of boundary edges");
  }
  const auto host_exposed_sides = Omega_h::HostRead<Omega_h::LO>(exposed_sides);
  for (int i = 0; i < exposed_sides.size(); ++i) {
    edge_ids[i] = host_exposed_sides[i];
  }
}

extern "C" OmegaHLibrary create_omegah_library() {
  const auto lib = new Omega_h::Library();
  return {reinterpret_cast<void *>(lib)};
}

extern "C" void destroy_omegah_library(OmegaHLibrary lib) {
  delete reinterpret_cast<Omega_h::Library *>(lib.pointer);
}

extern "C" OmegaHMesh create_omegah_mesh(OmegaHLibrary lib,
                                         const char *filename) {
  assert(lib.pointer != nullptr);
  auto *library = reinterpret_cast<Omega_h::Library *>(lib.pointer);
  const auto mesh = new Omega_h::Mesh(library);
  Omega_h::binary::read(filename, library->world(), mesh);
  return {reinterpret_cast<void *>(mesh)};
}

extern "C" void destroy_omegah_mesh(OmegaHMesh mesh) {
  delete reinterpret_cast<Omega_h::Mesh *>(mesh.pointer);
}

extern "C" void print_mesh_info(OmegaHMesh mesh) {
  const auto omega_h_mesh = reinterpret_cast<Omega_h::Mesh *>(mesh.pointer);
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
  const auto omega_h_mesh = reinterpret_cast<Omega_h::Mesh *>(mesh.pointer);
  return omega_h_mesh->nents(dim);
}

extern "C" int get_dim(OmegaHMesh mesh) {
  assert(mesh.pointer != nullptr);
  const auto omega_h_mesh = reinterpret_cast<Omega_h::Mesh *>(mesh.pointer);
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
                                               const bool print_debug,
                                               const double tol) {
  auto mesh = reinterpret_cast<Omega_h::Mesh *>(oh_mesh.pointer);
  const auto n_edges = mesh->nedges();
  auto edge_coefficients_view =
      Kokkos::View<double *[6]>("edge_coefficients_view", n_edges);

  compute_edge_coefficients(*mesh, edge_coefficients_view, print_debug, tol);
  auto host_edge_coefficients_view =
      Kokkos::create_mirror_view(edge_coefficients_view);
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

extern "C" bool capi_is_mesh_bounded_by_box(OmegaHMesh oh_mesh) {
  // check based on if the is_on_wall array is present
  auto mesh = reinterpret_cast<Omega_h::Mesh *>(oh_mesh.pointer);
  return mesh->has_tag(Omega_h::VERT, "isOnWall");
}

extern "C" bool capi_has_boundary_layer(OmegaHMesh oh_mesh) {
  auto mesh = reinterpret_cast<Omega_h::Mesh *>(oh_mesh.pointer);
  return mesh->has_tag(Omega_h::FACE, "offset_face");
}

extern "C" bool capi_get_mesh_int_tag_array(OmegaHMesh oh_mesh, const int dim,
                                            const char *name, int *tag_aray,
                                            const int size) {
  auto mesh = reinterpret_cast<Omega_h::Mesh *>(oh_mesh.pointer);
  if (!mesh->has_tag(dim, name)) {
    return false;
  }
  auto tag = mesh->get_tag<Omega_h::LO>(dim, name);
  if (tag->type() != OMEGA_H_I32) {
    throw std::runtime_error("Error: tag type is not int32.");
  }

  Omega_h::Read<Omega_h::LO> tag_data = tag->array();
  if (tag_data.size() != size) {
    throw std::runtime_error(
        "Error: size of tag array does not match provided size.");
  }
  auto host_tag_data = Omega_h::HostRead<Omega_h::LO>(tag_data);
  for (int i = 0; i < size; ++i) {
    tag_aray[i] = host_tag_data[i];
  }

  return true;
}

extern "C" void capi_get_cell_bounding_boxes(OmegaHMesh oh_mesh, double *bbox,
                                             const int size) {
  auto mesh = reinterpret_cast<Omega_h::Mesh *>(oh_mesh.pointer);
  const int n_elements = mesh->nelems();
  if (size != n_elements * 4) { // 2D bbox: xmin, ymin, xmax, ymax
    throw std::runtime_error(
        "Error: size of bbox array does not match number of elements * 4.");
  }
  const auto coords = mesh->coords();
  const auto face2node = mesh->ask_elem_verts();

  Omega_h::Write<double> bbox_view(n_elements * 4, "bbox_view");

  auto calculate_bbox = OMEGA_H_LAMBDA(const Omega_h::LO elem) {
    double xmin = INT64_MAX;
    double xmax = INT64_MIN;
    double ymin = INT64_MAX;
    double ymax = INT64_MIN;

    for (int vert = 0; vert < 3; ++vert) {
      const int node = face2node[elem * 3 + vert];
      auto elem_coords = Omega_h::get_vector<2>(coords, node);
      if (elem_coords[0] < xmin)
        xmin = elem_coords[0];
      if (elem_coords[0] > xmax)
        xmax = elem_coords[0];
      if (elem_coords[1] < ymin)
        ymin = elem_coords[1];
      if (elem_coords[1] > ymax)
        ymax = elem_coords[1];
    }

    bbox_view[elem * 4 + 0] = xmin;
    bbox_view[elem * 4 + 1] = ymin;
    bbox_view[elem * 4 + 2] = xmax;
    bbox_view[elem * 4 + 3] = ymax;
  };
  Omega_h::parallel_for(n_elements, calculate_bbox, "calculate_bbox");

  Omega_h::HostWrite host_bbox_view(bbox_view);
  for (int i = 0; i < host_bbox_view.size(); ++i) {
    bbox[i] = host_bbox_view[i];
  }
}

extern "C" void capi_get_cell_volumes(OmegaHMesh oh_mesh, double *volumes,
                                      int size) {
  auto mesh = reinterpret_cast<Omega_h::Mesh *>(oh_mesh.pointer);
  const int n_elements = mesh->nelems();
  if (size != n_elements) {
    throw std::runtime_error(
        "Error: size of volumes array does not match number of elements.");
  }

  const auto coords = mesh->coords();
  const auto elem2node = mesh->ask_elem_verts();
  Omega_h::Write<double> volumes_v(n_elements, "volumes_view");

  auto compute_volume = OMEGA_H_LAMBDA(const Omega_h::LO elem) {
    const auto v0 = Omega_h::get_vector<2>(coords, elem2node[elem * 3 + 0]);
    const auto v1 = Omega_h::get_vector<2>(coords, elem2node[elem * 3 + 1]);
    const auto v2 = Omega_h::get_vector<2>(coords, elem2node[elem * 3 + 2]);

    const auto centroid = (v0 + v1 + v2) / 3.0;

    const double area = 0.5 * Kokkos::abs(Omega_h::cross(v1 - v0, v2 - v0));
    volumes_v[elem] = 2 * Kokkos::numbers::pi_v<double> * area * centroid[0];
  };
  Omega_h::parallel_for(n_elements, compute_volume, "compute_volumes");

  Omega_h::HostWrite host_volumes_v(volumes_v);
  for (int i = 0; i < host_volumes_v.size(); ++i) {
    volumes[i] = host_volumes_v[i];
  }
}

extern "C" void capi_get_cell_centroids(OmegaHMesh oh_mesh, double *centroids,
                                        int size) {
  auto mesh = reinterpret_cast<Omega_h::Mesh *>(oh_mesh.pointer);
  const int n_elements = mesh->nelems();
  if (size != 2 * n_elements) {
    throw std::runtime_error(
        "Error: size of volumes array does not match number of elements.");
  }

  const auto coords = mesh->coords();
  const auto elem2node = mesh->ask_elem_verts();

  Omega_h::Write<double> centroids_v(2 * n_elements, "centroids_view");

  auto compute_centroids = OMEGA_H_LAMBDA(const Omega_h::LO elem) {
    const auto v0 = Omega_h::get_vector<2>(coords, elem2node[elem * 3 + 0]);
    const auto v1 = Omega_h::get_vector<2>(coords, elem2node[elem * 3 + 1]);
    const auto v2 = Omega_h::get_vector<2>(coords, elem2node[elem * 3 + 2]);

    const auto centroid = (v0 + v1 + v2) / 3.0;

    centroids_v[2 * elem + 0] = centroid[0];
    centroids_v[2 * elem + 1] = centroid[1];
  };
  Omega_h::parallel_for(n_elements, compute_centroids, "compute_volumes");

  Omega_h::HostWrite centroids_v_host(centroids_v);
  for (int i = 0; i < centroids_v_host.size(); ++i) {
    centroids[i] = centroids_v_host[i];
  }
}

extern "C" void capi_get_edge_coordinates(OmegaHMesh oh_mesh,
                                          double *edge_coords, int size) {
  auto mesh = reinterpret_cast<Omega_h::Mesh *>(oh_mesh.pointer);
  const int n_edges = mesh->nedges();
  int array_size = 4 * n_edges; // each edge has 4 coordinates (x0, y0, x1, y1)

  if (size != array_size) {
    throw std::runtime_error(
        "Error: size of edge_coords array does not match number of edges * 4.");
  }

  const auto coords = mesh->coords();
  const auto edge2node = mesh->ask_down(Omega_h::EDGE, Omega_h::VERT).ab2b;

  Omega_h::Write<double> edge_coords_v(array_size, "edge_coords_v");
  auto compute_edge_coords = OMEGA_H_LAMBDA(const Omega_h::LO edge) {
    const auto v0 = Omega_h::get_vector<2>(coords, edge2node[edge * 2 + 0]);
    const auto v1 = Omega_h::get_vector<2>(coords, edge2node[edge * 2 + 1]);

    edge_coords_v[edge * 4 + 0] = v0[0];
    edge_coords_v[edge * 4 + 1] = v0[1];
    edge_coords_v[edge * 4 + 2] = v1[0];
    edge_coords_v[edge * 4 + 3] = v1[1];
  };
  Omega_h::parallel_for(n_edges, compute_edge_coords, "compute_edge_coords");

  Omega_h::HostWrite edge_coords_v_host(edge_coords_v);
  for (int i = 0; i < array_size; ++i) {
    edge_coords[i] = edge_coords_v_host[i];
  }
}

extern "C" int capi_get_number_of_edges_inside_wall(OmegaHMesh oh_mesh) {
  auto mesh = reinterpret_cast<Omega_h::Mesh *>(oh_mesh.pointer);
  if (!mesh->has_tag(Omega_h::FACE, "offset_face")) {
    throw std::runtime_error(
        "Error: mesh does not have 'offset_face' tag on faces.");
  }

  const auto offset_face_tag =
      mesh->get_tag<Omega_h::LO>(Omega_h::FACE, "offset_face")->array();

  Omega_h::LO num_edges_inside_wall = 0;
  const auto edge2face = mesh->ask_up(Omega_h::EDGE, Omega_h::FACE).ab2b;
  const auto edge2face_offset = mesh->ask_up(Omega_h::EDGE, Omega_h::FACE).a2ab;

  auto count_edges = KOKKOS_LAMBDA(const Omega_h::LO &edge, int &count) {
    const int n_adj_faces = edge2face_offset[edge + 1] - edge2face_offset[edge];
    if (n_adj_faces > 1) {
      if (offset_face_tag[edge2face[edge2face_offset[edge]]] == 0 &&
          offset_face_tag[edge2face[edge2face_offset[edge] + 1]] == 0) {
        count += 1;
      }
    }
  };
  Kokkos::parallel_reduce("count edges", mesh->nedges(), count_edges,
                          num_edges_inside_wall);

  return num_edges_inside_wall;
}

extern "C" void capi_get_edge_to_face_connectivity(OmegaHMesh oh_mesh,
                                                   int *faces, int size) {
  auto mesh = reinterpret_cast<Omega_h::Mesh *>(oh_mesh.pointer);
  const int n_edges = mesh->nedges();
  if (size != n_edges * 2) {
    throw std::runtime_error(
        "Error: size of faces array does not match number of edges * 2.");
  }

  const auto edge2face_adj = mesh->ask_up(Omega_h::EDGE, Omega_h::FACE);
  const auto edge2face = Omega_h::HostRead(edge2face_adj.ab2b);
  const auto edge2face_offset = Omega_h::HostRead(edge2face_adj.a2ab);

  for (int edge = 0; edge < n_edges; ++edge) {
    const int n_adj_faces = edge2face_offset[edge + 1] - edge2face_offset[edge];
    assert(n_adj_faces == 1 || n_adj_faces == 2);

    // first adjacent face
    faces[edge * 2 + 0] = edge2face[edge2face_offset[edge]];
    // second adjacent face
    if (n_adj_faces == 2) {
      faces[edge * 2 + 1] = edge2face[edge2face_offset[edge] + 1];
    } else {
      faces[edge * 2 + 1] = -1; // no second adjacent face
    }
  }
}

extern "C" void capi_get_wall_edge_ids(OmegaHMesh oh_mesh, int *edge_ids,
                                       int size) {
  auto mesh = reinterpret_cast<Omega_h::Mesh *>(oh_mesh.pointer);
  const Omega_h::LOs wall_edge_ids = get_wall_edge_ids(mesh);
  auto host_wall_edge_ids = Omega_h::HostRead(wall_edge_ids);

  if (size != wall_edge_ids.size()) {
    throw std::runtime_error(
        "Error: size of edge_ids array does not match number of wall edges.");
  }

  for (int i = 0; i < wall_edge_ids.size(); ++i) {
    edge_ids[i] = host_wall_edge_ids[i];
  }
}

extern "C" void capi_get_wall_adjacent_triangles(OmegaHMesh oh_mesh,
                                                 int *triangles, int size) {
  auto mesh = reinterpret_cast<Omega_h::Mesh *>(oh_mesh.pointer);

  Omega_h::LOs wall_adjacent_triangles = get_wall_adjacent_triangles(mesh);
  printf("Number of wall adjacent triangles: %d\n",
         wall_adjacent_triangles.size());
  if (size != wall_adjacent_triangles.size()) {
    throw std::runtime_error("Error: size of triangles array does not match "
                             "number of wall adjacent triangles.");
  }

  const auto host_wall_adjacent_triangles =
      Omega_h::HostRead(wall_adjacent_triangles);
  for (int i = 0; i < wall_adjacent_triangles.size(); ++i) {
    triangles[i] = host_wall_adjacent_triangles[i];
  }
}
