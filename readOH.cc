#include <cmath>
#include <iostream>
#include <vector>
#include <string>
#include <Kokkos_Core.hpp>

#include "Omega_h_file.hpp"
#include "Omega_h_for.hpp"
#include "Omega_h_mesh.hpp"

/**
 * The equation of the line is:
 *     y = mx + c
 * where,
 *     m = (z2 - z1) / (x2 - x1)
 * and
 *     c = z2 - m * x2
 *
 * When this line revolves around the z axis, it will form a cone. The equation of the cone will be:
 *     x^2 + y^2 = r^2
 * where, r = x (x is the distance from the z axis). Therefore, from the equation of the line,
 *     x = (z - c) / m
 * Substituting this in the equation of the cone, we get:
 *     x^2 + y^2 = (1 / m^2) * (z - c)^2
 *
 * Expanding and simplifying, we get:
 *     m^2 * x^2 + m^2 * y^2 - z^2 + 2 * c * z - c^2 = 0
 *
 * This equation needs to be generalized for any line in the xz plane. It will not work for cylinders as m will be undefined.
 * There is one way of avoiding this by checking if x1 = x2. If it is, then use `ZCylinder` instead of `Quadratic` for the face.
 */
std::vector<double> compute_coefficients(Omega_h::Vector<2> vert1, Omega_h::Vector<2> vert2);

/*!
 * \brief Read the mesh file and go to each vertex to get its coordinates
*/
int main(int argc, char** argv) {
  // calls MPI_init(&argc, &argv) and Kokkos::initialize(argc, argv)
  auto lib = Omega_h::Library(&argc, &argv);
  // encapsulates many MPI functions, e.g., world.barrier()
  const auto world = lib.world();
  // create a distributed mesh object, calls mesh.balance() and then returns it
  Omega_h::Mesh mesh(&lib);

  // read the mesh filename
  std::string mesh_filename = argv[1];
  Omega_h::binary::read(mesh_filename, world, &mesh);
  // ***************** Mesh reading is done ***************** //

  // *********** Read all the edges of the mesh *********** //
  // kokkos view to store nedges * 3 doubles : m^2, 2c, -c^2
  auto edge_coeffs_view = Kokkos::View<double*[4]>("edge_coeffs", mesh.nedges());

  // get the adjacency for the edges
  auto edge2vert = mesh.get_adj(1, 0);
  auto edgeOffsets = edge2vert.a2ab;
  auto edgeVertices = edge2vert.ab2b;

  // * Step 1: loop over all edges and print the associated vertices
  for (Omega_h::LO i = 0; i < mesh.nedges(); ++i) {
    //auto offsets = edge2vert.a2ab(i); // no function a2ab in Omega_h::Adj
    auto vert1 = edgeVertices[2*i];
    auto vert2 = edgeVertices[2*i + 1];
    auto v1coords = Omega_h::get_vector<2>(mesh.coords(), vert1);
    auto v2coords = Omega_h::get_vector<2>(mesh.coords(), vert2);

    // print the coordinates of the vertices
    std::cout << "Edge " << i << " connects vertices " << v1coords[0] << 
            " " << v1coords[1] << " and " << v2coords[0] << " " << v2coords[1] << "\n";
    
    // coefficient vector of doubles of size 3
    std::vector<double> edge_coeffs(4, 0.0);

    if (std::abs(v1coords[0] - v2coords[0]) < 1e-10) {
      // cylinder surface: x^2 + y^2 - r^2 = 0
      edge_coeffs = {1.0, 0.0, -v1coords[0] * v1coords[0]};
    } else if (std::abs(v1coords[1] - v2coords[1]) < 1e-10) {
      // z plane: z-z0 = 0
      edge_coeffs = {0.0, 1.0, -v1coords[1]};
    } else {
      // compute the coefficients of the line passing through vert1 and vert2
      edge_coeffs = compute_coefficients(v1coords, v2coords);
    }

    // store the coefficients in the edge_coeffs view
    edge_coeffs_view(i, 0) = edge_coeffs[0];
    edge_coeffs_view(i, 1) = edge_coeffs[1];
    edge_coeffs_view(i, 2) = edge_coeffs[2];
    edge_coeffs_view(i, 3) = edge_coeffs[3];

    // print the coefficients of the edge
    std::cout << "Edge " << i << " has coefficients: " << edge_coeffs[0] << " " 
            << edge_coeffs[1] << " " << edge_coeffs[2] << " Top Bottom: " << edge_coeffs[3] << "\n";
  }
  // ********** Reading the Edge done ********** //
  auto face2vert = mesh.ask_down(2, 0);
  auto face2vertVerts = face2vert.ab2b;

  // * Step 2: loop over all faces
  auto face2edge = mesh.get_adj(2, 1);
  auto face2edgeOffsets = face2edge.a2ab;
  auto face2edgeEdges = face2edge.ab2b;

  for (Omega_h::LO i = 0; i < mesh.nfaces(); ++i) {
    auto edge1 = face2edgeEdges[3*i];
    auto edge2 = face2edgeEdges[3*i + 1];
    auto edge3 = face2edgeEdges[3*i + 2];

    // print the edges of the face
    std::cout << "Face " << i << " has edges " << edge1 << " " << edge2 << " " << edge3 << "\n";

    // extract the vertices of the face
    auto vert1 = face2vertVerts[3*i];
    auto vert2 = face2vertVerts[3*i + 1];
    auto vert3 = face2vertVerts[3*i + 2];

    auto v1coords = Omega_h::get_vector<2>(mesh.coords(), vert1);
    auto v2coords = Omega_h::get_vector<2>(mesh.coords(), vert2);
    auto v3coords = Omega_h::get_vector<2>(mesh.coords(), vert3);

    // print the vertices of the face
    std::cout << "Face " << i << " has vertices " << v1coords[0] << " " << v1coords[1] << ", " 
          << v2coords[0] << " " << v2coords[1] << ", " << v3coords[0] << " " << v3coords[1] << "\n";
    
  }



  // TODO: Remove this following code
  // return type Omega_h::Reals derived from Omega_h::Read<Omega_h::Real>
  const auto coords = mesh.coords(); // ? returns a Reals object: Read<Real> object: Real is double
  // array of type Omega_h::Write must be used when modified as below
  const Omega_h::Write<Omega_h::Real> u_w(mesh.nverts());
  // Omega_h::LO is a 32 bits int for local indexes on each proc
  const auto initialize_u = OMEGA_H_LAMBDA(Omega_h::LO r) {
    // get_vector<2> abstracts the storage convention inside coords array
    const auto x_r = Omega_h::get_vector<2>(coords, r);
    // quadratic 2d function between -1 and 1 on the square [0,1] x [0,1]
    u_w[r] = x_r[1] - std::pow(2 * x_r[0] - 1, 2);
  };
  // encapsulates Kokkos::parallel_for
  Omega_h::parallel_for(mesh.nverts(), initialize_u);

  // write the coordinates of the vertices to the terminal
  for (Omega_h::LO i = 0; i < mesh.nverts(); ++i) {
    const auto x_i = Omega_h::get_vector<2>(coords, i);
    std::cout << x_i[0] << " " << x_i[1] << "\n";
  }

  return 0;
}

std::vector<double> compute_coefficients(Omega_h::Vector<2> vert1, Omega_h::Vector<2> vert2) {
  // compute the coefficients of the line passing through vert1 and vert2
  double m = (vert2[1] - vert1[1]) / (vert2[0] - vert1[0]);
  double c = vert1[1] - m * vert1[0];
  double c2 = c * c;
  // if z > c, topbottomflag = 1, else -1
  double topbottomflag = (vert1[1] > c) ? 1 : -1;
  return {m * m, 2 * c, -c2, topbottomflag};
}
