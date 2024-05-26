#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <Kokkos_Core.hpp>

#include "Omega_h_file.hpp"
#include "Omega_h_for.hpp"
#include "Omega_h_mesh.hpp"
#include "Omega_h_mark.hpp"

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
std::vector<double> compute_coefficients(Omega_h::Few<double, 2> &vert1, Omega_h::Few<double, 2> &vert2);
/**
 * @brief Check weather to keep the inside or outside of the face
 * @details The 3 verts of the face and the edge coefficints are passed and substibuted in the plane equation
 * to get the inoroutflag
 * 
 * @param vert1 The first vertex of the face
 * @param vert2 The second vertex of the face
 * @param vert3 The third vertex of the face
 * @param edgeCoeffs The coefficients of the edge
 * @return int The flag to keep the inside or outside of the face
 * 
 * flag = 1 for outside, -1 for inside
 * c1*x^2 + c1*y^2 - z^2 + c2*z + c3 > 0 for outside and < 0 for inside
 * c are the coefficients for the edge.
 * 
 * For 2 vertices of the edge, it will be zero. Only for the third vertex, it will be non-zero.
*/
int inorout(Omega_h::Vector<2> vert1, Omega_h::Vector<2> vert2, Omega_h::Vector<2> vert3, std::vector<double> edgeCoeffs);

/**
 * @brief Determines in or out based on the vertices and the **line** coefficients
 * @details It also checks with the line: if the third vertex is above or below the line
*/
int inoroutWline(Omega_h::Vector<2> vert1, Omega_h::Vector<2> vert2, Omega_h::Vector<2> vert3, std::vector<double> edgeCoeffs);


/**
 * @brief Determines if a point is above or below a line
 * 
 * @param point The point to check
 * @param coeffs {m, c} The coefficients of the line
 * @return int 1 if above, -1 if below
*/
int above_or_below_line(Omega_h::Vector<2> point, std::vector<double> coeffs);

/*!
 * \brief Read the mesh file and go to each vertex to get its coordinates
*/
int main(int argc, char** argv) {
  auto start = std::chrono::steady_clock::now();
  // calls MPI_init(&argc, &argv) and Kokkos::initialize(argc, argv)
  auto lib = Omega_h::Library(&argc, &argv);
  // encapsulates many MPI functions, e.g., world.barrier()
  const auto world = lib.world();
  // create a distributed mesh object, calls mesh.balance() and then returns it
  Omega_h::Mesh mesh(&lib);

  // read the mesh filename
  std::string mesh_filename = argv[1];
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <mesh_filename> <printflag>=0\n";
    return 1;
  }
  bool printflag = false;
  if (argc == 3){
    int print = std::stoi(argv[2]);
    printflag = print;
  }
  
  
  Omega_h::binary::read(mesh_filename, world, &mesh);
  // ***************** Mesh reading is done ***************** //

  // ******** Get the boundary entities ******** //
  std::vector<int> bdrs;

  auto setup_time = std::chrono::steady_clock::now();
  std::cout << "Setup time: " << std::chrono::duration_cast<std::chrono::milliseconds>(setup_time - start).count() << "ms\n";
  // get boundary with mark_exposed_sides function
  auto exposed_sides = Omega_h::mark_exposed_sides(&mesh);
  for (Omega_h::LO i = 0; i < exposed_sides.size(); ++i) {
    if (exposed_sides[i]) {
      bdrs.push_back(int(i));
    }
  }

  auto bdrs_time = std::chrono::steady_clock::now();
  std::cout << "Boundary time: " << std::chrono::duration_cast<std::chrono::milliseconds>(bdrs_time - setup_time).count() << "ms\n";


  // *********** Read all the edges of the mesh *********** //
  // kokkos view to store nedges * 3 doubles : m^2, 2c, -c^2
  auto edge_coeffs_view = Kokkos::View<double*[6]>("edge_coeffs", mesh.nedges());

  // get the adjacency for the edges
  auto edge2vert = mesh.get_adj(1, 0);
  auto edgeOffsets = edge2vert.a2ab;
  auto edgeVertices = edge2vert.ab2b;

  std::ofstream edge_coeffs_file;
  edge_coeffs_file.open("edge_coeffs.dat");

  auto mesh_query_time = std::chrono::steady_clock::now();
  std::cout << "Mesh query time: " << std::chrono::duration_cast<std::chrono::milliseconds>(mesh_query_time - bdrs_time).count() << "ms\n";

  // * Step 1: loop over all edges and print the associated vertices
  const auto create_edge_coeffs = OMEGA_H_LAMBDA(Omega_h::LO i) {
    auto vert1 = edgeVertices[2*i];
    auto vert2 = edgeVertices[2*i + 1];
    auto v1coords = Omega_h::get_vector<2>(mesh.coords(), vert1);
    auto v2coords = Omega_h::get_vector<2>(mesh.coords(), vert2);

    if (printflag) {
      // print the coordinates of the vertices
      std::cout << "Edge " << i << " connects vertices " << v1coords[0] << 
          " " << v1coords[1] << " and " << v2coords[0] << " " << v2coords[1] << "\n";
    }
    
    // coefficient vector of doubles of size 3
    std::vector<double> edge_coeffs(5, 0.0);

    if (std::abs(v1coords[0] - v2coords[0]) < 1e-10) {
      // cylinder surface: x^2 + y^2 - r^2 = 0
      edge_coeffs = {1.0, 0.0, 0.0, -v1coords[0] * v1coords[0], 0, 1.0};
    } else if (std::abs(v1coords[1] - v2coords[1]) < 1e-10) {
      // z plane: z-z0 = 0
      edge_coeffs = {0.0, 0.0, 1.0, -v1coords[1], 0, 0};
    } else {
      // compute the coefficients of the line passing through vert1 and vert2
      edge_coeffs = compute_coefficients(v1coords, v2coords);
    }

    // store the coefficients in the edge_coeffs view
    edge_coeffs_view(i, 0) = edge_coeffs[0];
    edge_coeffs_view(i, 1) = edge_coeffs[1];
    edge_coeffs_view(i, 2) = edge_coeffs[2];
    edge_coeffs_view(i, 3) = edge_coeffs[3];
    edge_coeffs_view(i, 4) = edge_coeffs[4];
    edge_coeffs_view(i, 5) = edge_coeffs[5];

    if (printflag) {
      // print the coefficients of the edge
      std::cout << "Edge " << i << " has coefficients: " << edge_coeffs[0] << " " 
        << edge_coeffs[1] << " " << edge_coeffs[2] << " " << edge_coeffs[3] << " Top Bottom: " << edge_coeffs[4] << "\n";
    }

  };
  // loop over all edges
  Omega_h::parallel_for(mesh.nedges(), create_edge_coeffs);

  auto edge_calc_time = std::chrono::steady_clock::now();
  std::cout << "Edge calculation time: " << std::chrono::duration_cast<std::chrono::milliseconds>(edge_calc_time - mesh_query_time).count() << "ms\n";
  // write out the coefficients to a file
  for (Omega_h::LO i = 0; i < mesh.nedges(); ++i) {
    edge_coeffs_file << i << " " << edge_coeffs_view(i, 0) << " " << edge_coeffs_view(i, 1) << " " << edge_coeffs_view(i, 2) << " " << edge_coeffs_view(i, 3) << " " << edge_coeffs_view(i, 4) << "\n";
  }
  // the last line contains all the boundary edges
  edge_coeffs_file << "Boundary edges: " << bdrs.size() << "\n";
  for (Omega_h::LO i = 0; i < bdrs.size(); ++i) {
    edge_coeffs_file << bdrs[i] << " ";
  }
  edge_coeffs_file.close();

  auto edge_file_time = std::chrono::steady_clock::now();
  std::cout << "Edge file write time: " << std::chrono::duration_cast<std::chrono::milliseconds>(edge_file_time - edge_calc_time).count() << "ms\n";

  std::ofstream face2edgemap_file;
  face2edgemap_file.open("face2edgemap.dat");
  // ********** Reading the Edge done ********** //
  auto face2vert = mesh.ask_down(2, 0);
  auto face2vertVerts = face2vert.ab2b;

  // * Step 2: loop over all faces
  auto face2edge = mesh.get_adj(2, 1);
  //auto face2edgeOffsets = face2edge.a2ab;
  auto face2edgeEdges = face2edge.ab2b;

  // a kokkos view to store the face to edge map
  auto face2edgemap = Kokkos::View<int*[6]>("face2edgemap", mesh.nfaces());

  auto face_query_time = std::chrono::steady_clock::now();
  std::cout << "Face query time: " << std::chrono::duration_cast<std::chrono::milliseconds>(face_query_time - edge_file_time).count() << "ms\n";
  //for (Omega_h::LO i = 0; i < mesh.nfaces(); ++i) {
  const auto create_face2edgemap = OMEGA_H_LAMBDA(Omega_h::LO i) {
    auto edge1 = face2edgeEdges[3*i];
    auto edge2 = face2edgeEdges[3*i + 1];
    auto edge3 = face2edgeEdges[3*i + 2];

    if (printflag) {
      // print the edges of the face
      std::cout << "Face " << i << " has edges " << edge1 << " " << edge2 << " " << edge3 << "\n";
    }

    // extract the vertices of the face
    auto vert1 = face2vertVerts[3*i];
    auto vert2 = face2vertVerts[3*i + 1];
    auto vert3 = face2vertVerts[3*i + 2];

    auto v1coords = Omega_h::get_vector<2>(mesh.coords(), vert1);
    auto v2coords = Omega_h::get_vector<2>(mesh.coords(), vert2);
    auto v3coords = Omega_h::get_vector<2>(mesh.coords(), vert3);

    if (printflag) {
      // print the vertices of the face
      std::cout << "Face " << i << " has vertices " << v1coords[0] << " " << v1coords[1] << ", " 
        << v2coords[0] << " " << v2coords[1] << ", " << v3coords[0] << " " << v3coords[1] << "\n";
    }
    
    // each edge of a face has a flag associated with it
    int inoroutflag1 = inoroutWline(v1coords, v2coords, v3coords, {edge_coeffs_view(edge1, 0), edge_coeffs_view(edge1, 1), edge_coeffs_view(edge1, 2), edge_coeffs_view(edge1, 3), edge_coeffs_view(edge1, 4), edge_coeffs_view(edge1, 5)});
    int inoroutflag2 = inoroutWline(v1coords, v2coords, v3coords, {edge_coeffs_view(edge2, 0), edge_coeffs_view(edge2, 1), edge_coeffs_view(edge2, 2), edge_coeffs_view(edge2, 3), edge_coeffs_view(edge2, 4), edge_coeffs_view(edge2, 5)});
    int inoroutflag3 = inoroutWline(v1coords, v2coords, v3coords, {edge_coeffs_view(edge3, 0), edge_coeffs_view(edge3, 1), edge_coeffs_view(edge3, 2), edge_coeffs_view(edge3, 3), edge_coeffs_view(edge3, 4), edge_coeffs_view(edge3, 5)});

    // store the edges and inorout flags in the face2edgemap view
    face2edgemap(i, 0) = edge1;
    face2edgemap(i, 1) = inoroutflag1;
    face2edgemap(i, 2) = edge2;
    face2edgemap(i, 3) = inoroutflag2;
    face2edgemap(i, 4) = edge3;
    face2edgemap(i, 5) = inoroutflag3;
  };

  // loop over all faces
  Omega_h::parallel_for(mesh.nfaces(), create_face2edgemap);

  auto face_calc_time = std::chrono::steady_clock::now();
  std::cout << "Face calculation time: " << std::chrono::duration_cast<std::chrono::milliseconds>(face_calc_time - face_query_time).count() << "ms\n";
  // write out the face to edge map to a file
  for (Omega_h::LO i = 0; i < mesh.nfaces(); ++i) {
    face2edgemap_file << i << " " << face2edgemap(i, 0) << " " << face2edgemap(i, 1) << " " << face2edgemap(i, 2) << " " << face2edgemap(i, 3) << " " << face2edgemap(i, 4) << " " << face2edgemap(i, 5) << "\n";
  }
  face2edgemap_file.close();

  auto face_file_time = std::chrono::steady_clock::now();
  std::cout << "Face file write time: " << std::chrono::duration_cast<std::chrono::milliseconds>(face_file_time - face_calc_time).count() << "ms\n";
  std::cout << "Total time: " << std::chrono::duration_cast<std::chrono::milliseconds>(face_file_time - start).count() << "ms\n";
  return 0;
}

std::vector<double> compute_coefficients(Omega_h::Few<double, 2> &vert1, Omega_h::Few<double, 2> &vert2) {
  // compute the coefficients of the line passing through vert1 and vert2
  double m = (vert2[1] - vert1[1]) / (vert2[0] - vert1[0]);
  double c = vert1[1] - (m * vert1[0]);
  double c2 = c * c;
  // if z > c, topbottomflag = 1, else -1
  double topbottomflag = (vert1[1] > c) ? 1 : -1;
  return {m * m, -1, 2 * c, -c2, topbottomflag, m};
}

int inorout(Omega_h::Vector<2> vert1, Omega_h::Vector<2> vert2, Omega_h::Vector<2> vert3, std::vector<double> edgeCoeffs) {
  // evaluate the equation with the coefficients for 3 vertices
  std::vector<double> evals(3, 0.0);
  // ev    =            c1*x^2                   +        c2*z^2                       +            c3*z          +        c4
  evals[0] = edgeCoeffs[0] * vert1[0] * vert1[0] + edgeCoeffs[1] * vert1[1] * vert1[1] + edgeCoeffs[2] * vert1[1] + edgeCoeffs[3];
  evals[1] = edgeCoeffs[0] * vert2[0] * vert2[0] + edgeCoeffs[1] * vert2[1] * vert2[1] + edgeCoeffs[2] * vert2[1] + edgeCoeffs[3];
  evals[2] = edgeCoeffs[0] * vert3[0] * vert3[0] + edgeCoeffs[1] * vert3[1] * vert3[1] + edgeCoeffs[2] * vert3[1] + edgeCoeffs[3];

  // print the coefficients of the edge
  //std::cout << "Edge has coefficients: " << edgeCoeffs[0] << " " << edgeCoeffs[1] << " " << edgeCoeffs[2] << " " << edgeCoeffs[3] << "\n";
  // print the evaluations of the vertices
  //std::cout << "Evaluations are: " << evals[0] << " " << evals[1] << " " << evals[2] << "\n";
  // loop over the evals and check if the face is inside or outside
  for (double ev : evals) {
    // if ev is not close to 0 and positive, inoroutflag = 1 (outside), else -1 (inside)
    if ((std::abs(ev) > 1e-6) && (ev > 0)) {
      return 1;
    } else if ((std::abs(ev) > 1e-6) && (ev < 0)) {
      return -1;
    }
  }
  // should not reach here
  std::cerr << "Looks like all three vertices are on the edge. Exiting...\n";
  return 0;
}

int inoroutWline(Omega_h::Vector<2> vert1, Omega_h::Vector<2> vert2, Omega_h::Vector<2> vert3, std::vector<double> edgeCoeffs) {
  
  // ********* this part is same as before ********* //
  // * because, this will work or non cone surfaces
  // evaluate the equation with the coefficients for 3 vertices
  std::vector<double> evals(3, 0.0);
  // ev    =            c1*x^2                   +        c2*z^2                       +            c3*z          +        c4
  evals[0] = edgeCoeffs[0] * vert1[0] * vert1[0] + edgeCoeffs[1] * vert1[1] * vert1[1] + edgeCoeffs[2] * vert1[1] + edgeCoeffs[3];
  evals[1] = edgeCoeffs[0] * vert2[0] * vert2[0] + edgeCoeffs[1] * vert2[1] * vert2[1] + edgeCoeffs[2] * vert2[1] + edgeCoeffs[3];
  evals[2] = edgeCoeffs[0] * vert3[0] * vert3[0] + edgeCoeffs[1] * vert3[1] * vert3[1] + edgeCoeffs[2] * vert3[1] + edgeCoeffs[3];
  
  // print the coefficients of the edge
  //std::cout << "Edge has coefficients: " << edgeCoeffs[0] << " " << edgeCoeffs[1] << " " << edgeCoeffs[2] << " " << edgeCoeffs[3] << "\n";
  // print the evaluations of the vertices
  //std::cout << "Evaluations are: " << evals[0] << " " << evals[1] << " " << evals[2] << "\n";
  // *********************************************** //

  // ************** in case of cone **************** //
  // * check if it is a cone: if m and c both are non-zero
  double m = edgeCoeffs[5];
  double c = edgeCoeffs[2]/2.0;
  int topbottomflag = edgeCoeffs[4];
  int lineflag = 0;
  if (std::abs(edgeCoeffs[1]+1) < 1e-10) { // for cone, coefficient of z^2 is -1
    // check if the third vertex is above or below the line
    int lineevals[3];
    lineevals[0] = above_or_below_line(vert1, {m, c});
    lineevals[1] = above_or_below_line(vert2, {m, c});
    lineevals[2] = above_or_below_line(vert3, {m, c});
    // any two of them will be zero and the third will be non-zero
    for (int ev : lineevals) {
      //std::cout << "lin eval: " << ev << "\n";
      if (ev == -1) {lineflag = -1;}
      if (ev == 1) {lineflag = 1;}
    }
    // print for debugging
    //std::cout << "Evaluating for cone: with m = " << m << " and c = " << c << " and lineflag = " << lineflag << ", topbottomflag = " << topbottomflag << "\n";

    // if any or lineflag or topbottomflag is 1 and the other is -1, return 1
    if (lineflag+topbottomflag == 0) {
      return 1;
    } else {
      return -1;
    }
  }
  // ******************** The rest will continue if it is not a cone **************** //


  // loop over the evals and check if the face is inside or outside
  for (double ev : evals) {
    // if ev is not close to 0 and positive, inoroutflag = 1 (outside), else -1 (inside)
    if ((std::abs(ev) > 1e-6) && (ev > 0)) {
      return 1;
    } else if ((std::abs(ev) > 1e-6) && (ev < 0)) {
      return -1;
    }
  }
  // should not reach here
  std::cerr << "Looks like all three vertices are on the edge. Exiting...\n";
  return 0;
}

int above_or_below_line(Omega_h::Vector<2> point, std::vector<double> coeffs) {
  // evaluate z for the point's x : z = m * x1 + c
  double z = coeffs[0] * point[0] + coeffs[1];
  // compare the point's z with the computed z: if point's z > z, return 1, if < return -1 else 0
  double diff = point[1] - z;
  if (diff > 1e-6) {
    return 1;
  } else if (diff < -1e-6) {
    return -1;
  } else {
    return 0;
  }
}
