#include <Kokkos_Core.hpp>
#include <Omega_h_fail.hpp>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "Omega_h_bbox.hpp"
#include "Omega_h_file.hpp"
#include "Omega_h_for.hpp"
#include "Omega_h_mark.hpp"
#include "Omega_h_mesh.hpp"

#include <netcdf>
/**
 * The equation of the line is:
 *     y = mx + c
 * where,
 *     m = (z2 - z1) / (x2 - x1)
 * and
 *     c = z2 - m * x2
 *
 * When this line revolves around the z axis, it will form a cone. The equation
 * of the cone will be: x^2 + y^2 = r^2 where, r = x (x is the distance from the
 * z axis). Therefore, from the equation of the line, x = (z - c) / m
 * Substituting this in the equation of the cone, we get:
 *     x^2 + y^2 = (1 / m^2) * (z - c)^2
 *
 * Expanding and simplifying, we get:
 *     m^2 * x^2 + m^2 * y^2 - z^2 + 2 * c * z - c^2 = 0
 *
 * This equation needs to be generalized for any line in the xz plane. It will
 * not work for cylinders as m will be undefined. There is one way of avoiding
 * this by checking if x1 = x2. If it is, then use `ZCylinder` instead of
 * `Quadratic` for the face.
 */
std::vector<double> compute_coefficients(Omega_h::Few<double, 2> &vert1,
                                         Omega_h::Few<double, 2> &vert2);
/**
 * @brief Check weather to keep the inside or outside of the face
 * @details The 3 verts of the face and the edge coefficints are passed and
 * substibuted in the plane equation to get the inoroutflag
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
 * For 2 vertices of the edge, it will be zero. Only for the third vertex, it
 * will be non-zero.
 */
int inorout(Omega_h::Vector<2> vert1, Omega_h::Vector<2> vert2,
            Omega_h::Vector<2> vert3, std::vector<double> edgeCoeffs);

/**
 * @brief Determines in or out based on the vertices and the **line**
 * coefficients
 * @details It also checks with the line: if the third vertex is above or below
 * the line
 */
int inoroutWline(Omega_h::Vector<2> vert1, Omega_h::Vector<2> vert2,
                 Omega_h::Vector<2> vert3, std::vector<double> edgeCoeffs);

/**
 * @brief Determines if a point is above or below a line
 *
 * @param point The point to check
 * @param coeffs {m, c} The coefficients of the line
 * @return int 1 if above, -1 if below
 */
int above_or_below_line(Omega_h::Vector<2> point, std::vector<double> coeffs);

/**
 * @brief Get the closest node to the vertical axis
 *
 * @param mesh The mesh object
 * @return int The index of the closest node
 */
int get_closest_node_to_vertical_axis(const Omega_h::Mesh &mesh);

int get_lower_left_corner(Omega_h::Mesh &mesh);
int get_closest_wall_node(Omega_h::Mesh &mesh);
int get_number_of_wall_nodes(Omega_h::Mesh &mesh);
void walk_on_curve_square_curve(Omega_h::Mesh &mesh,
                                std::vector<int> &sorted_curve_edges,
                                std::vector<int> &sorted_curve_nodes,
                                int starting_node);
void walk_on_curve_wall_curve(Omega_h::Mesh &mesh,
                              std::vector<int> &sorted_curve_edges,
                              std::vector<int> &sorted_curve_nodes,
                              int starting_node);
void sort_curve(Omega_h::Mesh &mesh, std::vector<int> &sorted_curve_edges,
                std::vector<int> &sorted_curve_nodes, std::string type);
std::pair<int, int> findTwoSmallestIndices(const std::vector<int> &vec);

/*!
 * \brief Read the mesh file and go to each vertex to get its coordinates
 */
int main(int argc, char **argv) {
  auto start = std::chrono::steady_clock::now();
  // calls MPI_init(&argc, &argv) and Kokkos::initialize(argc, argv)
  auto lib = Omega_h::Library(&argc, &argv);
  // encapsulates many MPI functions, e.g., world.barrier()
  const auto world = lib.world();
  // create a distributed mesh object, calls mesh.balance() and then returns it
  Omega_h::Mesh mesh(&lib);

  // read the mesh filename
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <mesh_filename> <printflag>=0\n";
    return 1;
  }
  std::string mesh_filename = argv[1];
  bool printflag = false;
  if (argc == 3) {
    int print = std::stoi(argv[2]);
    printflag = print;
  }

  Omega_h::binary::read(mesh_filename, world, &mesh);
  // ***************** Mesh reading is done ***************** //

  // ******** Get the boundary entities ******** //
  std::vector<int> bdrs;

  auto setup_time = std::chrono::steady_clock::now();
  std::cout << "Setup time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   setup_time - start)
                   .count()
            << "ms\n";
  // get boundary with mark_exposed_sides function
  auto exposed_sides = Omega_h::mark_exposed_sides(&mesh);
  for (Omega_h::LO i = 0; i < exposed_sides.size(); ++i) {
    if (exposed_sides[i]) {
      bdrs.push_back(int(i));
    }
  }

  auto bdrs_time = std::chrono::steady_clock::now();
  std::cout << "Boundary time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(bdrs_time -
                                                                     setup_time)
                   .count()
            << "ms\n";

  // *********** Read all the edges of the mesh *********** //
  // kokkos view to store nedges * 3 doubles : m^2, 2c, -c^2
  auto edge_coeffs_view =
      Kokkos::View<double *[6]>("edge_coeffs", mesh.nedges());

  // get the adjacency for the edges
  auto edge2vert = mesh.get_adj(1, 0);
  auto edgeOffsets = edge2vert.a2ab;
  auto edgeVertices = edge2vert.ab2b;

  std::ofstream edge_coeffs_file;
  edge_coeffs_file.open("edge_coeffs.dat");

  auto mesh_query_time = std::chrono::steady_clock::now();
  std::cout << "Mesh query time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   mesh_query_time - bdrs_time)
                   .count()
            << "ms\n";

  // * Step 1: loop over all edges and print the associated vertices
  const auto create_edge_coeffs = OMEGA_H_LAMBDA(Omega_h::LO i) {
    auto vert1 = edgeVertices[2 * i];
    auto vert2 = edgeVertices[2 * i + 1];
    auto v1coords = Omega_h::get_vector<2>(mesh.coords(), vert1);
    auto v2coords = Omega_h::get_vector<2>(mesh.coords(), vert2);

    if (printflag) {
      // print the coordinates of the vertices
      std::cout << "Edge " << i << " connects vertices " << v1coords[0] << " "
                << v1coords[1] << " and " << v2coords[0] << " " << v2coords[1]
                << "\n";
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
      std::cout << "Edge " << i << " has coefficients: " << edge_coeffs[0]
                << " " << edge_coeffs[1] << " " << edge_coeffs[2] << " "
                << edge_coeffs[3] << " Top Bottom: " << edge_coeffs[4] << "\n";
    }
  };
  // loop over all edges
  Omega_h::parallel_for(mesh.nedges(), create_edge_coeffs);

  auto edge_calc_time = std::chrono::steady_clock::now();
  std::cout << "Edge calculation time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   edge_calc_time - mesh_query_time)
                   .count()
            << "ms\n";
  // write out the coefficients to a file
  // for (Omega_h::LO i = 0; i < mesh.nedges(); ++i) {
  //  edge_coeffs_file << i << " " << edge_coeffs_view(i, 0) << " "
  //                   << edge_coeffs_view(i, 1) << " " << edge_coeffs_view(i,
  //                   2)
  //                   << " " << edge_coeffs_view(i, 3) << " "
  //                   << edge_coeffs_view(i, 4) << "\n";
  //}
  // the last line contains all the boundary edges
  edge_coeffs_file << "Boundary edges: " << bdrs.size() << "\n";
  for (Omega_h::LO i = 0; i < bdrs.size(); ++i) {
    edge_coeffs_file << bdrs[i] << " ";
  }
  edge_coeffs_file.close();

  auto edge_file_time = std::chrono::steady_clock::now();
  std::cout << "Edge file write time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   edge_file_time - edge_calc_time)
                   .count()
            << "ms\n";

  std::ofstream face2edgemap_file;
  face2edgemap_file.open("face2edgemap.dat");
  // ********** Reading the Edge done ********** //
  auto face2vert = mesh.ask_down(2, 0);
  auto face2vertVerts = face2vert.ab2b;

  // * Step 2: loop over all faces
  auto face2edge = mesh.get_adj(2, 1);
  // auto face2edgeOffsets = face2edge.a2ab;
  auto face2edgeEdges = face2edge.ab2b;

  // a kokkos view to store the face to edge map
  auto face2edgemap = Kokkos::View<int *[6]>("face2edgemap", mesh.nfaces());

  auto face_query_time = std::chrono::steady_clock::now();
  std::cout << "Face query time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   face_query_time - edge_file_time)
                   .count()
            << "ms\n";
  // for (Omega_h::LO i = 0; i < mesh.nfaces(); ++i) {
  const auto create_face2edgemap = OMEGA_H_LAMBDA(Omega_h::LO i) {
    auto edge1 = face2edgeEdges[3 * i];
    auto edge2 = face2edgeEdges[3 * i + 1];
    auto edge3 = face2edgeEdges[3 * i + 2];

    if (printflag) {
      // print the edges of the face
      std::cout << "Face " << i << " has edges " << edge1 << " " << edge2 << " "
                << edge3 << "\n";
    }

    // extract the vertices of the face
    auto vert1 = face2vertVerts[3 * i];
    auto vert2 = face2vertVerts[3 * i + 1];
    auto vert3 = face2vertVerts[3 * i + 2];

    auto v1coords = Omega_h::get_vector<2>(mesh.coords(), vert1);
    auto v2coords = Omega_h::get_vector<2>(mesh.coords(), vert2);
    auto v3coords = Omega_h::get_vector<2>(mesh.coords(), vert3);

    if (printflag) {
      // print the vertices of the face
      std::cout << "Face " << i << " has vertices " << v1coords[0] << " "
                << v1coords[1] << ", " << v2coords[0] << " " << v2coords[1]
                << ", " << v3coords[0] << " " << v3coords[1] << "\n";
    }

    // each edge of a face has a flag associated with it
    int inoroutflag1 =
        inoroutWline(v1coords, v2coords, v3coords,
                     {edge_coeffs_view(edge1, 0), edge_coeffs_view(edge1, 1),
                      edge_coeffs_view(edge1, 2), edge_coeffs_view(edge1, 3),
                      edge_coeffs_view(edge1, 4), edge_coeffs_view(edge1, 5)});
    int inoroutflag2 =
        inoroutWline(v1coords, v2coords, v3coords,
                     {edge_coeffs_view(edge2, 0), edge_coeffs_view(edge2, 1),
                      edge_coeffs_view(edge2, 2), edge_coeffs_view(edge2, 3),
                      edge_coeffs_view(edge2, 4), edge_coeffs_view(edge2, 5)});
    int inoroutflag3 =
        inoroutWline(v1coords, v2coords, v3coords,
                     {edge_coeffs_view(edge3, 0), edge_coeffs_view(edge3, 1),
                      edge_coeffs_view(edge3, 2), edge_coeffs_view(edge3, 3),
                      edge_coeffs_view(edge3, 4), edge_coeffs_view(edge3, 5)});

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
  std::cout << "Face calculation time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   face_calc_time - face_query_time)
                   .count()
            << "ms\n";
  // write out the face to edge map to a file
  // for (Omega_h::LO i = 0; i < mesh.nfaces(); ++i) {
  //  face2edgemap_file << i << " " << face2edgemap(i, 0) << " "
  //                    << face2edgemap(i, 1) << " " << face2edgemap(i, 2) << "
  //                    "
  //                    << face2edgemap(i, 3) << " " << face2edgemap(i, 4) << "
  //                    "
  //                    << face2edgemap(i, 5) << "\n";
  //}
  face2edgemap_file.close();

  auto face_file_time = std::chrono::steady_clock::now();
  std::cout << "Face file write time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   face_file_time - face_calc_time)
                   .count()
            << "ms\n";

  // *********** Write to netcdf file *********** //
  // ******************************************** //
  // create a netcdf file
  std::string nc_filename = "osh2degas2in.nc";
  netCDF::NcFile ncFile(nc_filename, netCDF::NcFile::replace);

  // create dimensions for scalars: number of edges and faces
  netCDF::NcDim scalars = ncFile.addDim("scalars", 1);
  netCDF::NcVar nfacesVar = ncFile.addVar("ncells", netCDF::ncInt, scalars);
  netCDF::NcVar nedgesVar = ncFile.addVar("nsurfaces", netCDF::ncInt, scalars);

  // * write the number of edges and faces
  int nfaces = mesh.nfaces();
  int nedges = mesh.nedges();
  nfacesVar.putVar(&nfaces);
  nedgesVar.putVar(&nedges);

  // assert that the mesh.dim == 2
  assert(mesh.dim() == 2); // only 2D meshes are used here

  // * get the mesh bounding box
  Omega_h::BBox<2> bbox = Omega_h::get_bounding_box<2>(&mesh);
  Omega_h::Vector<2> min = bbox.min;
  Omega_h::Vector<2> max = bbox.max;

  // print the bounding box
  std::cout << "Bounding box: " << min[0] << " " << min[1] << " " << max[0]
            << " " << max[1] << "\n";

  std::vector<double> universal_cell_min = {-max[0], -max[0], min[1]};
  std::vector<double> universal_cell_max = {max[0], max[0], max[1]};

  // * write the bounding box to the netcdf file
  netCDF::NcDim bbox_dim = ncFile.addDim("bbox_dim", 3);
  netCDF::NcVar bboxMinVar =
      ncFile.addVar("universal_cell_min", netCDF::ncDouble, bbox_dim);
  netCDF::NcVar bboxMaxVar =
      ncFile.addVar("universal_cell_max", netCDF::ncDouble, bbox_dim);
  bboxMinVar.putAtt("units", "m");
  bboxMaxVar.putAtt("units", "m");
  bboxMinVar.putVar(universal_cell_min.data());
  bboxMaxVar.putVar(universal_cell_max.data());

  // *** write map between node number to simNumbering
  netCDF::NcDim node2sim_dim = ncFile.addDim("sim_number_dim", mesh.nverts());
  netCDF::NcVar node2simVar =
      ncFile.addVar("node2simNumberingmap", netCDF::ncInt, node2sim_dim);

  // get the node numbering simNumber from the mesh
  std::vector<int> node2sim(mesh.nverts());
  if (!mesh.has_tag(0, "simNumbering")) {
    std::cerr << "No simNumbering tag found on vertices\n";
    return 1;
  }

  auto simNumbering = mesh.get_array<Omega_h::LO>(0, "simNumbering");
  for (Omega_h::LO i = 0; i < mesh.nverts(); ++i) {
    node2sim[i] = simNumbering[i];
  }
  node2simVar.putVar(node2sim.data());

  // a reverse map from simNumbering to node number
  std::vector<int> sim2node(mesh.nverts());
  for (Omega_h::LO i = 0; i < mesh.nverts(); ++i) {
    sim2node[simNumbering[i]] = i;
  }

  // * write the nodes for each edge to the netcdf file
  netCDF::NcDim edge2node_dim = ncFile.addDim("node_number_dim", 2);
  netCDF::NcDim edges_dim = ncFile.addDim("edges_dim", nedges);
  std::vector<netCDF::NcDim> edge_dims{edges_dim, edge2node_dim};
  netCDF::NcVar edge2nodeVar =
      ncFile.addVar("edge2nodemap", netCDF::ncInt, edge_dims);

  // get the adjacency for the edges
  auto edge2node = mesh.get_adj(Omega_h::EDGE, Omega_h::VERT);
  auto edge2nodeNodes = edge2node.ab2b;
  // create the edge2node array nedges by 2
  std::vector<std::array<int, 2>> edge2node_data(nedges);
  for (Omega_h::LO i = 0; i < nedges; ++i) {
    edge2node_data[i][0] =
        node2sim[edge2nodeNodes[2 * i]]; // edge2nodeNodes[2 * i];
    edge2node_data[i][1] =
        node2sim[edge2nodeNodes[2 * i + 1]]; // edge2nodeNodes[2 * i + 1];
  }
  edge2nodeVar.putVar(edge2node_data.data());

  // * write element to edge map to the netcdf file
  netCDF::NcDim face2edge_dim = ncFile.addDim("edge_number_dim", 3);
  netCDF::NcDim faces_dim = ncFile.addDim("faces_dim", nfaces);
  std::vector<netCDF::NcDim> face_dims{faces_dim, face2edge_dim};
  netCDF::NcVar face2edgeVar =
      ncFile.addVar("face2edgemap", netCDF::ncInt, face_dims);

  // create the face2edge array nfaces by 3
  std::vector<std::array<int, 3>> face2edge_data(nfaces);
  for (Omega_h::LO i = 0; i < nfaces; ++i) {
    face2edge_data[i][0] = face2edgemap(i, 0);
    face2edge_data[i][1] = face2edgemap(i, 2);
    face2edge_data[i][2] = face2edgemap(i, 4);
  }
  face2edgeVar.putVar(face2edge_data.data());

  // * write edge to face map to the netcdf file
  netCDF::NcDim edge2face_dim = ncFile.addDim("face_number_dim", 4);
  std::vector<netCDF::NcDim> edge2face_dims{edges_dim, edge2face_dim};
  netCDF::NcVar edge2faceVar =
      ncFile.addVar("edge2facemap", netCDF::ncInt, edge2face_dims);

  // create the edge2face array nedges by 2
  std::vector<std::array<int, 4>> edge2face_data(nedges);
  auto edge2face = mesh.get_adj(Omega_h::EDGE, Omega_h::FACE);
  auto edge2faceFaces = edge2face.ab2b;
  // for this case, offset is important because one edge can have 2 faces
  // (internal) or 1 face (boundary)
  auto edge2faceOffsets = edge2face.a2ab;
  // if only one face, put -1 in the second entry
  for (Omega_h::LO i = 0; i < nedges; ++i) {
    edge2face_data[i][0] = edge2faceFaces[edge2faceOffsets[i]];
    edge2face_data[i][1] = (edge2faceOffsets[i + 1] - edge2faceOffsets[i] == 1)
                               ? -1
                               : edge2faceFaces[edge2faceOffsets[i] + 1];
    // get the inorout flag for the edge
    std::vector<int> face0edges = {face2edgemap(edge2face_data[i][0], 0),
                                   face2edgemap(edge2face_data[i][0], 2),
                                   face2edgemap(edge2face_data[i][0], 4)};
    for (int j = 0; j < 3; ++j) {
      if (face0edges[j] == i) {
        edge2face_data[i][2] = face2edgemap(edge2face_data[i][0], 2 * j + 1);
        break;
      }
    }
    // same for the second face only if it exists
    if (edge2face_data[i][1] != -1) {
      std::vector<int> face1edges = {face2edgemap(edge2face_data[i][1], 0),
                                     face2edgemap(edge2face_data[i][1], 2),
                                     face2edgemap(edge2face_data[i][1], 4)};
      for (int j = 0; j < 3; ++j) {
        if (face1edges[j] == i) {
          edge2face_data[i][3] = face2edgemap(edge2face_data[i][1], 2 * j + 1);
          break;
        }
      }
    } else /* 0 if only one face*/ {
      edge2face_data[i][3] = 0;
    }
  }
  edge2faceVar.putVar(edge2face_data.data());

  // * write the coordinates of the vertices to the netcdf file
  netCDF::NcDim vertices_dim = ncFile.addDim(
      "vertices_dim", mesh.nverts() + 1); // simNumbering starts from 1
  netCDF::NcDim coords_dim = ncFile.addDim("coords_dim", 2);
  std::vector<netCDF::NcDim> coords_dims{vertices_dim, coords_dim};
  netCDF::NcVar coordsVar =
      ncFile.addVar("coords", netCDF::ncDouble, coords_dims);

  // create the coords array nverts by 2
  std::vector<std::array<double, 2>> coords_data(mesh.nverts() + 1);

  auto coords = mesh.coords();
  for (Omega_h::LO i = 0; i < mesh.nverts(); ++i) {
    auto vcoords = Omega_h::get_vector<2>(coords, i);
    // coords_data[i][0] = vcoords[0];
    // coords_data[i][1] = vcoords[1];
    coords_data[node2sim[i]][0] = vcoords[0];
    coords_data[node2sim[i]][1] = vcoords[1];
  }
  coordsVar.putVar(coords_data.data());

  // get the wall nodes
  std::vector<int> sorted_square_curve_edges;
  std::vector<int> sorted_square_curve_nodes;
  sort_curve(mesh, sorted_square_curve_edges, sorted_square_curve_nodes,
             "square");
  std::cout << "Number of sorted square nodes: "
            << sorted_square_curve_nodes.size() << "\n";
  // sim numbers of the boundary nodes
  std::vector<int> sorted_square_curve_nodes_sim(
      sorted_square_curve_nodes.size());
  for (int i = 0; i < sorted_square_curve_nodes.size(); ++i) {
    sorted_square_curve_nodes_sim[i] = node2sim[sorted_square_curve_nodes[i]];
  }

  if (printflag) {
    std::cout << "Square nodes are ... \n";
    for (int i = 0; i < sorted_square_curve_nodes.size(); ++i) {
      std::cout << sorted_square_curve_nodes[i] << "\t"
                << sorted_square_curve_nodes_sim[i] << "\n";
    }
    for (int i = 0; i < sorted_square_curve_nodes_sim.size(); ++i) {
      std::cout << sorted_square_curve_nodes_sim[i] << ", ";
    }
    std::cout << "\n";
  }

  std::vector<int> sorted_wall_curve_edges;
  std::vector<int> sorted_wall_curve_nodes;
  sort_curve(mesh, sorted_wall_curve_edges, sorted_wall_curve_nodes, "wall");
  std::cout << "Number of sorted wall nodes: " << sorted_wall_curve_nodes.size()
            << "\n";

  std::vector<int> sorted_wall_curve_nodes_sim(sorted_wall_curve_nodes.size());
  for (int i = 0; i < sorted_wall_curve_nodes.size(); ++i) {
    sorted_wall_curve_nodes_sim[i] = node2sim[sorted_wall_curve_nodes[i]];
  }

  if (printflag) {
    std::cout << "Wall nodes are ... \n";
    for (int i = 0; i < sorted_wall_curve_nodes.size(); ++i) {
      std::cout << sorted_wall_curve_nodes[i] << "\t"
                << sorted_wall_curve_nodes_sim[i] << "\n";
    }
    for (int i = 0; i < sorted_wall_curve_nodes_sim.size(); ++i) {
      std::cout << sorted_wall_curve_nodes_sim[i] << ", ";
    }
    std::cout << "\n";
  }
  std::cout << "Number of wall nodes - post sim numbering: "
            << sorted_wall_curve_nodes.size() << "\n";
  std::cout << "Number of square nodes - post sim numbering: "
            << sorted_square_curve_nodes.size() << "\n";

  // * write the boundary edges to the netcdf file
  netCDF::NcDim bdrs_dim = ncFile.addDim("bdrs_dim", bdrs.size());
  netCDF::NcVar bdrsVar =
      ncFile.addVar("boundary_edges", netCDF::ncInt, bdrs_dim);

  // * write the boundary nodes to the netcdf file
  netCDF::NcDim bdrnodes_dim = ncFile.addDim("bdrnodes_dim", bdrs.size());
  netCDF::NcVar bdrnodesVar =
      ncFile.addVar("boundary_nodes", netCDF::ncInt, bdrnodes_dim);

  bdrsVar.putVar(sorted_wall_curve_edges.data());
  // bdrnodesVar.putVar(bdrnodes.data());
  bdrnodesVar.putVar(sorted_square_curve_nodes_sim.data());

  // * write the wall nodes to the netcdf file
  netCDF::NcDim wallnodes_dim =
      ncFile.addDim("wallnodes_dim", sorted_wall_curve_nodes.size());
  netCDF::NcVar wallnodesVar =
      ncFile.addVar("wall_nodes", netCDF::ncInt, wallnodes_dim);

  // * write the wall edges to the netcdf file
  netCDF::NcDim walledges_dim =
      ncFile.addDim("walledges_dim", sorted_wall_curve_edges.size());
  netCDF::NcVar walledgesVar =
      ncFile.addVar("wall_edges", netCDF::ncInt, walledges_dim);

  wallnodesVar.putVar(sorted_wall_curve_nodes_sim.data());
  walledgesVar.putVar(sorted_wall_curve_edges.data());

  // * done writing to the netcdf file

  // close the netcdf file
  ncFile.close();

  auto nc_file_time = std::chrono::steady_clock::now();
  std::cout << "NetCDF file write time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   nc_file_time - face_file_time)
                   .count()
            << "ms\n";

  std::cout << "Total time: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   nc_file_time - start)
                   .count()
            << "ms\n";

  return 0;
} // main

int get_closest_node_to_vertical_axis(const Omega_h::Mesh &mesh) {
  // get the coordinates of the vertices
  auto coords = mesh.coords();
  // loop thourgh all the vertices and save the i of the closest node
  int closest_node = 0;
  double distance = 1e10;
  for (Omega_h::LO i = 0; i < mesh.nverts(); ++i) {
    auto vcoords = Omega_h::get_vector<2>(coords, i);
    // the 1st coordinate is distance from the vertical axis
    if (vcoords[0] < distance) {
      distance = vcoords[0];
      closest_node = i;
    }
  }
  OMEGA_H_CHECK(distance < 1e10);
  OMEGA_H_CHECK(closest_node < mesh.nverts());
  OMEGA_H_CHECK(distance > 0.0);

  return closest_node;
}

std::vector<double> compute_coefficients(Omega_h::Few<double, 2> &vert1,
                                         Omega_h::Few<double, 2> &vert2) {
  // compute the coefficients of the line passing through vert1 and vert2
  double m = (vert2[1] - vert1[1]) / (vert2[0] - vert1[0]);
  double c = vert1[1] - (m * vert1[0]);
  double c2 = c * c;
  // if z > c, topbottomflag = 1, else -1
  double topbottomflag = (vert1[1] > c) ? 1 : -1;
  return {m * m, -1, 2 * c, -c2, topbottomflag, m};
}

int inorout(Omega_h::Vector<2> vert1, Omega_h::Vector<2> vert2,
            Omega_h::Vector<2> vert3, std::vector<double> edgeCoeffs) {
  // evaluate the equation with the coefficients for 3 vertices
  std::vector<double> evals(3, 0.0);
  // ev    =            c1*x^2                   +        c2*z^2 + c3*z + c4
  evals[0] = edgeCoeffs[0] * vert1[0] * vert1[0] +
             edgeCoeffs[1] * vert1[1] * vert1[1] + edgeCoeffs[2] * vert1[1] +
             edgeCoeffs[3];
  evals[1] = edgeCoeffs[0] * vert2[0] * vert2[0] +
             edgeCoeffs[1] * vert2[1] * vert2[1] + edgeCoeffs[2] * vert2[1] +
             edgeCoeffs[3];
  evals[2] = edgeCoeffs[0] * vert3[0] * vert3[0] +
             edgeCoeffs[1] * vert3[1] * vert3[1] + edgeCoeffs[2] * vert3[1] +
             edgeCoeffs[3];

  // print the coefficients of the edge
  // std::cout << "Edge has coefficients: " << edgeCoeffs[0] << " " <<
  // edgeCoeffs[1] << " " << edgeCoeffs[2] << " " << edgeCoeffs[3] << "\n";
  // print the evaluations of the vertices
  // std::cout << "Evaluations are: " << evals[0] << " " << evals[1] << " " <<
  // evals[2] << "\n";
  // loop over the evals and check if the face is inside or outside
  for (double ev : evals) {
    // if ev is not close to 0 and positive, inoroutflag = 1 (outside), else -1
    // (inside)
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

int inoroutWline(Omega_h::Vector<2> vert1, Omega_h::Vector<2> vert2,
                 Omega_h::Vector<2> vert3, std::vector<double> edgeCoeffs) {

  // ********* this part is same as before ********* //
  // * because, this will work or non cone surfaces
  // evaluate the equation with the coefficients for 3 vertices
  std::vector<double> evals(3, 0.0);
  // ev    =            c1*x^2                   +        c2*z^2 + c3*z + c4
  evals[0] = edgeCoeffs[0] * vert1[0] * vert1[0] +
             edgeCoeffs[1] * vert1[1] * vert1[1] + edgeCoeffs[2] * vert1[1] +
             edgeCoeffs[3];
  evals[1] = edgeCoeffs[0] * vert2[0] * vert2[0] +
             edgeCoeffs[1] * vert2[1] * vert2[1] + edgeCoeffs[2] * vert2[1] +
             edgeCoeffs[3];
  evals[2] = edgeCoeffs[0] * vert3[0] * vert3[0] +
             edgeCoeffs[1] * vert3[1] * vert3[1] + edgeCoeffs[2] * vert3[1] +
             edgeCoeffs[3];

  // print the coefficients of the edge
  // std::cout << "Edge has coefficients: " << edgeCoeffs[0] << " " <<
  // edgeCoeffs[1] << " " << edgeCoeffs[2] << " " << edgeCoeffs[3] << "\n";
  // print the evaluations of the vertices
  // std::cout << "Evaluations are: " << evals[0] << " " << evals[1] << " " <<
  // evals[2] << "\n";
  // *********************************************** //

  // ************** in case of cone **************** //
  // * check if it is a cone: if m and c both are non-zero
  double m = edgeCoeffs[5];
  double c = edgeCoeffs[2] / 2.0;
  int topbottomflag = edgeCoeffs[4];
  int lineflag = 0;
  if (std::abs(edgeCoeffs[1] + 1) <
      1e-10) { // for cone, coefficient of z^2 is -1
    // check if the third vertex is above or below the line
    int lineevals[3];
    lineevals[0] = above_or_below_line(vert1, {m, c});
    lineevals[1] = above_or_below_line(vert2, {m, c});
    lineevals[2] = above_or_below_line(vert3, {m, c});
    // any two of them will be zero and the third will be non-zero
    for (int ev : lineevals) {
      // std::cout << "lin eval: " << ev << "\n";
      if (ev == -1) {
        lineflag = -1;
      }
      if (ev == 1) {
        lineflag = 1;
      }
    }
    // print for debugging
    // std::cout << "Evaluating for cone: with m = " << m << " and c = " << c <<
    // " and lineflag = " << lineflag << ", topbottomflag = " << topbottomflag
    // << "\n";

    // if any or lineflag or topbottomflag is 1 and the other is -1, return 1
    if (lineflag + topbottomflag == 0) {
      return 1;
    } else {
      return -1;
    }
  }
  // ******************** The rest will continue if it is not a cone
  // **************** //

  // loop over the evals and check if the face is inside or outside
  for (double ev : evals) {
    // if ev is not close to 0 and positive, inoroutflag = 1 (outside), else -1
    // (inside)
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
  // compare the point's z with the computed z: if point's z > z, return 1, if <
  // return -1 else 0
  double diff = point[1] - z;
  if (diff > 1e-6) {
    return 1;
  } else if (diff < -1e-6) {
    return -1;
  } else {
    return 0;
  }
}

void sort_curve(Omega_h::Mesh &mesh, std::vector<int> &sorted_curve_edges,
                std::vector<int> &sorted_curve_nodes, std::string type) {
  std::cout << "Sorting the curve for " << type << "...\n";
  auto coords = mesh.coords();
  auto node2sim = mesh.get_array<Omega_h::LO>(0, "simNumbering");

  int starting_node = -1;

  int nEdges_onCurve = 0; // number of nodes are the same

  if (type == "square") {
    starting_node = get_lower_left_corner(mesh);
    std::cout << "Starting node for the square curve: " << starting_node
              << "\n";
    Omega_h::Read<Omega_h::I8> exposed_sides =
        Omega_h::mark_exposed_sides(&mesh);
    for (int i = 0; i < exposed_sides.size(); ++i) {
      if (exposed_sides[i] == 1) {
        nEdges_onCurve++;
      }
    }
    std::cout << "Number of square edges: " << nEdges_onCurve << "\n";
  } else if (type == "wall") {
    starting_node = get_closest_wall_node(mesh);
    std::cout << "Starting node for the wall curve: " << starting_node << "\n";
    nEdges_onCurve = get_number_of_wall_nodes(mesh);
    std::cout << "Number of wall nodes: " << nEdges_onCurve << "\n";
  } else {
    std::cerr << "Unknown curve type. Exiting...\n";
    return;
  }
  assert(starting_node != -1);
  int current_node = starting_node;

  if (type == "wall") {
    std::cout << "Closest node to the vertical axis: " << starting_node
              << " and simNumbering is " << node2sim[starting_node] << "\n";
    std::cout << "Closest Node Coordinates: "
              << Omega_h::get_vector<2>(coords, starting_node)[0] << " "
              << Omega_h::get_vector<2>(coords, starting_node)[1] << "\n";
  }

  // node to edge map
  auto node2edge = mesh.ask_up(0, 1);
  auto node2edgeEdges = node2edge.ab2b;
  auto node2edgeOffsets = node2edge.a2ab;

  sorted_curve_edges.resize(nEdges_onCurve);
  sorted_curve_nodes.resize(nEdges_onCurve);

  if (type == "square") {
    walk_on_curve_square_curve(mesh, sorted_curve_edges, sorted_curve_nodes,
                               starting_node);
  } else if (type == "wall") {
    walk_on_curve_wall_curve(mesh, sorted_curve_edges, sorted_curve_nodes,
                             starting_node);
  }
}

void walk_on_curve_square_curve(Omega_h::Mesh &mesh,
                                std::vector<int> &sorted_curve_edges,
                                std::vector<int> &sorted_curve_nodes,
                                int starting_node) {
  auto exposed_sides = Omega_h::mark_exposed_sides(&mesh);
  auto node2edge = mesh.ask_up(0, 1);
  auto node2edgeEdges = node2edge.ab2b;
  auto node2edgeOffsets = node2edge.a2ab;

  auto edge2node = mesh.get_adj(Omega_h::EDGE, Omega_h::VERT);
  auto edge2nodeNodes = edge2node.ab2b;

  auto coords = mesh.coords();

  sorted_curve_nodes[0] = starting_node;
  int current_node = starting_node;

  for (int i = 0; i < sorted_curve_nodes.size(); ++i) {
    // get the adjacent edges to the closest node
    int nAdjacentEdges =
        node2edgeOffsets[current_node + 1] - node2edgeOffsets[current_node];
    std::vector<int> adjacentEdges(nAdjacentEdges);
    for (int j = 0; j < nAdjacentEdges; ++j) {
      adjacentEdges[j] = node2edgeEdges[node2edgeOffsets[current_node] + j];
    }

    // get the 2 adjacent edges that are exposed
    std::vector<int> exposedAdjacentEdges;
    for (int j = 0; j < nAdjacentEdges; ++j) {
      if (exposed_sides[adjacentEdges[j]]) {
        exposedAdjacentEdges.push_back(adjacentEdges[j]);
      }
    }

    // get the other 2 nodes of the exposed edges
    std::vector<int> exposedAdjacentNodes(2);
    for (int j = 0; j < 2; ++j) {
      // get nodes of the edge
      std::vector<int> edge_nodes = {
          edge2nodeNodes[2 * exposedAdjacentEdges[j]],
          edge2nodeNodes[2 * exposedAdjacentEdges[j] + 1]};
      if (edge_nodes[0] == current_node) {
        exposedAdjacentNodes[j] = edge_nodes[1];
      } else {
        exposedAdjacentNodes[j] = edge_nodes[0];
      }
    }
    int next_edge = -1;
    int next_node = -1;
    if (i == 0) { // for the initial case it needs to get the upward node
      // get current node coordinates
      auto current_node_coords = Omega_h::get_vector<2>(coords, current_node);
      // get the coordinates of the exposed adjacent nodes
      auto exposed_node1_coords =
          Omega_h::get_vector<2>(coords, exposedAdjacentNodes[0]);
      auto exposed_node2_coords =
          Omega_h::get_vector<2>(coords, exposedAdjacentNodes[1]);

      // next node will be the one that is right of the current node : 1st
      // coordinate is higher

      if ((abs(exposed_node1_coords[0] - exposed_node2_coords[0])) > 1e-6) {
        next_node = exposedAdjacentNodes[0];
        next_edge = exposedAdjacentEdges[0];
      } else {
        next_node = exposedAdjacentNodes[1];
        next_edge = exposedAdjacentEdges[1];
      }
      // store the boundary edge and node
      sorted_curve_edges[0] = next_edge;
      sorted_curve_nodes[1] = next_node;
    } else {
      // get the next node and edge based on if it is already the last edge or
      // node

      int prev_node = sorted_curve_nodes[i - 1];
      int prev_edge = sorted_curve_edges[i - 1];
      // among exposedAdjacentNodes, and exposedAdjacentEdges, get the one that
      // is not prev_node
      for (int j = 0; j < 2; ++j) {
        if (exposedAdjacentNodes[j] != prev_node) {
          next_node = exposedAdjacentNodes[j];
          next_edge = exposedAdjacentEdges[j];
          break;
        }
      }
      // store the boundary edge and node
      sorted_curve_edges[i] = next_edge;
      // if it is the last one then the node is already stored
      if (i != sorted_curve_nodes.size() - 1) {
        sorted_curve_nodes[i + 1] = next_node;
      }
    }

    // std::cout << "Current node: " << current_node << " Next node: " <<
    // next_node << "\n"; std::cout << "Next edge: " << next_edge << "\n";

    // update the current node
    current_node = next_node;
  }
}

int get_lower_left_corner(Omega_h::Mesh &mesh) {
  // get the coordinates of the vertices
  auto coords = mesh.coords();
  // loop thourgh all the vertices and save the i of the lower left corner:
  // both x and y are the smallest
  int closest_node = 0;
  double distance = 1e10;
  for (Omega_h::LO i = 0; i < mesh.nverts(); ++i) {
    auto vcoords = Omega_h::get_vector<2>(coords, i);
    // the 1st coordinate is distance from the vertical axis
    if (vcoords[0] < distance && vcoords[1] < distance) {
      distance = vcoords[0];
      closest_node = i;
    }
  }
  OMEGA_H_CHECK(distance < 1e10);
  OMEGA_H_CHECK(closest_node < mesh.nverts());
  OMEGA_H_CHECK(distance > 0.0); // tomms mesh only resides in positive x

  return closest_node;
}

int get_closest_wall_node(Omega_h::Mesh &mesh) {
  auto coords = mesh.coords();

  auto onwall = mesh.get_array<Omega_h::LO>(0, "isOnWall");
  int closest_node = 0;
  double distance = 1e10;

  for (Omega_h::LO i = 0; i < mesh.nverts(); ++i) {
    if (onwall[i] == 1) {
      auto vcoords = Omega_h::get_vector<2>(coords, i);
      if (vcoords[0] < distance) {
        distance = vcoords[0];
        closest_node = i;
      }
    }
  }
  OMEGA_H_CHECK(distance < 1e10);
  OMEGA_H_CHECK(closest_node < mesh.nverts());
  OMEGA_H_CHECK(distance > 0.0); // tomms mesh only resides in positive x

  return closest_node;
}

int get_number_of_wall_nodes(Omega_h::Mesh &mesh) {
  auto onwall = mesh.get_array<Omega_h::LO>(0, "isOnWall");
  int nNodes_onWall = 0;
  for (Omega_h::LO i = 0; i < mesh.nverts(); ++i) {
    if (onwall[i] == 1) {
      nNodes_onWall++;
    }
  }
  return nNodes_onWall;
}

void walk_on_curve_wall_curve(Omega_h::Mesh &mesh,
                              std::vector<int> &sorted_curve_edges,
                              std::vector<int> &sorted_curve_nodes,
                              int starting_node) {
  auto onwall = mesh.get_array<Omega_h::LO>(0, "isOnWall");
  auto edge_class_id = mesh.get_array<Omega_h::I32>(1, "class_id");
  auto node2edge = mesh.ask_up(0, 1);
  auto node2edgeEdges = node2edge.ab2b;
  auto node2edgeOffsets = node2edge.a2ab;

  auto edge2node = mesh.get_adj(Omega_h::EDGE, Omega_h::VERT);
  auto edge2nodeNodes = edge2node.ab2b;

  auto coords = mesh.coords();

  sorted_curve_nodes[0] = starting_node;
  int current_node = starting_node;
  int current_edge = -1;

  for (int i = 0; i < sorted_curve_nodes.size(); ++i) {
    // get the adjacent edges to the closest node
    int nAdjacentEdges =
        node2edgeOffsets[current_node + 1] - node2edgeOffsets[current_node];
    std::vector<int> adjacentEdges(nAdjacentEdges);
    for (int j = 0; j < nAdjacentEdges; ++j) {
      adjacentEdges[j] = node2edgeEdges[node2edgeOffsets[current_node] + j];
    }

    // get the 2 adjacent edges that are on the wall
    std::vector<int> wallAdjacentEdges;
    for (int j = 0; j < nAdjacentEdges; ++j) {
      if (onwall[edge2nodeNodes[2 * adjacentEdges[j]]] &&
          onwall[edge2nodeNodes[2 * adjacentEdges[j] + 1]]) {
        wallAdjacentEdges.push_back(adjacentEdges[j]);
      }
    }
    if (wallAdjacentEdges.size() < 2) {
      std::cerr << "There should be 2 or more wall edges adjacent to a wall "
                   "node. Has: "
                << wallAdjacentEdges.size() << " Exiting...\n";
      return;
    }
    if (wallAdjacentEdges.size() > 2 && i != 0) {
      // keep 2 that has the closest class_id
      std::vector<int> closeness(wallAdjacentEdges.size());
      int current_edge_class_id = edge_class_id[current_edge];
      for (int j = 0; j < wallAdjacentEdges.size(); ++j) {
        closeness[j] =
            abs(edge_class_id[wallAdjacentEdges[j]] - current_edge_class_id);
      }
      // get the least 2 closeness
      std::vector<int> least_closeness(2);
      std::pair<int, int> closest_edge_indices =
          findTwoSmallestIndices(closeness);
      std::vector<int> closeness_based_wallAdjacentEdges(2);
      closeness_based_wallAdjacentEdges[0] =
          wallAdjacentEdges[closest_edge_indices.first];
      closeness_based_wallAdjacentEdges[1] =
          wallAdjacentEdges[closest_edge_indices.second];
      wallAdjacentEdges = closeness_based_wallAdjacentEdges;
    }

    // get the other 2 nodes of the wall edges
    std::vector<int> wallAdjacentNodes(2);
    for (int j = 0; j < 2; ++j) {
      // get nodes of the edge
      std::vector<int> edge_nodes = {
          edge2nodeNodes[2 * wallAdjacentEdges[j]],
          edge2nodeNodes[2 * wallAdjacentEdges[j] + 1]};
      if (edge_nodes[0] == current_node) {
        wallAdjacentNodes[j] = edge_nodes[1];
      } else {
        wallAdjacentNodes[j] = edge_nodes[0];
      }
    }

    // for the 1st itertation, get the next node that is below the current node:
    // 2nd coordinate is lower
    int next_node = -1;
    int next_edge = -1;

    if (i == 0) {
      if (Omega_h::get_vector<2>(coords, wallAdjacentNodes[0])[1] <
          Omega_h::get_vector<2>(coords, wallAdjacentNodes[1])[1]) {
        next_node = wallAdjacentNodes[0];
        next_edge = wallAdjacentEdges[0];
      } else {
        next_node = wallAdjacentNodes[1];
        next_edge = wallAdjacentEdges[1];
      }
      sorted_curve_edges[0] = next_edge;
      sorted_curve_nodes[1] = next_node;
      current_node = next_node;
      current_edge = next_edge;
    } else {
      // get the next node and edge based on if it is already the last edge or
      // node

      int prev_node = sorted_curve_nodes[i - 1];
      // among exposedAdjacentNodes, and exposedAdjacentEdges, get the one that
      // is not prev_node
      for (int j = 0; j < 2; ++j) {
        if (wallAdjacentNodes[j] != prev_node) {
          next_node = wallAdjacentNodes[j];
          next_edge = wallAdjacentEdges[j];
          break;
        }
      }
      // store the boundary edge and node
      sorted_curve_edges[i] = next_edge;
      sorted_curve_nodes[i + 1] = next_node;
      current_node = next_node;
      current_edge = next_edge;
      // if it is the last one then the node is already stored
      if (i != sorted_curve_nodes.size() - 1) {
        sorted_curve_nodes[i + 1] = next_node;
      }
    }
  }
}

std::pair<int, int> findTwoSmallestIndices(const std::vector<int> &vec) {
  if (vec.size() < 2) {
    throw std::invalid_argument(
        "The vector should contain at least two elements.");
  }

  // Initialize the minimum and second minimum values and their indices
  int min1 = std::numeric_limits<int>::max(),
      min2 = std::numeric_limits<int>::max();
  int min1Index = -1, min2Index = -1;

  for (int i = 0; i < vec.size(); ++i) {
    if (vec[i] < min1) {
      // Update second minimum before updating the minimum
      min2 = min1;
      min2Index = min1Index;
      // Update minimum
      min1 = vec[i];
      min1Index = i;
    } else if (vec[i] < min2) {
      // Update second minimum
      min2 = vec[i];
      min2Index = i;
    }
  }

  return {min1Index, min2Index};
}