#include <Kokkos_Core.hpp>
#include <Omega_h_fail.hpp>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <Omega_h_bbox.hpp>
#include <Omega_h_file.hpp>
#include <Omega_h_for.hpp>
#include <Omega_h_mark.hpp>
#include <netcdf>

#include "compute_surface.h"

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
