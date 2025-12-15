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
#include <Omega_h_cmdline.hpp>
#include <Omega_h_file.hpp>
#include <Omega_h_for.hpp>
#include <Omega_h_mark.hpp>
#include <netcdf>

#include "Timer.h"
#include "compute_surface.h"

void read_cli(int argc, char **argv, Omega_h::Library &lib, bool &print_flag,
              Omega_h::filesystem::path &mesh_filename);

/*!
 * \brief Read the mesh file and go to each vertex to get its coordinates
 */
int main(int argc, char **argv) {
  Timers timers;
  Timer *main_timer = timers.add("Total");
  Timer *setup_timer = timers.add("Setup & load mesh");

  auto lib = Omega_h::Library(&argc, &argv);
  Omega_h::Mesh mesh(&lib);

  bool print_flag;
  Omega_h::filesystem::path mesh_filename;
  read_cli(argc, argv, lib, print_flag, mesh_filename);

  Omega_h::binary::read(mesh_filename, lib.world(), &mesh);
  setup_timer->stop();

  Timer *boundary_timer = timers.add("Find boundary edges");
  Omega_h::LOs boundary_edge_ids = get_boundary_edge_ids(mesh);
  boundary_timer->stop();

  Timer *edge_query_timer = timers.add("Calculate edge coefficients");
  // Kokkos view to store nedges by 6 doubles :
  // m^2, tba, 2c, -c^2, top-bottom, m
  auto edge_coefficients_v =
      Kokkos::View<double *[6]>("edge_coefficients_view", mesh.nedges());

  compute_edge_coefficients(mesh, edge_coefficients_v, print_flag);
  edge_query_timer->stop();

  Timer *edge_file_write_timer = timers.add("edge file write");
  std::ofstream edge_coefficient_file;
  edge_coefficient_file.open("edge-coefficients.dat");

  auto coefficinets_host = Kokkos::create_mirror_view(edge_coefficients_v);
  Kokkos::deep_copy(coefficinets_host, edge_coefficients_v);
  for (Omega_h::LO i = 0; i < mesh.nedges(); ++i) {
    edge_coefficient_file << i << " " << coefficinets_host(i, 0) << " "
                          << coefficinets_host(i, 1) << " "
                          << coefficinets_host(i, 2) << " "
                          << coefficinets_host(i, 3) << " "
                          << coefficinets_host(i, 4) << "\n";
  }
  // the last line contains all the boundary edges
  auto bdrs_host = Omega_h::HostRead<Omega_h::LO>(boundary_edge_ids);
  edge_coefficient_file << "Boundary edges: " << bdrs_host.size() << "\n";
  for (int bdr : bdrs_host) {
    edge_coefficient_file << bdr << " ";
  }
  edge_coefficient_file.close();
  edge_file_write_timer->stop();

  Timer *face_calc_timer = timers.add("Determine face-edge connectivity");
  Kokkos::View<int *[6]> face2edge_connectivity =
      calculate_face_connectivity(mesh, edge_coefficients_v, print_flag);
  face_calc_timer->stop();

  Timer *face_file_write_timer = timers.add("Connectivity file write");
  std::ofstream face2edge_connectivity_file;
  face2edge_connectivity_file.open("face-edge-connectivity.dat");

  auto connectivity_host = Kokkos::create_mirror_view(face2edge_connectivity);
  Kokkos::deep_copy(connectivity_host, face2edge_connectivity);
  for (Omega_h::LO i = 0; i < mesh.nfaces(); ++i) {
    face2edge_connectivity_file
        << i << " " << connectivity_host(i, 0) << " " << connectivity_host(i, 1)
        << " " << connectivity_host(i, 2) << " " << connectivity_host(i, 3)
        << " " << connectivity_host(i, 4) << " " << connectivity_host(i, 5)
        << "\n";
  }
  face2edge_connectivity_file.close();
  face_file_write_timer->stop();

  /*
  // *********** Write to netcdf file *********** //
  // ******************************************** //
  // create a netcdf file
  Timer *nc_file_write_timer = timers.add("netcdf file write");
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
    } else { // 0 if only one face
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

  nc_file_write_timer->stop();
 */
  main_timer->stop();

  timers.print();
  return 0;
} // main

// The lib needs to be passed as reference otherwise it cannot read the mesh
void read_cli(int argc, char **argv, Omega_h::Library &lib, bool &print_flag,
              Omega_h::filesystem::path &mesh_filename) {
  Omega_h::CmdLine cmdline;
  auto &mesh_flag = cmdline.add_flag("--mesh", "Input mesh (.osh) file path");
  mesh_flag.add_arg<std::string>("mesh_filename");
  cmdline.add_flag("--print", "Print details");
  if (!cmdline.parse_final(lib.world(), &argc, argv)) {
    throw std::runtime_error("Error in command line parsing");
  }
  if (!cmdline.parsed("--mesh")) {
    cmdline.show_help(argv);
    throw std::runtime_error("Error parsing command line arguments");
  }
  if (!cmdline.parsed("--print")) {
    print_flag = false;
  } else {
    print_flag = true;
  }
  mesh_filename = cmdline.get<std::string>("--mesh", "mesh_filename");
  if (!exists(mesh_filename)) {
    const std::string msg =
        "Mesh file " + mesh_filename.string() + " not found";
    throw std::runtime_error(msg);
  }
}
