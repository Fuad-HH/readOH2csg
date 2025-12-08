/**
 *@file compute_surface.cpp
 */

#include "compute_surface.h"

#include <Omega_h_mark.hpp>

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
