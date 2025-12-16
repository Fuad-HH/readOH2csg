/**
 *
 * @file compute_surface.h
 * @brief Helper functions to compute CSG surfaces from edges
 *
 */

#ifndef OMEGA_H_READ_COMPUTE_SURFACE_H
#define OMEGA_H_READ_COMPUTE_SURFACE_H

#include <Omega_h_mesh.hpp>

Kokkos::View<int *[6]>
calculate_face_connectivity(Omega_h::Mesh mesh,
                            Kokkos::View<double *[6]> edge_coefficients_v,
                            bool print_flag = false, double tol = 1e-6);

/**
 *
 * @param mesh Input mesh
 * @param print_flag Print flag for debugging
 * @param edge_coefficients_v Output edge coefficients of size (num_edges, 6)
 * @param tol Tolerance for numerical comparisons
 */
void compute_edge_coefficients(Omega_h::Mesh &mesh,
                               Kokkos::View<double *[6]> edge_coefficients_v,
                               bool print_flag = false, double tol = 1e-10);

[[nodiscard]] Omega_h::LOs get_boundary_edge_ids(Omega_h::Mesh &mesh);

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
Omega_h::Vector<6> compute_coefficients(const Omega_h::Few<double, 2> &vert1,
                                        const Omega_h::Few<double, 2> &vert2);
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
                 Omega_h::Vector<2> vert3, std::vector<double> edgeCoeffs,
                 double tol = 1e-6);

/**
 * @brief Determines if a point is above or below a line
 *
 * @param point The point to check
 * @param coeffs {m, c} The coefficients of the line
 * @param tol Tolerance for numerical comparisons
 * @return int 1 if above, -1 if below
 */
int above_or_below_line(Omega_h::Vector<2> point, std::vector<double> coeffs,
                        double tol = 1e-6);

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

#endif // OMEGA_H_READ_COMPUTE_SURFACE_H
