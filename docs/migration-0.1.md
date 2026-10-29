# Migrating from 0.0.x to 0.1

0.1 replaces the ctypes C API with a C++ pipeline and pybind11 bindings.
Removed names have no compatibility shim. Pin `omegah2csg<0.1`, or use the
`release/0.0.x` branch, to keep the old API.

## Python API

| 0.0.x | 0.1 |
|---|---|
| `with OmegaHMesh(path) as m:` | `g = compute_csg(path)` / `d = compute_degas2(path)` |
| `kokkos_runtime`, `KokkosRuntime` | removed (runtime starts on first use; `initialize(kokkos_args)` to configure) |
| `m.dim`, `m.num_entities(d)` | `g.nedges`, `g.nfaces`, `g.nverts` |
| `m.get_node_coordinates()` | `g.node_coordinates` (shape `(n, 2)`) |
| `m.get_edge_coefficients()` | `g.surface_coeffs`, `g.surface_type` |
| `m.get_face_connectivity()` | `g.face_edges`, `g.face_sides` (separate `(nfaces, 3)` arrays) |
| `m.get_boundary_edge_ids()` | `g.boundary_edges`, `g.boundary_sides` (unsigned ids + sides) |
| `m.get_all_geometry_info()` | the `CsgGeometry` properties above |
| `m.get_cell_volumes()` / `get_cell_centroids()` / `get_cell_bounding_boxes()` | `g.cell_volume` / `g.cell_centroid` `(n, 2)` / `g.cell_bbox` `(n, 4)` |
| `m.get_edge_to_face_map()` | `g.edge_faces_sorted` (+ side face first) |
| `m.get_wall_edge_ids()` / `get_wall_adjacent_triangles()` | `g.wall_edges` / `g.wall_faces` `(n, 2)` |
| `m.get_number_of_edges_inside_wall()` | `g.interior_edge_count` |
| `m.get_integer_tag_array(2, "offset_face")` | `g.face_region_tag` |
| `convert2openmc.create_openmc_universe(m, materials)` | `to_openmc_universe(g, materials=...)` (default fill is void, not a dummy material) |
| `convert2openmcXML(filename, tol)` | `convert_to_openmc(path, options)` or `write_openmc_xml(g, path)` |
| `convert2degas2(...)` | `convert_to_degas2(path, options)` or `write_degas2(compute_degas2(path), path)` |
| `is_horizontal`, `get_line_equation`, `create_z_cone`, ... | removed (classification runs in C++) |
| `python -m omegah2csg.convert2openmc` | `convert2openmc` executable |

## Command line

| 0.0.x | 0.1 |
|---|---|
| `convert2degas2 -v` (version) | `--version`; `-v` now means verbose |
| `--auxfiles` | `-a/--aux-files`; new `--aux-dir` (default: directory of `-o`) |
| aux files written to the working directory | written next to `-o` (or `--aux-dir`) |
| outputs overwritten silently | existing outputs need `--force` |
| `convert2openmc` always wrote `./geometry.xml` | `-o/--output` |
| Python API default `tol=1e-6` | single default `1e-10` everywhere |

## Output changes
- OpenMC: reflective boundaries are now set on the boundary surfaces. 0.0.x
  used signed ids as list indices and marked the wrong surfaces. Cell and
  surface ids start at 1, plus optional offsets.
- DEGAS2: written as CDF-2 (64-bit offset) instead of `NETCDF4_CLASSIC`.
  `--nc-format` selects another format.
- DEGAS2: the last zone's `zone_volume` is computed directly. 0.0.x
  subtracted from an uninitialized array.
- DEGAS2: the boundary entry of edge 0 keeps its sign. 0.0.x wrote 0.
