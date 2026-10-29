#include "bindings.hpp"

#include "omegah2csg/options.hpp"

#include <pybind11/stl.h>

namespace omegah2csg {

void bind_options(pybind11::module_ &m) {
  pybind11::enum_<SurfaceType>(m, "SurfaceType")
      .value("ZPlane", SurfaceType::ZPlane)
      .value("ZCylinder", SurfaceType::ZCylinder)
      .value("ZCone", SurfaceType::ZCone);
  pybind11::enum_<BoundaryType>(m, "BoundaryType")
      .value("Transmission", BoundaryType::Transmission)
      .value("Vacuum", BoundaryType::Vacuum)
      .value("Reflective", BoundaryType::Reflective)
      .value("White", BoundaryType::White);
  pybind11::enum_<NcFormat>(m, "NcFormat")
      .value("Auto", NcFormat::Auto)
      .value("Cdf2", NcFormat::Cdf2)
      .value("Cdf5", NcFormat::Cdf5)
      .value("NetCdf4", NcFormat::NetCdf4);

  pybind11::class_<CommonOptions>(m, "CommonOptions")
      .def_readwrite("tol", &CommonOptions::tol)
      .def_readwrite("verbosity", &CommonOptions::verbosity)
      .def_readwrite("timing", &CommonOptions::timing)
      .def_readwrite("debug_entities", &CommonOptions::debug_entities)
      .def_readwrite("force", &CommonOptions::force)
      .def_readwrite("output", &CommonOptions::output);
  pybind11::class_<OpenMCOptions, CommonOptions>(m, "OpenMCOptions")
      .def(pybind11::init<>())
      .def_readwrite("material_ids_file", &OpenMCOptions::material_ids_file)
      .def_readwrite("boundary_type", &OpenMCOptions::boundary_type)
      .def_readwrite("cell_id_offset", &OpenMCOptions::cell_id_offset)
      .def_readwrite("surface_id_offset", &OpenMCOptions::surface_id_offset)
      .def("validate", [](const OpenMCOptions &o) { validate(o); });
  pybind11::class_<Degas2Options, CommonOptions>(m, "Degas2Options")
      .def(pybind11::init<>())
      .def_readwrite("aux_files", &Degas2Options::aux_files)
      .def_readwrite("aux_dir", &Degas2Options::aux_dir)
      .def_readwrite("target_temperature_K",
                     &Degas2Options::target_temperature_K)
      .def_readwrite("target_recyc_coef", &Degas2Options::target_recyc_coef)
      .def_readwrite("nc_format", &Degas2Options::nc_format)
      .def("validate", [](const Degas2Options &o) { validate(o); });
}

} // namespace omegah2csg
