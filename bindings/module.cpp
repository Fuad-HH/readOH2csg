#include "bindings.hpp"

#include "omegah2csg/errors.hpp"

PYBIND11_MODULE(_core, m) {
  m.doc() = "omegah2csg C++/Kokkos pipeline";
  pybind11::register_exception<omegah2csg::GeometryError>(m, "GeometryError",
                                                          PyExc_RuntimeError);
  pybind11::register_exception<omegah2csg::InputError>(m, "InputError",
                                                       PyExc_ValueError);
  omegah2csg::bind_options(m);
  omegah2csg::bind_results(m);
  omegah2csg::bind_pipeline(m);
  m.attr("__version__") = OMEGAH2CSG_VERSION;
}
