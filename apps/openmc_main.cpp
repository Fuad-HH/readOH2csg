#include "cli.hpp"

#include "omegah2csg/pipeline.hpp"

int main(int argc, char **argv) {
  return omegah2csg::run_cli<omegah2csg::OpenMCOptions>(
      argc, argv, "Convert an Omega_h mesh to OpenMC geometry.xml.",
      &omegah2csg::convert_to_openmc, false);
}
