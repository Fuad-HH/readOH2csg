#include "cli.hpp"

#include "omegah2csg/pipeline.hpp"

int main(int argc, char **argv) {
  return omegah2csg::run_cli<omegah2csg::Degas2Options>(
      argc, argv,
      "Convert an Omega_h mesh (with offset_face and isOnWall tags) to a "
      "DEGAS2 geometry.nc.",
      &omegah2csg::convert_to_degas2, true);
}
