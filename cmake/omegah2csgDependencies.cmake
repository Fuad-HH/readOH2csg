# Required dependencies besides Omega_h. Defines OMEGAH2CSG_NETCDF_TARGET.

find_package(pugixml CONFIG REQUIRED)

# netcdf-c ships a CMake config only when built with CMake; otherwise use
# its pkg-config file. Either way it is required.
find_package(netCDF CONFIG QUIET)
if(TARGET netCDF::netcdf)
  set(OMEGAH2CSG_NETCDF_TARGET netCDF::netcdf)
else()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(NETCDF REQUIRED IMPORTED_TARGET netcdf)
  set(OMEGAH2CSG_NETCDF_TARGET PkgConfig::NETCDF)
endif()
