#include "omegah2csg/io/degas2_netcdf.hpp"
#include "omegah2csg/degas2_schema.hpp"
#include "omegah2csg/errors.hpp"
#include "omegah2csg/io/atomic_file.hpp"

#include <netcdf.h>

namespace omegah2csg {

namespace {
void check(int status, const std::string &what) {
  if (status != NC_NOERR)
    throw std::runtime_error("netCDF " + what + ": " + nc_strerror(status));
}

/// netCDF file handle closed on destruction.
struct NcFile {
  int id = -1;
  NcFile(const std::string &path, int mode) {
    check(nc_create(path.c_str(), NC_CLOBBER | mode, &id), "create " + path);
  }
  ~NcFile() {
    if (id >= 0)
      nc_close(id);
  }
  void close() {
    check(nc_close(id), "close");
    id = -1;
  }
};

int nc_type_of(DType t) {
  return t == DType::I32 ? NC_INT : (t == DType::F64 ? NC_DOUBLE : NC_CHAR);
}

std::size_t element_size(DType t) {
  return t == DType::I32 ? 4 : (t == DType::F64 ? 8 : 1);
}

int choose_mode(NcFormat format, std::size_t largest_bytes) {
  if (format == NcFormat::Auto)
    format = largest_bytes >= (std::size_t(1) << 32) - 4 ? NcFormat::Cdf5
                                                         : NcFormat::Cdf2;
  if (format == NcFormat::Cdf5)
    return NC_64BIT_DATA;
  if (format == NcFormat::NetCdf4)
    return NC_NETCDF4 | NC_CLASSIC_MODEL;
  return NC_64BIT_OFFSET;
}
} // namespace

void write_degas2_netcdf(
    const std::function<HostVar(const std::string &, bool)> &get,
    const std::string &path, NcFormat format, bool force) {
  std::size_t largest = 0;
  const auto dims = infer_degas2_dims([&](const std::string &name) {
    const auto v = get(name, false);
    largest = std::max(largest, v.size() * element_size(v.type));
    return v.shape;
  });
  AtomicOutput out(path, force);
  NcFile file(out.tmp(), choose_mode(format, largest));
  std::map<std::string, int> dim_ids;
  for (const auto &d : degas2_dims())
    check(nc_def_dim(file.id, d.name, dims.at(d.name), &dim_ids[d.name]),
          std::string("define dimension ") + d.name);
  std::vector<int> var_ids;
  for (const auto &v : degas2_vars()) {
    std::vector<int> ids;
    for (const char *dim : v.dims)
      ids.push_back(dim_ids.at(dim));
    var_ids.emplace_back();
    check(nc_def_var(file.id, v.name, nc_type_of(v.type),
                     static_cast<int>(ids.size()), ids.data(), &var_ids.back()),
          std::string("define variable ") + v.name);
  }
  check(nc_enddef(file.id), "enddef");
  for (std::size_t i = 0; i < degas2_vars().size(); ++i) {
    const auto &spec = degas2_vars()[i];
    const auto v = get(spec.name, true);
    if (v.type != spec.type)
      throw InputError(std::string("DEGAS2 variable '") + spec.name +
                       "' has the wrong data type");
    const void *data = v.data.get();
    const int status =
        v.type == DType::I32 ? nc_put_var_int(file.id, var_ids[i],
                                              static_cast<const int *>(data))
        : v.type == DType::F64
            ? nc_put_var_double(file.id, var_ids[i],
                                static_cast<const double *>(data))
            : nc_put_var_text(file.id, var_ids[i],
                              static_cast<const char *>(data));
    check(status, std::string("write ") + spec.name);
  }
  file.close();
  out.commit();
}

} // namespace omegah2csg
