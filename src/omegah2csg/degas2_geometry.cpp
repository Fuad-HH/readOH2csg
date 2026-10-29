#include "omegah2csg/degas2_geometry.hpp"
#include "omegah2csg/transfer.hpp"

namespace omegah2csg {

namespace {
/// Shares ownership of a container while pointing at its elements.
template <class C> std::shared_ptr<const void> share(C value) {
  auto owner = std::make_shared<const C>(std::move(value));
  return std::shared_ptr<const void>(owner, owner->data());
}

DType dtype(const Omega_h::LOs &) { return DType::I32; }
DType dtype(const std::vector<std::int32_t> &) { return DType::I32; }
DType dtype(const Omega_h::Reals &) { return DType::F64; }
DType dtype(const std::vector<double> &) { return DType::F64; }
DType dtype(const std::string &) { return DType::Char; }

template <class T> std::shared_ptr<const void> load(const Omega_h::Read<T> &d) {
  return share(to_host(d));
}
template <class C> std::shared_ptr<const void> load(const C &d) {
  return share(d);
}
} // namespace

HostVar host_var(const Degas2Variable &v, bool want_data) {
  HostVar out;
  out.shape = v.shape;
  std::visit(
      [&](const auto &d) {
        out.type = dtype(d);
        if (want_data)
          out.data = load(d);
      },
      v.data);
  return out;
}

} // namespace omegah2csg
