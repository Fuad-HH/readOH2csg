#include "bindings.hpp"

#include "omegah2csg/runtime_info.hpp"

#include <Kokkos_Core.hpp>
#include <Omega_h_fence.hpp>

#include <memory>

namespace omegah2csg {

namespace {
std::shared_ptr<Omega_h::Library> &instance() {
  static std::shared_ptr<Omega_h::Library> lib;
  return lib;
}

// Runs after Python objects are gone, so no device array outlives the
// Library. The Library finalizes Kokkos/MPI only if it started them (it
// defers to PyOmega_h when that was imported first).
void release() { instance().reset(); }

void remember(std::shared_ptr<Omega_h::Library> lib) {
  instance() = std::move(lib);
  Py_AtExit(release);
}
} // namespace

Omega_h::Library &library() {
  if (!instance())
    remember(std::make_shared<Omega_h::Library>());
  return *instance();
}

void initialize(const std::vector<std::string> &kokkos_args) {
  if (instance() || Kokkos::is_initialized())
    throw std::runtime_error("omegah2csg runtime (or Kokkos) already "
                             "initialized; call initialize() first");
  std::vector<std::string> storage{"omegah2csg"};
  storage.insert(storage.end(), kokkos_args.begin(), kokkos_args.end());
  std::vector<char *> argv;
  for (auto &s : storage)
    argv.push_back(s.data());
  int argc = static_cast<int>(argv.size());
  char **argv_ptr = argv.data();
  remember(std::make_shared<Omega_h::Library>(&argc, &argv_ptr));
}

void require_same_libomega_h(const Omega_h::Mesh &mesh) {
  const void *vtable = *reinterpret_cast<const void *const *>(&mesh);
  const auto mine =
      library_path_of(reinterpret_cast<const void *>(&Omega_h::fence));
  const auto theirs = library_path_of(vtable);
  if (mine != theirs)
    throw pybind11::type_error(
        "this mesh comes from " + theirs + " but omegah2csg uses " + mine +
        "; both must use the same libomega_h (pass a file path instead)");
}

} // namespace omegah2csg
