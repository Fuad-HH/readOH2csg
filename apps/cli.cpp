#include "cli.hpp"

#include "omegah2csg/errors.hpp"
#include "omegah2csg/mesh_input.hpp"
#include "omegah2csg/options.hpp"
#include "omegah2csg/profiling.hpp"
#include "omegah2csg/runtime_info.hpp"

#include <algorithm>
#include <cstdio>

namespace omegah2csg {

namespace {
struct UsageError : std::runtime_error {
  explicit UsageError(const std::string &what) : std::runtime_error(what) {}
};

bool starts_with(const std::string &s, const char *prefix) {
  return s.rfind(prefix, 0) == 0;
}

void print_help(const char *prog, const std::string &about,
                const std::vector<Flag> &flags) {
  std::printf("usage: %s MESH.osh [options]\n\n%s\n\noptions:\n", prog,
              about.c_str());
  for (const auto &f : flags) {
    const std::string name = (f.short_name.empty() ? "" : f.short_name + ", ") +
                             f.name +
                             (f.metavar.empty() ? "" : " " + f.metavar);
    std::printf("  %-30s %s%s\n", name.c_str(), f.help.c_str(),
                f.default_value.empty()
                    ? ""
                    : (" (default: " + f.default_value + ")").c_str());
  }
  std::printf("  %-30s %s\n  %-30s %s\n  %-30s %s\n", "--check",
              "validate the mesh and exit", "--version",
              "print the version (with -v: build configuration)", "-h, --help",
              "show this help");
}

/// Parses argv into `o`; returns the mesh path ("" for --help/--version).
template <class Options>
std::string parse(int argc, char **argv, const std::string &about, Options &o,
                  bool &check) {
  auto fl = flags(o);
  std::string mesh;
  bool version = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "-h" || a == "--help") {
      print_help(argv[0], about, fl);
      return "";
    }
    if (a == "--version" || a == "--check") {
      (a == "--check" ? check : version) = true;
      continue;
    }
    if (starts_with(a, "--kokkos-") || starts_with(a, "--osh-"))
      continue;
    if (a.size() > 2 && a.find_first_not_of('v', 1) == std::string::npos) {
      o.verbosity += static_cast<int>(a.size()) - 1; // -vv
      continue;
    }
    const auto f = std::find_if(fl.begin(), fl.end(), [&](const Flag &x) {
      return a == x.name || (!x.short_name.empty() && a == x.short_name);
    });
    if (f == fl.end()) {
      if (a[0] == '-' || !mesh.empty())
        throw UsageError("unexpected argument '" + a + "'");
      mesh = a;
    } else if (f->metavar.empty()) {
      f->set("");
    } else if (++i < argc) {
      f->set(argv[i]);
    } else {
      throw UsageError(a + " needs a value");
    }
  }
  if (version) {
    std::printf("omegah2csg %s\n", runtime_info().at("omegah2csg").c_str());
    if (o.verbosity > 1)
      for (const auto &[key, value] : runtime_info())
        std::printf("  %-16s %s\n", key.c_str(), value.c_str());
    return "";
  }
  if (mesh.empty())
    throw UsageError("missing MESH.osh (see --help)");
  return mesh;
}
} // namespace

template <class Options>
int run_cli(int argc, char **argv, const std::string &about,
            void (*convert)(Omega_h::Library &, const std::string &,
                            const Options &),
            bool need_wall_tags) {
  Omega_h::Library lib(&argc, &argv);
  try {
    Options o;
    bool check = false;
    const std::string mesh = parse(argc, argv, about, o, check);
    if (mesh.empty())
      return 0;
    check_single_rank(lib);
    if (check) {
      auto m = read_mesh(lib, mesh);
      check_required_tags(m, need_wall_tags);
      std::printf("%s: ok\n", mesh.c_str());
      return 0;
    }
    convert(lib, mesh, o);
    if (o.timing)
      std::fputs(format_timing().c_str(), stdout);
    if (o.verbosity > 0)
      std::fprintf(stderr, "wrote %s\n", o.output.c_str());
    return 0;
  } catch (const UsageError &e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  } catch (const InputError &e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  } catch (const GeometryError &e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 3;
  }
}

template int run_cli<OpenMCOptions>(int, char **, const std::string &,
                                    void (*)(Omega_h::Library &,
                                             const std::string &,
                                             const OpenMCOptions &),
                                    bool);
template int run_cli<Degas2Options>(int, char **, const std::string &,
                                    void (*)(Omega_h::Library &,
                                             const std::string &,
                                             const Degas2Options &),
                                    bool);

} // namespace omegah2csg
