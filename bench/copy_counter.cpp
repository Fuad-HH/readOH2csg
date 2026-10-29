/// @file copy_counter.cpp
/// @brief Kokkos Tools plugin: counts deep copies between different memory
/// spaces and prints {"cross_space_copies": N, "bytes": B} at exit (to the
/// file named by OMEGAH2CSG_COPY_COUNTER_OUT, else stderr).

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
std::uint64_t copies = 0, bytes = 0;
} // namespace

struct SpaceHandle {
  char name[64];
};

extern "C" void kokkosp_begin_deep_copy(SpaceHandle dst, const char *,
                                        const void *, SpaceHandle src,
                                        const char *, const void *,
                                        std::uint64_t size) {
  if (std::strcmp(dst.name, src.name) != 0 && size > 0) {
    ++copies;
    bytes += size;
  }
}

extern "C" void kokkosp_finalize_library() {
  const char *path = std::getenv("OMEGAH2CSG_COPY_COUNTER_OUT");
  std::FILE *out = path ? std::fopen(path, "w") : stderr;
  if (!out)
    return;
  std::fprintf(out, "{\"cross_space_copies\": %llu, \"bytes\": %llu}\n",
               static_cast<unsigned long long>(copies),
               static_cast<unsigned long long>(bytes));
  if (out != stderr)
    std::fclose(out);
}
