#include "omegah2csg/profiling.hpp"

#include <Kokkos_Core.hpp>

#include <chrono>
#include <cstdio>

namespace omegah2csg {

namespace {
std::vector<std::pair<std::string, double>> &report() {
  static std::vector<std::pair<std::string, double>> r;
  return r;
}

double now() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
} // namespace

StageTimer::StageTimer(std::string stage) : stage_(std::move(stage)) {
  Kokkos::fence("omegah2csg::" + stage_);
  Kokkos::Profiling::pushRegion("omegah2csg::" + stage_);
  start_ = now();
}

StageTimer::~StageTimer() {
  Kokkos::fence("omegah2csg::" + stage_);
  report().emplace_back(stage_, now() - start_);
  Kokkos::Profiling::popRegion();
}

const std::vector<std::pair<std::string, double>> &timing_report() {
  return report();
}

void clear_timing() { report().clear(); }

std::string format_timing() {
  std::string out;
  char line[128];
  for (const auto &[stage, seconds] : report()) {
    std::snprintf(line, sizeof line, "%-24s %10.4f s\n", stage.c_str(),
                  seconds);
    out += line;
  }
  return out;
}

} // namespace omegah2csg
