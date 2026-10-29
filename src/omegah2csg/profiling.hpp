#ifndef OMEGAH2CSG_PROFILING_HPP
#define OMEGAH2CSG_PROFILING_HPP
/// @file profiling.hpp
/// @brief Per-stage timing (Kokkos profiling regions + wall time).

#include <string>
#include <utility>
#include <vector>

namespace omegah2csg {

/// Times one stage from construction to destruction (fenced).
class StageTimer {
public:
  explicit StageTimer(std::string stage);
  ~StageTimer();
  StageTimer(const StageTimer &) = delete;
  StageTimer &operator=(const StageTimer &) = delete;

private:
  std::string stage_;
  double start_;
};

/// (stage, seconds) in completion order; cleared by clear_timing().
const std::vector<std::pair<std::string, double>> &timing_report();
void clear_timing();
std::string format_timing();

} // namespace omegah2csg

#endif
