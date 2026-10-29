#include "omegah2csg/io/atomic_file.hpp"
#include "omegah2csg/errors.hpp"

#include <unistd.h>

#include <filesystem>

namespace omegah2csg {

AtomicOutput::AtomicOutput(std::string path, bool force)
    : path_(std::move(path)),
      tmp_(path_ + ".tmp." + std::to_string(::getpid())) {
  const auto dir = std::filesystem::absolute(path_).parent_path();
  if (!std::filesystem::is_directory(dir))
    throw InputError("output directory does not exist: " + dir.string());
  if (!force && std::filesystem::exists(path_))
    throw InputError(path_ + " exists; use --force to overwrite");
}

AtomicOutput::~AtomicOutput() {
  if (!committed_) {
    std::error_code ec;
    std::filesystem::remove(tmp_, ec);
  }
}

void AtomicOutput::commit() {
  std::filesystem::rename(tmp_, path_);
  committed_ = true;
}

} // namespace omegah2csg
