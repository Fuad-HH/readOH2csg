#ifndef OMEGAH2CSG_IO_ATOMIC_FILE_HPP
#define OMEGAH2CSG_IO_ATOMIC_FILE_HPP
/// @file atomic_file.hpp
/// @brief Write to a temporary file, then rename into place on commit().

#include <string>

namespace omegah2csg {

class AtomicOutput {
public:
  /// Throws InputError if `path` exists and `force` is false, or if its
  /// directory does not exist.
  AtomicOutput(std::string path, bool force);
  ~AtomicOutput(); ///< removes the temporary file unless committed
  AtomicOutput(const AtomicOutput &) = delete;
  AtomicOutput &operator=(const AtomicOutput &) = delete;

  const std::string &tmp() const { return tmp_; }
  void commit();

private:
  std::string path_, tmp_;
  bool committed_ = false;
};

} // namespace omegah2csg

#endif
