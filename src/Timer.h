/**
 * @file Timer.h
 * @brief Simple Class to time code execution.
 *
 */

#ifndef OMEGA_H_READ_TIMER_H
#define OMEGA_H_READ_TIMER_H

#include <chrono>
#include <memory>
#include <string>
#include <utility>
#include <vector>

class Timer {
public:
  explicit Timer(std::string name) : name_(std::move(name)) {}

  void start();
  void stop();
  [[nodiscard]] double duration(const std::string &unit = "ms") const;
  [[nodiscard]] std::string name() const { return name_; }
  void print_duration(const std::string &unit = "ms") const;
  void print_all(const std::string &unit = "ms") const;
  bool is_stopped() const { return is_started_ && is_stopped_; }

private:
  const std::string name_;
  std::chrono::time_point<std::chrono::steady_clock> start_time_{};
  std::chrono::time_point<std::chrono::steady_clock> end_time_{};
  std::chrono::duration<double> duration_{};
  bool is_stopped_ = false;
  bool is_started_ = false;
};

/**
 * @brief Class to manage multiple timers.
 */
class Timers {
public:
  /**
   * @brief Vector of timers.
   */
  std::vector<std::unique_ptr<Timer>> timers_{};

  /**
   * @brief Start a new timer.
   * @param name Name of the timer to add.
   */
  [[nodiscard]] Timer *add(const std::string &name);

  /**
   * @brief Print all timer durations.
   * @param unit Unit to print the durations in. Default is milliseconds (ms).
   * Options are "s", "ms"
   *
   * Generally used in a program's final output to summarize timing information.
   */
  void print(const std::string &unit = "ms");
  void stop_all();
};

#endif // OMEGA_H_READ_TIMER_H