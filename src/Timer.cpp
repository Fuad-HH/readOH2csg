/**
 *@file Timer.cpp
 */

#include "Timer.h"
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

void Timer::start() {
  if (is_started_) {
    throw std::runtime_error("Timer " + name_ + " has already started.");
  }
  start_time_ = std::chrono::steady_clock::now();
  is_started_ = true;
}

void Timer::stop() {
  if (is_started_ && !is_stopped_) {
    end_time_ = std::chrono::steady_clock::now();
    is_stopped_ = true;

    duration_ = end_time_ - start_time_;
  } else {
    const std::string msg = "Timer " + name_ +
                            " was not started or has already "
                            "been stopped.";
    throw std::runtime_error(msg);
  }
}

double Timer::duration(const std::string &unit) const {
  if (is_stopped_ && is_started_) {
    if (unit == "s") {
      return std::chrono::duration<double>(duration_).count();
    }
    if (unit == "ms") {
      return std::chrono::duration<double, std::milli>(duration_).count();
    }

    throw std::invalid_argument(
        "Invalid time unit. Use 's' for seconds or 'ms' for milliseconds.");
  }

  const std::string msg = "Timer " + name_ + " was not stopped.";
  throw std::runtime_error(msg);
}

void Timer::print_duration(const std::string &unit) const {
  const std::string message =
      name_ + " :\t" + std::to_string(duration(unit)) + unit + "\n";
  std::cout << message;
}

void Timer::print_all(const std::string &unit) const {
  // print start, end and duration
  std::string message = "Timer: " + name_ + "\n";

  if (unit == "s") {
    message += "Duration: " + std::to_string(duration("s")) + " seconds\n";
    // print start and end as date time
    message += "Start Time: " +
               std::to_string(
                   std::chrono::duration<double>(start_time_.time_since_epoch())
                       .count()) +
               " seconds since epoch\n";
    message += "End Time: " +
               std::to_string(
                   std::chrono::duration<double>(end_time_.time_since_epoch())
                       .count()) +
               "seconds since epoch\n";
  } else if (unit == "ms") {
    message +=
        "Duration: " + std::to_string(duration("ms")) + " milliseconds\n";
    message += "Start Time: " +
               std::to_string(std::chrono::duration<double, std::milli>(
                                  start_time_.time_since_epoch())
                                  .count()) +
               " milliseconds since epoch\n";
    message += "End Time: " +
               std::to_string(std::chrono::duration<double, std::milli>(
                                  end_time_.time_since_epoch())
                                  .count()) +
               " milliseconds since epoch\n";
  } else {
    throw std::invalid_argument(
        "Invalid time unit. Use 's' for seconds or 'ms' for milliseconds.");
  }
  std::cout << message << std::endl;
}

Timer *Timers::add(const std::string &name) {
  timers_.push_back(std::make_unique<Timer>(name));
  timers_.back()->start();
  return timers_.back().get();
}

void Timers::stop_all() {
  for (const auto &timer : timers_) {
    if (!timer->is_stopped()) {
      printf("Warning: Timer '%s' was not stopped. Stopping it now.\n",
             timer->name().c_str());
      timer->stop();
    }
  }
}

void Timers::print(const std::string &unit) {
  // stop all timers that are still running
  stop_all();

  // Find maximum name length
  size_t max_name_length = 0;
  for (const auto &timer : timers_) {
    max_name_length = std::max(max_name_length, timer->name().length());
  }

  std::string message = "\nTiming Summary:\n";

  for (const auto &timer : timers_) {
    std::stringstream line;
    line << std::left << std::setw(max_name_length + 3)
         << (timer->name() + ": ") << std::fixed
         << std::setprecision(2) // Fixed precision: 2 decimal places
         << std::right << std::setw(12) << timer->duration(unit) << unit;

    message += line.str() + "\n";
  }

  std::cout << message << std::endl;
}