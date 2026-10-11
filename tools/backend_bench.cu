// SPDX-License-Identifier: Unlicense
// Real-GPU microbenchmarks. No sleeps or client/network work are timed.
#include "gpu_backend.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
template<class T, class = void> struct HasReport : std::false_type {};
template<class T> struct HasReport<T, std::void_t<decltype(std::declval<T>().report_summary)>>
    : std::true_type {};
template<class T> auto report(T& item, bool enabled, int)
    -> decltype(item.report_summary = enabled, void()) { item.report_summary = enabled; }
template<class T> void report(T&, bool, long) {}
double micros(Clock::duration duration) {
  return std::chrono::duration<double, std::micro>(duration).count();
}
double percentile(std::vector<double> values, double fraction) {
  std::sort(values.begin(), values.end());
  return values.at(static_cast<std::size_t>(fraction * static_cast<double>(values.size() - 1)));
}
double mean(const std::vector<double>& values) {
  double total = 0; for (double value : values) total += value;
  return total / static_cast<double>(values.size());
}
struct Pending {
  std::vector<std::vector<int>> slots;
  std::vector<gpu::BatchCompletion> completed;
};
Pending allocate(gpu::Backend& backend, const std::vector<std::size_t>& shapes) {
  Pending state; state.slots.resize(backend.config().lanes);
  state.completed.resize(backend.config().lanes);
  for (unsigned lane = 0; lane < backend.config().lanes; ++lane)
    for (unsigned job = 0; job < shapes.size(); ++job)
      state.slots[lane].push_back(backend.allocate(lane, shapes[job], 31 + lane * 17 + job));
  return state;
}
void retire(gpu::Backend& backend, Pending& state) {
  std::vector<bool> completed(backend.config().lanes, false);
  unsigned left = backend.config().lanes;
  const auto deadline = Clock::now() + std::chrono::seconds(30);
  while (left) {
    if (Clock::now() > deadline) throw std::runtime_error("benchmark completion timeout");
    for (unsigned lane = 0; lane < backend.config().lanes; ++lane) {
      if (completed[lane]) continue;
      if (auto result = backend.poll(lane)) {
        state.completed[lane] = std::move(*result); completed[lane] = true; --left;
      }
    }
  }
}
void run(const char* name, const std::vector<std::size_t>& shapes,
         unsigned repetitions, bool fresh, bool summaries) {
  gpu::Backend backend;
  constexpr unsigned warmup = 10, iterations = 4;
  Pending state;
  if (!fresh) state = allocate(backend, shapes);
  std::vector<double> submits, walls, events, releases;
  for (unsigned turn = 0; turn < warmup + repetitions; ++turn) {
    if (fresh) state = allocate(backend, shapes);
    const auto begin = Clock::now();
    for (unsigned lane = 0; lane < backend.config().lanes; ++lane) {
      std::vector<gpu::BatchItem> items;
      for (int slot : state.slots[lane]) {
        gpu::BatchItem item{slot, iterations, fresh || turn == 0,
                            fresh || turn + 1 == warmup + repetitions, 1};
        report(item, summaries || item.finalize, 0); items.push_back(item);
      }
      const auto submitted = Clock::now();
      backend.submit(lane, items);
      if (turn >= warmup) submits.push_back(micros(Clock::now() - submitted));
    }
    retire(backend, state);
    if (turn >= warmup) {
      walls.push_back(micros(Clock::now() - begin));
      for (const auto& complete : state.completed) events.push_back(complete.gpu_ms * 1000.0);
    }
    if (fresh || turn + 1 == warmup + repetitions) {
      for (unsigned lane = 0; lane < backend.config().lanes; ++lane) {
        const auto& complete = state.completed[lane];
        if (complete.results.size() != shapes.size()) throw std::runtime_error("missing result");
        for (unsigned job = 0; job < shapes.size(); ++job) {
          const auto total_iterations = static_cast<std::uint64_t>(iterations) * (fresh ? 1 : turn + 1);
          const auto& result = complete.results[job];
          if (result.first != gpu::reference_value(31 + lane * 17 + job, 0, total_iterations) ||
              result.last != gpu::reference_value(31 + lane * 17 + job, shapes[job] - 1, total_iterations))
            throw std::runtime_error("benchmark numerical mismatch");
          const auto released = Clock::now();
          backend.release(lane, state.slots[lane][job]);
          if (turn >= warmup) releases.push_back(micros(Clock::now() - released));
        }
      }
    }
  }
  std::printf("{\"case\":\"%s\",\"repetitions\":%u,\"lanes\":%u,\"jobs_per_lane\":%zu,"
              "\"summaries_every_slice\":%s,\"pool_bytes\":%zu,\"pinned_bytes\":%zu,"
              "\"submit_us_mean\":%.3f,\"submit_us_p50\":%.3f,\"submit_us_p95\":%.3f,"
              "\"four_lane_wall_us_mean\":%.3f,\"four_lane_wall_us_p50\":%.3f,"
              "\"four_lane_wall_us_p95\":%.3f,\"batch_event_us_mean\":%.3f,"
              "\"release_cpu_us_mean\":%.3f}\n",
              name,repetitions,backend.config().lanes,shapes.size(),
              summaries || !HasReport<gpu::BatchItem>::value ? "true" : "false",
              backend.info().pool_bytes,backend.info().pinned_bytes,mean(submits),
              percentile(submits,.5),percentile(submits,.95),mean(walls),percentile(walls,.5),
              percentile(walls,.95),mean(events),mean(releases));
}
} // namespace
int main(int argc, char** argv) {
  try {
    unsigned repetitions = 150;
    if (argc == 3 && std::string(argv[1]) == "--repetitions") {
      auto parsed = std::stoul(argv[2]);
      if (parsed < 30 || parsed > 10000) throw std::invalid_argument("repetitions must be 30..10000");
      repetitions = static_cast<unsigned>(parsed);
    } else if (argc != 1) throw std::invalid_argument("usage: backend_bench [--repetitions 30..10000]");
    run("new-large", {262144}, repetitions, true, true);
    run("resident-uniform", std::vector<std::size_t>(16,65536), repetitions, false, true);
    std::vector<std::size_t> mixed(16,1); mixed.back() = 262144;
    run("resident-mixed", mixed, repetitions, false, true);
    run("resident-no-summary", std::vector<std::size_t>(16,65536), repetitions, false, false);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr,"backend benchmark failed: %s\n",error.what()); return 1;
  }
}
