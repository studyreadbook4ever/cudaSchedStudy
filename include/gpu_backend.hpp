// SPDX-License-Identifier: Unlicense
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace gpu {
struct CapacityError : std::runtime_error { using std::runtime_error::runtime_error; };
struct BackendConfig {
  int device = 0;
  unsigned lanes = 4;
  unsigned slots_per_lane = 16;
  std::size_t max_elements_per_slot = 262144;
  unsigned sm_per_lane = 4;
  bool require_green = true; // true: fail-closed SM partitioning; false: explicit shared-stream baseline.
};
struct DeviceInfo {
  std::string name;
  int sm_count = 0, cc_major = 0, cc_minor = 0;
  unsigned lanes = 0;
  std::vector<unsigned> sm_counts;
  std::string mode;
  std::size_t total_memory = 0, pool_bytes = 0, pinned_bytes = 0;
};
struct BatchItem {
  int slot = -1;
  std::uint32_t iterations = 1;
  bool initialize = false;
  bool finalize = false;
  std::uint32_t rounds = 1;
  // Intermediate slices can omit GPU summaries/D2H; final slices must report.
  bool report_summary = true;
};
struct BatchResult {
  int slot = -1;
  float first = 0, last = 0;
  double sample_checksum = 0;
  std::array<float, 8> sample_values{};
  unsigned sample_count = 0;
  std::uint64_t sm_mask = 0;
  bool finalized = false;
  // False results still carry slot/finalized metadata, but no numerical summary.
  bool summary_available = false;
};
struct BatchCompletion {
  float gpu_ms = 0;
  std::vector<BatchResult> results;
};
// One scheduler thread owns this backend. Device pointers never leave the broker.
// A lane may have one submitted batch at a time. poll() retires it only after
// the final stream event (after optional D2H); slots cannot be released or reused
// before retirement. release() queues a full wipe on that same lane's stream:
// reuse is safe by stream FIFO, without blocking the host until the wipe finishes.
// A failed CUDA enqueue poisons the lane; destroy the backend rather than reuse it.
class Backend {
public:
  explicit Backend(const BackendConfig& config = {});
  ~Backend();
  Backend(const Backend&) = delete;
  Backend& operator=(const Backend&) = delete;
  const DeviceInfo& info() const noexcept;
  const BackendConfig& config() const noexcept;
  int allocate(unsigned lane, std::size_t elements, std::uint32_t seed);
  void release(unsigned lane, int slot);
  void submit(unsigned lane, const std::vector<BatchItem>& items);
  std::optional<BatchCompletion> poll(unsigned lane);
  bool busy(unsigned lane) const;
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
// Exact float reference for the published affine workload; sample indices use
// the first min(8,elements) values, plus elements-1 for the last value.
float initial_value(std::uint32_t seed, std::size_t index) noexcept;
float reference_value(std::uint32_t seed, std::size_t index,
                      std::uint64_t total_iterations);
} // namespace gpu
