// SPDX-License-Identifier: Unlicense
// Reproducible contract stress for the real GPU backend. CPU expectations are
// computed independently here rather than calling backend reference helpers.
#include "gpu_backend.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
constexpr unsigned kLanes = 4, kSlots = 6;
constexpr std::size_t kMaximumElements = 4097;
constexpr std::uint32_t kRandomSeed = 0x16605eed;
struct Counters {
  std::uint64_t batches = 0, checked_jobs = 0, allocated = 0, released = 0;
  std::uint64_t finalized = 0, cancelled = 0, pending_allocations = 0;
  std::uint64_t rejected_operations = 0, sample_values = 0, actual_iterations = 0;
  std::uint64_t simultaneous_four_lane_epochs = 0;
  std::uint64_t omitted_summaries = 0, reported_summaries = 0;
  std::array<std::uint64_t, kLanes> observed_masks{};
  std::array<bool, 97> last_residue_seen{};
} counters;
void require(bool value, const std::string& explanation) {
  if (!value) throw std::runtime_error(explanation);
}
template<class Exception = std::exception, class Fn>
void reject(Fn action, const char* explanation) {
  bool threw = false;
  try { action(); } catch (const Exception&) { threw = true; }
  require(threw, explanation);
  ++counters.rejected_operations;
}
struct Job {
  std::size_t elements = 0;
  std::uint32_t seed = 0;
  std::uint64_t iterations = 0, sm_mask = 0;
  bool initialized = false, finalized = false;
  std::array<float, 8> expected_samples{};
  float expected_last = 0;
  explicit Job(std::size_t n, std::uint32_t s) : elements(n), seed(s) {
    const auto seed_term = static_cast<float>(seed % 97) * 0.001f;
    for (unsigned i = 0; i < expected_samples.size(); ++i)
      expected_samples[i] = seed_term + static_cast<float>(i % 97) * 0.001f;
    expected_last = seed_term + static_cast<float>((n - 1) % 97) * 0.001f;
    counters.last_residue_seen[(n - 1) % 97] = true;
  }
  void advance(const gpu::BatchItem& item) {
    require(item.initialize == !initialized, "stress oracle initialization mismatch");
    const std::uint64_t steps = static_cast<std::uint64_t>(item.iterations) * item.rounds;
    const float addend = 0.0001f * static_cast<float>(seed % 97);
    for (std::uint64_t k = 0; k < steps; ++k) {
      for (unsigned i = 0; i < std::min<std::size_t>(8, elements); ++i)
        expected_samples[i] = std::fma(expected_samples[i], 0.9999f, addend);
      expected_last = std::fma(expected_last, 0.9999f, addend);
    }
    iterations += steps;
    counters.actual_iterations += steps;
    initialized = true;
    finalized = item.finalize;
  }
  void verify(const gpu::BatchResult& actual, unsigned lane, int slot, bool reported) {
    const unsigned samples = static_cast<unsigned>(std::min<std::size_t>(8, elements));
    require(actual.slot == slot, "result identity mismatch");
    require(actual.finalized == finalized, "finalization disagrees with submitted descriptor");
    require(actual.summary_available == reported, "summary availability mismatch");
    ++counters.checked_jobs;
    if (!reported) {
      ++counters.omitted_summaries;
      require(actual.first == 0 && actual.last == 0 && actual.sample_checksum == 0 &&
              actual.sample_count == 0 && actual.sm_mask == 0 &&
              std::all_of(actual.sample_values.begin(), actual.sample_values.end(), [](float x){return x == 0;}),
              "omitted summary exposed stale result data");
      return;
    }
    ++counters.reported_summaries;
    require(actual.sample_count == samples, "sample count mismatch");
    require(actual.first == expected_samples[0] && actual.last == expected_last,
            "first/last numerical mismatch, lane=" + std::to_string(lane) + " slot=" + std::to_string(slot));
    double checksum = 0;
    for (unsigned i = 0; i < samples; ++i) {
      require(actual.sample_values[i] == expected_samples[i], "sample numerical mismatch");
      checksum += static_cast<double>(expected_samples[i]);
    }
    for (unsigned i = samples; i < 8; ++i)
      require(actual.sample_values[i] == 0, "unused summary samples contain stale data");
    require(actual.sample_checksum == checksum, "sample checksum numerical mismatch");
    require(actual.sm_mask != 0, "kernel did not report an actual SM");
    require((actual.sm_mask & sm_mask) == sm_mask, "job's accumulated SM mask lost previous observations");
    require(__builtin_popcountll(actual.sm_mask) <= 4, "job escaped its four-SM lane");
    sm_mask = actual.sm_mask;
    counters.observed_masks[lane] |= actual.sm_mask;
    counters.sample_values += samples;
  }
};
using Jobs = std::map<int, Job>;
std::size_t shape(std::mt19937& random) {
  // Also cover every residue of the published periodic input, through last().
  if (counters.allocated < 97) return static_cast<std::size_t>(counters.allocated + 1);
  constexpr std::size_t boundaries[] = {1,2,7,8,9,31,63,127,128,255,256,257,263,511,512,513,
                                      1023,1024,1025,2047,2048,4095,4096,4097};
  if (random() % 4 == 0) return 1 + random() % kMaximumElements;
  return boundaries[random() % (sizeof(boundaries) / sizeof(boundaries[0]))];
}
std::uint32_t seed(std::mt19937& random) {
  constexpr std::uint32_t edges[] = {0,1,96,97,98,194,std::numeric_limits<std::uint32_t>::max()};
  return random() % 3 == 0 ? edges[random() % (sizeof(edges) / sizeof(edges[0]))] : random();
}
int allocate(gpu::Backend& backend, unsigned lane, Jobs& jobs, std::mt19937& random) {
  const auto n = shape(random); const auto s = seed(random);
  int slot = backend.allocate(lane, n, s);
  require(jobs.emplace(slot, Job(n, s)).second, "backend allocated a live slot twice");
  ++counters.allocated;
  return slot;
}
void release(gpu::Backend& backend, unsigned lane, Jobs& jobs, int slot) {
  auto it = jobs.find(slot); require(it != jobs.end(), "stress oracle released missing slot");
  const bool final = it->second.finalized;
  backend.release(lane, slot);
  jobs.erase(it); ++counters.released;
  if (final) ++counters.finalized; else ++counters.cancelled;
  reject<std::logic_error>([&] { backend.release(lane, slot); }, "double release accepted");
}
void check_disjoint() {
  for (unsigned a = 0; a < kLanes; ++a)
    for (unsigned b = a + 1; b < kLanes; ++b)
      require((counters.observed_masks[a] & counters.observed_masks[b]) == 0, "observed SM lanes overlap");
}
void check_primary_context(CUcontext primary) {
  CUcontext current{};
  require(cuCtxGetCurrent(&current) == CUDA_SUCCESS && current == primary,
          "backend leaked a green context onto the host context stack");
}
void invalid_configurations(const gpu::BackendConfig& valid) {
  for (unsigned cycle = 0; cycle < 8; ++cycle) {
    auto config = valid;
    config.lanes = 0;
    reject<std::invalid_argument>([&] { gpu::Backend rejected(config); }, "zero lanes accepted");
    config = valid; config.slots_per_lane = 0;
    reject<std::invalid_argument>([&] { gpu::Backend rejected(config); }, "zero slots accepted");
    config = valid; config.max_elements_per_slot = 0;
    reject<std::invalid_argument>([&] { gpu::Backend rejected(config); }, "zero slot capacity accepted");
    config = valid; config.sm_per_lane = 3;
    reject<std::invalid_argument>([&] { gpu::Backend rejected(config); }, "misaligned SM partition accepted");
    config = valid; config.sm_per_lane = 1;
    reject<std::invalid_argument>([&] { gpu::Backend rejected(config); }, "sub-minimum SM partition accepted");
    config = valid; config.lanes = 4; config.sm_per_lane = 6;
    reject<std::invalid_argument>([&] { gpu::Backend rejected(config); }, "SM overcommit accepted");
    config = valid; config.lanes = 4; config.slots_per_lane = 128; config.max_elements_per_slot = 4194304;
    reject<gpu::CapacityError>([&] { gpu::Backend rejected(config); }, "oversized pool accepted");
    config = valid; config.max_elements_per_slot = 4194305;
    reject<std::invalid_argument>([&] { gpu::Backend rejected(config); }, "oversized element bound accepted");
  }
}
void invalid_batch_atomicity(gpu::Backend& backend, unsigned lane, const std::vector<gpu::BatchItem>& valid) {
  require(!valid.empty(), "empty stress batch");
  reject<std::invalid_argument>([&] { backend.submit(lane, {}); }, "empty batch accepted");
  auto duplicate = valid; duplicate.push_back(valid.front());
  reject<std::invalid_argument>([&] { backend.submit(lane, duplicate); }, "duplicate slot batch accepted");
  auto bad_slot = valid; bad_slot.push_back({999,1,true,false,1});
  reject<std::out_of_range>([&] { backend.submit(lane, bad_slot); }, "invalid second slot accepted");
  auto negative = valid; negative.back().slot = -1;
  reject<std::out_of_range>([&] { backend.submit(lane, negative); }, "negative slot accepted");
  auto zero = valid; zero.back().iterations = 0;
  reject<std::invalid_argument>([&] { backend.submit(lane, zero); }, "zero iteration batch accepted");
  zero = valid; zero.back().rounds = 0;
  reject<std::invalid_argument>([&] { backend.submit(lane, zero); }, "zero round batch accepted");
  auto overflow = valid; overflow.back().iterations = std::numeric_limits<std::uint32_t>::max();
  overflow.back().rounds = std::numeric_limits<std::uint32_t>::max();
  reject<std::invalid_argument>([&] { backend.submit(lane, overflow); }, "iteration product overflow accepted");
  auto initialized = valid; initialized.front().initialize = !initialized.front().initialize;
  reject<std::logic_error>([&] { backend.submit(lane, initialized); }, "invalid initialization state accepted");
  auto unreported_final = valid; unreported_final.front().finalize = true;
  unreported_final.front().report_summary = false;
  reject<std::invalid_argument>([&] { backend.submit(lane, unreported_final); }, "unreported final slice accepted");
  require(!backend.busy(lane), "rejected batch changed pending ownership");
  require(!backend.poll(lane).has_value(), "rejected batch produced a completion");
}
} // namespace

int main(int argc, char** argv) {
  try {
    unsigned epochs = 350;
    if (argc == 3 && std::string(argv[1]) == "--epochs") {
      auto parsed = std::stoul(argv[2]);
      require(parsed >= 250 && parsed <= 5000, "epochs must be 250..5000");
      epochs = static_cast<unsigned>(parsed);
    } else require(argc == 1, "usage: backend_stress [--epochs 250..5000]");
    gpu::BackendConfig config;
    config.slots_per_lane = kSlots; config.max_elements_per_slot = kMaximumElements;
    invalid_configurations(config);
    std::mt19937 random(kRandomSeed);
    {
      gpu::Backend backend(config);
      require(backend.info().mode == "green-contexts" && backend.info().sm_count == 22,
              "stress requires the actual GTX 1660 SUPER with 22 SMs");
      CUcontext primary{}; require(cuCtxGetCurrent(&primary) == CUDA_SUCCESS && primary, "missing primary context");
      std::array<Jobs, kLanes> jobs;
      reject<std::out_of_range>([&] { backend.allocate(kLanes,1,0); }, "invalid allocation lane accepted");
      reject<std::out_of_range>([&] { backend.poll(kLanes); }, "invalid polling lane accepted");
      reject<std::out_of_range>([&] { backend.busy(kLanes); }, "invalid busy lane accepted");
      reject<std::invalid_argument>([&] { backend.allocate(0,0,0); }, "zero elements accepted");
      reject<std::invalid_argument>([&] { backend.allocate(0,kMaximumElements+1,0); }, "oversized elements accepted");
      for (unsigned lane = 0; lane < kLanes; ++lane) {
        for (unsigned n = 0; n < kSlots; ++n) allocate(backend,lane,jobs[lane],random);
        reject<gpu::CapacityError>([&] { backend.allocate(lane,1,0); }, "slot capacity bypassed");
        std::vector<int> allocated; for (const auto& [slot,_] : jobs[lane]) allocated.push_back(slot);
        for (int slot : allocated) release(backend,lane,jobs[lane],slot);
      }
      for (unsigned epoch = 0; epoch < epochs; ++epoch) {
        std::array<std::vector<gpu::BatchItem>, kLanes> batches;
        for (unsigned lane = 0; lane < kLanes; ++lane) {
          auto& active = jobs[lane];
          std::vector<int> departing;
          for (const auto& [slot,job] : active) {
            if (job.finalized) {
              const int tested_slot = slot;
              reject<std::logic_error>([&] { backend.submit(lane,{{tested_slot,1,false,false,1}}); }, "finalized slot reused before release");
              departing.push_back(slot);
            } else if (random() % 5 == 0) departing.push_back(slot);
          }
          for (int slot : departing) release(backend,lane,active,slot);
          if (active.size() == kSlots) release(backend,lane,active,active.begin()->first);
          const auto target = static_cast<std::size_t>(1 + random() % (kSlots - 1));
          while (active.size() < target) allocate(backend,lane,active,random);
          std::vector<int> order; for (const auto& [slot,_] : active) order.push_back(slot);
          std::shuffle(order.begin(),order.end(),random);
          const auto count = static_cast<std::size_t>(1 + random() % order.size());
          for (std::size_t i = 0; i < count; ++i) {
            auto& job = active.at(order[i]);
            constexpr unsigned rounds[] = {1,2,3,7,16};
            std::uint32_t iterations = 1 + random() % 13, round = rounds[random() % 5];
            if (epoch % 127 == 0 && i == 0) { iterations = 16; round = 256; }
            const bool final = random() % 4 == 0;
            const bool summary = final || random() % 3 == 0;
            batches[lane].push_back({order[i], iterations, !job.initialized, final, round, summary});
          }
          if (epoch % 7 == 0) invalid_batch_atomicity(backend,lane,batches[lane]);
          backend.submit(lane,batches[lane]); ++counters.batches;
          require(backend.busy(lane), "submitted lane not marked pending");
          reject<std::logic_error>([&] { backend.submit(lane,batches[lane]); }, "second pending batch accepted");
          reject<std::logic_error>([&] { backend.release(lane,batches[lane].front().slot); }, "pending participating slot released");
          int joined = allocate(backend,lane,active,random); ++counters.pending_allocations;
          reject<std::logic_error>([&] { backend.release(lane,joined); }, "new nonparticipating slot released during pending lane");
          require(!active.at(joined).initialized, "pending allocation initialized itself");
          check_primary_context(primary);
        }
        for (unsigned lane = 0; lane < kLanes; ++lane)
          require(backend.busy(lane), "four lanes were not simultaneously pending before retirement");
        ++counters.simultaneous_four_lane_epochs;
        std::array<bool,kLanes> completed{};
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        unsigned left = kLanes;
        while (left) {
          require(std::chrono::steady_clock::now() < deadline, "GPU stress batch timed out");
          for (unsigned lane = 0; lane < kLanes; ++lane) {
            if (completed[lane]) continue;
            auto result = backend.poll(lane);
            check_primary_context(primary);
            if (!result) continue;
            require(result->results.size() == batches[lane].size(), "continuous batch lost results");
            require(std::isfinite(result->gpu_ms) && result->gpu_ms >= 0, "invalid CUDA event duration");
            for (std::size_t i = 0; i < batches[lane].size(); ++i) {
              auto& expected = jobs[lane].at(batches[lane][i].slot);
              expected.advance(batches[lane][i]);
              expected.verify(result->results[i],lane,batches[lane][i].slot,batches[lane][i].report_summary);
            }
            completed[lane] = true; --left;
            require(!backend.busy(lane) && !backend.poll(lane), "completion ownership retired twice");
          }
          if (left) std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
        check_disjoint();
      }
      for (unsigned lane = 0; lane < kLanes; ++lane) {
        std::vector<int> remaining; for (const auto& [slot,_] : jobs[lane]) remaining.push_back(slot);
        for (int slot : remaining) release(backend,lane,jobs[lane],slot);
        require(jobs[lane].empty() && !backend.busy(lane), "GPU slots remain after explicit cleanup");
        check_primary_context(primary);
      }
    }
    // Exercise destructor synchronization and resource cleanup without poll().
    for (unsigned cycle = 0; cycle < 8; ++cycle) {
      gpu::Backend backend(config);
      for (unsigned lane = 0; lane < kLanes; ++lane) {
        std::vector<gpu::BatchItem> batch;
        for (unsigned n = 0; n < 3; ++n) {
          int slot = backend.allocate(lane,513,cycle*97+n);
          batch.push_back({slot,19,true,false,3});
        }
        backend.submit(lane,batch);
      }
    }
    require(counters.batches >= 1000 && counters.allocated == counters.released,
            "stress did not reach required batch count or leaked an oracle-owned reservation");
    require(std::all_of(counters.last_residue_seen.begin(),counters.last_residue_seen.end(),[](bool x){return x;}),
            "periodic input residue coverage incomplete");
    std::printf("{\"seed\":%u,\"epochs\":%u,\"batches\":%llu,\"checked_jobs\":%llu,\"allocated\":%llu,\"released\":%llu,\"finalized\":%llu,\"cancelled\":%llu,\"pending_allocations\":%llu,\"rejected_operations\":%llu,\"checked_sample_values\":%llu,\"actual_affine_iterations\":%llu,\"four_lane_pending_epochs\":%llu,\"omitted_summaries\":%llu,\"reported_summaries\":%llu,\"observed_masks\":[\"0x%llx\",\"0x%llx\",\"0x%llx\",\"0x%llx\"]}\n",
                kRandomSeed,epochs,(unsigned long long)counters.batches,(unsigned long long)counters.checked_jobs,
                (unsigned long long)counters.allocated,(unsigned long long)counters.released,
                (unsigned long long)counters.finalized,(unsigned long long)counters.cancelled,
                (unsigned long long)counters.pending_allocations,(unsigned long long)counters.rejected_operations,
                (unsigned long long)counters.sample_values,(unsigned long long)counters.actual_iterations,
                (unsigned long long)counters.simultaneous_four_lane_epochs,
                (unsigned long long)counters.omitted_summaries,(unsigned long long)counters.reported_summaries,
                (unsigned long long)counters.observed_masks[0],(unsigned long long)counters.observed_masks[1],
                (unsigned long long)counters.observed_masks[2],(unsigned long long)counters.observed_masks[3]);
    std::puts("PASS: real GPU contract stress, independent numeric summaries, 97 input residues, disjoint SMs, simultaneous lanes, invalid-input atomicity, bounded pools, pending join/leave guards, constructor and destructor cleanup");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr,"backend stress failure: %s\n",error.what()); return 1;
  }
}
