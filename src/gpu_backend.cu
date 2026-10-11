// SPDX-License-Identifier: Unlicense
#include "gpu_backend.hpp"
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <utility>

namespace gpu {
namespace {
constexpr unsigned kThreads = 256;
constexpr std::uint32_t kMaxIterationsPerSlice = 4096;
void check(cudaError_t e, const char* operation) {
  if (e != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(e));
}
void check_driver(CUresult result, const char* operation) {
  if (result != CUDA_SUCCESS) {
    const char* message = nullptr;
    cuGetErrorString(result, &message);
    throw std::runtime_error(std::string(operation) + ": " + (message ? message : "unknown driver error"));
  }
}
// Runtime Green streams are context-aware without this scope. Explicitly making
// the corresponding supported driver context current avoids internal mismatch
// diagnostics in driver 610 / CUDA 13.3 debugger and default sanitizer reporting.
// No assumptions or casts are made about the opaque runtime execution context.
class ContextScope {
  bool pushed_ = false;
public:
  explicit ContextScope(CUcontext context) {
    if (context) {
      check_driver(cuCtxPushCurrent(context), "activate stream's green context");
      pushed_ = true;
    }
  }
  ~ContextScope() {
    if (pushed_) {
      CUcontext old{};
      if (cuCtxPopCurrent(&old) != CUDA_SUCCESS)
        std::fprintf(stderr, "GPU context stack restoration failed\n");
    }
  }
  ContextScope(const ContextScope&) = delete;
  ContextScope& operator=(const ContextScope&) = delete;
};
std::size_t checked_product(std::size_t a, std::size_t b) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b)
    throw std::invalid_argument("backend allocation size overflow");
  return a * b;
}
struct DeviceJob {
  float* values;
  unsigned long long* sm_mask;
  std::uint32_t elements, iterations, seed;
  std::uint32_t block_begin, block_end, initialize, summary_index;
};
struct DeviceResult {
  float first, last, sample_values[8];
  double sample_checksum;
  unsigned long long sm_mask;
  unsigned sample_count;
};
__device__ __forceinline__ void affine_element(const DeviceJob& job, std::uint32_t index) {
  if (index >= job.elements) return;
  if (threadIdx.x == 0) {
    unsigned sm;
    asm("mov.u32 %0, %%smid;" : "=r"(sm));
    // The supported broker target (Turing SM22) fits the 64-bit diagnostic.
    // Kernels still compute correctly on larger GPUs; masks only expose SM0..63.
    if (sm < 64) atomicOr(job.sm_mask, 1ull << sm);
  }
  // Keep the old host initialization's separate float multiply/add rounding.
  // Explicit intrinsics prevent the CUDA compiler from contracting this into FMA.
  float x = job.initialize
      ? __fadd_rn(__fmul_rn(static_cast<float>(job.seed % 97), 0.001f),
                  __fmul_rn(static_cast<float>(index % 97), 0.001f))
      : job.values[index];
  const float addend = 0.0001f * static_cast<float>(job.seed % 97);
  for (std::uint32_t k = 0; k < job.iterations; ++k)
    x = fmaf(x, 0.9999f, addend);
  job.values[index] = x;
}
__global__ void affine_batch(const DeviceJob* jobs) {
  affine_element(jobs[blockIdx.y], blockIdx.x * blockDim.x + threadIdx.x);
}
__global__ void affine_packed_batch(const DeviceJob* jobs, unsigned count) {
  // A packed grid launches only each job's useful blocks. Do the descriptor
  // lookup once per block instead of repeating it in every thread.
  __shared__ DeviceJob job;
  if (threadIdx.x == 0) {
    unsigned low = 0, high = count;
    while (low < high) {
      const unsigned middle = low + (high - low) / 2;
      if (blockIdx.x < jobs[middle].block_end) high = middle;
      else low = middle + 1;
    }
    job = jobs[low];
  }
  __syncthreads();
  affine_element(job, (blockIdx.x - job.block_begin) * blockDim.x + threadIdx.x);
}
__global__ void summarize_batch(const DeviceJob* jobs, DeviceResult* results) {
  const auto job = jobs[blockIdx.x];
  if (threadIdx.x != 0 || job.summary_index == 0xffffffffu) return;
  DeviceResult result{};
  result.first = job.values[0];
  result.last = job.values[job.elements - 1];
  result.sample_count = job.elements < 8 ? job.elements : 8;
  for (unsigned i = 0; i < result.sample_count; ++i) {
    result.sample_values[i] = job.values[i];
    result.sample_checksum += static_cast<double>(result.sample_values[i]);
  }
  result.sm_mask = *job.sm_mask;
  results[job.summary_index] = result;
}
} // namespace

struct Backend::Impl {
  struct Slot {
    bool allocated = false, initialized = false, finalized = false;
    std::size_t elements = 0;
    std::uint32_t seed = 0;
  };
  struct Lane {
    cudaExecutionContext_t context = nullptr;
    CUcontext driver_context = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr, done = nullptr;
    DeviceJob* host_jobs = nullptr;
    DeviceJob* device_jobs = nullptr;
    DeviceResult* host_results = nullptr;
    DeviceResult* device_results = nullptr;
    bool pending = false, completion_recorded = false;
    bool poisoned = false;
    std::vector<BatchItem> items;
    std::vector<Slot> slots;
  };
  BackendConfig cfg;
  DeviceInfo device_info;
  float* arena = nullptr;
  unsigned long long* masks = nullptr;
  std::vector<Lane> lanes;
  explicit Impl(const BackendConfig& c) : cfg(c) {}
  ~Impl() {
    // Cleanup is best-effort and noexcept even after an asynchronous CUDA fault.
    cudaSetDevice(cfg.device);
    for (auto& lane : lanes) {
      if (lane.stream) {
        try { ContextScope scope(lane.driver_context); cudaStreamSynchronize(lane.stream); }
        catch (...) { cudaStreamSynchronize(lane.stream); }
      }
      if (lane.start) cudaEventDestroy(lane.start);
      if (lane.done) cudaEventDestroy(lane.done);
      if (lane.device_jobs) cudaFree(lane.device_jobs);
      if (lane.device_results) cudaFree(lane.device_results);
      if (lane.host_jobs) cudaFreeHost(lane.host_jobs);
      if (lane.host_results) cudaFreeHost(lane.host_results);
      if (lane.stream) {
        try { ContextScope scope(lane.driver_context); cudaStreamDestroy(lane.stream); }
        catch (...) { cudaStreamDestroy(lane.stream); }
      }
      if (lane.context) cudaExecutionCtxDestroy(lane.context);
    }
    if (masks) cudaFree(masks);
    if (arena) cudaFree(arena);
  }
  Lane& lane_at(unsigned i) {
    if (i >= lanes.size()) throw std::out_of_range("GPU lane out of range");
    return lanes[i];
  }
  const Lane& lane_at(unsigned i) const {
    if (i >= lanes.size()) throw std::out_of_range("GPU lane out of range");
    return lanes[i];
  }
  Slot& slot_at(Lane& lane, int slot) {
    if (slot < 0 || static_cast<unsigned>(slot) >= cfg.slots_per_lane)
      throw std::out_of_range("GPU slot out of range");
    auto& state = lane.slots[slot];
    if (!state.allocated) throw std::logic_error("GPU slot is not allocated");
    return state;
  }
  float* slot_data(unsigned lane, int slot) {
    return arena + (static_cast<std::size_t>(lane) * cfg.slots_per_lane + slot) * cfg.max_elements_per_slot;
  }
  unsigned long long* slot_mask(unsigned lane, int slot) {
    return masks + static_cast<std::size_t>(lane) * cfg.slots_per_lane + slot;
  }
  void initialize() {
    if (cfg.lanes == 0 || cfg.lanes > 32 || cfg.slots_per_lane == 0 || cfg.slots_per_lane > 128 ||
        cfg.max_elements_per_slot == 0 || cfg.max_elements_per_slot > 4194304 || cfg.sm_per_lane == 0)
      throw std::invalid_argument("invalid or excessive GPU backend configuration");
    check(cudaSetDevice(cfg.device), "select CUDA device");
    cudaDeviceProp props{};
    check(cudaGetDeviceProperties(&props, cfg.device), "query CUDA device");
    device_info.name = props.name;
    device_info.sm_count = props.multiProcessorCount;
    device_info.cc_major = props.major;
    device_info.cc_minor = props.minor;
    device_info.total_memory = props.totalGlobalMem;
    device_info.lanes = cfg.lanes;
    device_info.pool_bytes = checked_product(checked_product(cfg.lanes, cfg.slots_per_lane),
                                            checked_product(cfg.max_elements_per_slot, sizeof(float)));
    const auto descriptor_bytes = checked_product(cfg.slots_per_lane, sizeof(DeviceJob));
    const auto result_bytes = checked_product(cfg.slots_per_lane, sizeof(DeviceResult));
    device_info.pinned_bytes = checked_product(cfg.lanes, descriptor_bytes + result_bytes);
    if (device_info.pinned_bytes > 1024ull * 1024 * 1024)
      throw CapacityError("pinned metadata budget exceeds 1 GiB; reduce slot dimensions");
    std::size_t free_memory = 0, total_memory = 0;
    check(cudaMemGetInfo(&free_memory, &total_memory), "query free VRAM");
    const auto reserve = std::min<std::size_t>(256ull * 1024 * 1024, free_memory / 4);
    if (device_info.pool_bytes > free_memory - reserve)
      throw CapacityError("slot pool cannot fit available VRAM with desktop reserve");
    lanes.resize(cfg.lanes);
    for (auto& lane : lanes) lane.slots.resize(cfg.slots_per_lane);
    cudaDevResource all{};
    const bool use_green = cfg.require_green;
    if (use_green)
      check(cudaDeviceGetDevResource(cfg.device, &all, cudaDevResourceTypeSm), "query SM partition support");
    std::vector<cudaDevResource> partitions(cfg.lanes);
    if (use_green) {
      if (cfg.sm_per_lane < all.sm.minSmPartitionSize ||
          cfg.sm_per_lane % all.sm.smCoscheduledAlignment != 0 ||
          static_cast<std::uint64_t>(cfg.sm_per_lane) * cfg.lanes > all.sm.smCount)
        throw std::invalid_argument("SM partition size violates device granularity or exceeds available SMs");
      unsigned groups = cfg.lanes;
      check(cudaDevSmResourceSplitByCount(partitions.data(), &groups, &all, nullptr, 0, cfg.sm_per_lane),
            "partition disjoint SM resources");
      if (groups != cfg.lanes)
        throw std::runtime_error("device returned fewer SM groups than the requested lane count");
      device_info.mode = "green-contexts";
    } else {
      device_info.mode = "streams-shared";
    }
    check(cudaMalloc(&arena, device_info.pool_bytes), "allocate bounded GPU arena");
    check(cudaMalloc(&masks, checked_product(checked_product(cfg.lanes, cfg.slots_per_lane), sizeof(*masks))),
          "allocate SM diagnostic masks");
    for (unsigned index = 0; index < cfg.lanes; ++index) {
      auto& lane = lanes[index];
      if (use_green) {
        // Request balanced workqueues as well as disjoint SMs. This reduces false
        // dependencies but CUDA does not promise independent forward progress.
        cudaDevResource resources[2]{};
        resources[0] = partitions[index];
        resources[1].type = cudaDevResourceTypeWorkqueueConfig;
        resources[1].wqConfig.device = cfg.device;
        resources[1].wqConfig.wqConcurrencyLimit = 1;
        resources[1].wqConfig.sharingScope = cudaDevWorkqueueConfigScopeGreenCtxBalanced;
        cudaDevResourceDesc_t descriptor{};
        check(cudaDevResourceGenerateDesc(&descriptor, resources, 2), "describe green-context resources");
        check(cudaGreenCtxCreate(&lane.context, descriptor, cfg.device, 0), "create green context");
        check(cudaExecutionCtxStreamCreate(&lane.stream, lane.context, cudaStreamNonBlocking, 0), "create green stream");
        CUgreenCtx green{};
        check_driver(cuStreamGetGreenCtx(lane.stream, &green), "query stream's green context");
        check_driver(cuCtxFromGreenCtx(&lane.driver_context, green), "get supported driver context handle");
        device_info.sm_counts.push_back(partitions[index].sm.smCount);
      } else {
        check(cudaStreamCreateWithFlags(&lane.stream, cudaStreamNonBlocking), "create shared stream");
        device_info.sm_counts.push_back(props.multiProcessorCount);
      }
      check(cudaEventCreate(&lane.start), "create batch start event");
      check(cudaEventCreate(&lane.done), "create batch completion event");
      check(cudaHostAlloc(&lane.host_jobs, descriptor_bytes, cudaHostAllocDefault), "allocate pinned descriptors");
      check(cudaHostAlloc(&lane.host_results, result_bytes, cudaHostAllocDefault), "allocate pinned summaries");
      check(cudaMalloc(&lane.device_jobs, descriptor_bytes), "allocate GPU descriptors");
      check(cudaMalloc(&lane.device_results, result_bytes), "allocate GPU summaries");
      ContextScope scope(lane.driver_context);
      check(cudaMemsetAsync(slot_data(index, 0), 0, device_info.pool_bytes / cfg.lanes, lane.stream), "clear lane arena");
      check(cudaMemsetAsync(slot_mask(index, 0), 0, cfg.slots_per_lane * sizeof(*masks), lane.stream), "clear lane SM masks");
      check(cudaStreamSynchronize(lane.stream), "initialize lane arena");
    }
  }
};

Backend::Backend(const BackendConfig& config) : impl_(std::make_unique<Impl>(config)) { impl_->initialize(); }
Backend::~Backend() = default;
const DeviceInfo& Backend::info() const noexcept { return impl_->device_info; }
const BackendConfig& Backend::config() const noexcept { return impl_->cfg; }
bool Backend::busy(unsigned lane) const { return impl_->lane_at(lane).pending; }
int Backend::allocate(unsigned lane_index, std::size_t elements, std::uint32_t seed) {
  auto& lane = impl_->lane_at(lane_index);
  if (lane.poisoned) throw std::runtime_error("GPU lane poisoned by a failed CUDA enqueue");
  if (elements == 0 || elements > impl_->cfg.max_elements_per_slot)
    throw std::invalid_argument("job element count exceeds slot capacity or is zero");
  for (unsigned slot = 0; slot < lane.slots.size(); ++slot) {
    if (!lane.slots[slot].allocated) {
      lane.slots[slot] = Impl::Slot{true, false, false, elements, seed};
      return static_cast<int>(slot);
    }
  }
  throw CapacityError("all slots in this GPU lane are allocated");
}
void Backend::release(unsigned lane_index, int slot) {
  auto& lane = impl_->lane_at(lane_index);
  auto& state = impl_->slot_at(lane, slot);
  if (lane.poisoned) throw std::runtime_error("GPU lane poisoned by a failed CUDA enqueue");
  if (lane.pending) throw std::logic_error("cannot release a slot while its lane has an unretired GPU batch");
  // A reservation cancelled before its first slice never touched GPU storage.
  // Constructor/previous-release wipes already precede any reuse in this stream.
  if (!state.initialized) { state = {}; return; }
  check(cudaSetDevice(impl_->cfg.device), "select CUDA device for release");
  ContextScope scope(lane.driver_context);
  const auto count = impl_->cfg.max_elements_per_slot;
  // Sanitize the complete reservation, not just the previous job's valid prefix.
  // Every later initialization/compute on this lane uses the same stream, so it
  // cannot overtake these wipes even if metadata is reassigned immediately.
  // If either enqueue fails, never expose partially sanitized storage for reuse.
  lane.poisoned = true;
  check(cudaMemsetAsync(impl_->slot_data(lane_index, slot), 0, count * sizeof(float), lane.stream), "wipe released slot");
  check(cudaMemsetAsync(impl_->slot_mask(lane_index, slot), 0, sizeof(unsigned long long), lane.stream), "wipe released diagnostic");
  lane.poisoned = false;
  state = {};
}
void Backend::submit(unsigned lane_index, const std::vector<BatchItem>& items) {
  auto& lane = impl_->lane_at(lane_index);
  if (lane.poisoned) throw std::runtime_error("GPU lane poisoned by a failed CUDA enqueue");
  if (lane.pending) throw std::logic_error("lane already has an unretired batch");
  if (items.empty() || items.size() > impl_->cfg.slots_per_lane)
    throw std::invalid_argument("batch is empty or exceeds descriptor capacity");
  std::vector<bool> seen(impl_->cfg.slots_per_lane, false);
  std::size_t maximum_elements = 0;
  // Fully validate before enqueuing anything: a rejected batch leaves slot state intact.
  for (const auto& item : items) {
    auto& slot = impl_->slot_at(lane, item.slot);
    if (slot.finalized) throw std::logic_error("finalized slot must be released before reuse");
    if (seen[item.slot]) throw std::invalid_argument("duplicate slot in batch");
    seen[item.slot] = true;
    if (item.iterations == 0 || item.rounds == 0 ||
        static_cast<std::uint64_t>(item.iterations) * item.rounds > kMaxIterationsPerSlice)
      throw std::invalid_argument("kernel slice must contain 1..4096 affine iterations");
    if (!item.initialize && !slot.initialized)
      throw std::logic_error("new slot must be initialized before compute");
    if (item.initialize && slot.initialized)
      throw std::logic_error("initialized slot cannot be reinitialized without release");
    if (item.finalize && !item.report_summary)
      throw std::invalid_argument("final slice must report a numerical summary");
    maximum_elements = std::max(maximum_elements, slot.elements);
  }
  check(cudaSetDevice(impl_->cfg.device), "select CUDA device for submit");
  ContextScope scope(lane.driver_context);
  lane.items = items;
  // Mark ownership before the first enqueue so even an enqueue failure cannot
  // let callers reuse a buffer while some preceding operations are still live.
  lane.pending = true;
  lane.poisoned = true;
  lane.completion_recorded = false;
  check(cudaEventRecord(lane.start, lane.stream), "record batch start");
  unsigned total_blocks = 0, reported_count = 0;
  for (std::size_t i = 0; i < items.size(); ++i) {
    const auto& item = items[i];
    auto& state = lane.slots[item.slot];
    auto* values = impl_->slot_data(lane_index, item.slot);
    // Constructor and every prior release already zeroed the diagnostic in this
    // stream. A never-initialized cancellation does not touch it, either.
    const auto job_blocks = static_cast<unsigned>((state.elements + kThreads - 1) / kThreads);
    lane.host_jobs[i] = DeviceJob{values, impl_->slot_mask(lane_index, item.slot),
                                 static_cast<std::uint32_t>(state.elements), item.iterations * item.rounds, state.seed,
                                 total_blocks, total_blocks + job_blocks, item.initialize ? 1u : 0u,
                                 item.report_summary ? reported_count++ : 0xffffffffu};
    total_blocks += job_blocks;
  }
  check(cudaMemcpyAsync(lane.device_jobs, lane.host_jobs, items.size() * sizeof(DeviceJob), cudaMemcpyHostToDevice, lane.stream),
        "copy continuous-batch descriptors H2D");
  const auto blocks = static_cast<unsigned>((maximum_elements + kThreads - 1) / kThreads);
  // Keep the cheaper rectangular mapping for similarly sized jobs. Switch to
  // packed mapping only when rectangular mapping adds >25% over useful blocks.
  const auto rectangular_blocks = static_cast<std::uint64_t>(blocks) * items.size();
  if (rectangular_blocks > static_cast<std::uint64_t>(total_blocks) * 5 / 4)
    affine_packed_batch<<<total_blocks, kThreads, 0, lane.stream>>>(lane.device_jobs, static_cast<unsigned>(items.size()));
  else
    affine_batch<<<dim3(blocks, static_cast<unsigned>(items.size())), kThreads, 0, lane.stream>>>(lane.device_jobs);
  check(cudaGetLastError(), "launch affine continuous batch");
  if (reported_count) {
    summarize_batch<<<static_cast<unsigned>(items.size()), 1, 0, lane.stream>>>(lane.device_jobs, lane.device_results);
    check(cudaGetLastError(), "launch batch summaries");
    check(cudaMemcpyAsync(lane.host_results, lane.device_results, reported_count * sizeof(DeviceResult), cudaMemcpyDeviceToHost, lane.stream),
          "copy batch summaries D2H");
  }
  check(cudaEventRecord(lane.done, lane.stream), "record final batch event");
  lane.completion_recorded = true;
  lane.poisoned = false;
}
std::optional<BatchCompletion> Backend::poll(unsigned lane_index) {
  auto& lane = impl_->lane_at(lane_index);
  if (lane.poisoned) throw std::runtime_error("GPU lane poisoned by a failed CUDA enqueue");
  if (!lane.pending) return std::nullopt;
  if (!lane.completion_recorded) throw std::runtime_error("GPU lane poisoned by incomplete batch submission");
  check(cudaSetDevice(impl_->cfg.device), "select CUDA device for poll");
  ContextScope scope(lane.driver_context);
  auto status = cudaEventQuery(lane.done);
  if (status == cudaErrorNotReady) return std::nullopt;
  if (status != cudaSuccess) lane.poisoned = true;
  check(status, "query asynchronous batch completion");
  BatchCompletion completion;
  check(cudaEventElapsedTime(&completion.gpu_ms, lane.start, lane.done), "measure batch GPU time");
  completion.results.reserve(lane.items.size());
  for (std::size_t i = 0; i < lane.items.size(); ++i) {
    const auto& item = lane.items[i];
    BatchResult result;
    result.slot = item.slot;
    result.summary_available = item.report_summary;
    if (item.report_summary) {
      const auto& raw = lane.host_results[lane.host_jobs[i].summary_index];
      result.first = raw.first;
      result.last = raw.last;
      result.sample_checksum = raw.sample_checksum;
      std::copy_n(raw.sample_values, 8, result.sample_values.begin());
      result.sample_count = raw.sample_count;
      result.sm_mask = raw.sm_mask;
    }
    result.finalized = item.finalize;
    completion.results.push_back(result);
    if (item.initialize) lane.slots[item.slot].initialized = true;
    if (item.finalize) lane.slots[item.slot].finalized = true;
  }
  lane.items.clear();
  lane.pending = false;
  return completion;
}
float initial_value(std::uint32_t seed, std::size_t index) noexcept {
  return static_cast<float>(seed % 97) * 0.001f + static_cast<float>(index % 97) * 0.001f;
}
float reference_value(std::uint32_t seed, std::size_t index, std::uint64_t total_iterations) {
  float value = initial_value(seed, index);
  const float addend = 0.0001f * static_cast<float>(seed % 97);
  for (std::uint64_t i = 0; i < total_iterations; ++i) value = std::fma(value, 0.9999f, addend);
  return value;
}
} // namespace gpu
