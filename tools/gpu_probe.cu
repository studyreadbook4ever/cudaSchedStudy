// SPDX-License-Identifier: Unlicense
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

static void check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
static void driver_check(CUresult status, const char* operation) {
  if (status != CUDA_SUCCESS) {
    const char* message = nullptr;
    cuGetErrorString(status, &message);
    throw std::runtime_error(std::string(operation) + ": " + (message ? message : "driver error"));
  }
}
class ContextScope {
public:
  explicit ContextScope(CUcontext context) { driver_check(cuCtxPushCurrent(context), "activate green context"); }
  ~ContextScope() { CUcontext old{}; cuCtxPopCurrent(&old); }
};
struct ProbeLane {
  cudaExecutionContext_t context = nullptr;
  CUcontext driver_context = nullptr;
  cudaStream_t stream = nullptr;
  ~ProbeLane() {
    if (stream) {
      bool pushed = driver_context && cuCtxPushCurrent(driver_context) == CUDA_SUCCESS;
      cudaStreamSynchronize(stream);
      cudaStreamDestroy(stream);
      if (pushed) { CUcontext old{}; cuCtxPopCurrent(&old); }
    }
    if (context) cudaExecutionCtxDestroy(context);
  }
};
template<class T> struct DeviceBuffer {
  T* data = nullptr;
  explicit DeviceBuffer(std::size_t count) { check(cudaMalloc(&data, count * sizeof(T)), "allocate probe buffer"); }
  ~DeviceBuffer() { if (data) cudaFree(data); }
};
__global__ void sample_sm(unsigned int* seen, unsigned int* output) {
  unsigned int sm;
  asm("mov.u32 %0, %%smid;" : "=r"(sm));
  if (threadIdx.x == 0) {
    if (sm < 256) atomicExch(seen + sm, 1u);
    unsigned int x = blockIdx.x + 3u;
    for (unsigned int i = 0; i < 120000u; ++i) x = x * 1664525u + 1013904223u;
    output[blockIdx.x] = x;
  }
}
int main() {
  try {
    check(cudaSetDevice(0), "set device");
    cudaDeviceProp props{};
    check(cudaGetDeviceProperties(&props, 0), "query device");
    int driver = 0, runtime = 0;
    check(cudaDriverGetVersion(&driver), "query driver version");
    check(cudaRuntimeGetVersion(&runtime), "query runtime version");
    cudaDevResource all{}, remainder{};
    check(cudaDeviceGetDevResource(0, &all, cudaDevResourceTypeSm), "query SM resources");
    std::printf("device=%s cc=%d.%d sm_count=%u min_partition=%u alignment=%u driver=%d runtime=%d\n",
                props.name, props.major, props.minor, all.sm.smCount,
                all.sm.minSmPartitionSize, all.sm.smCoscheduledAlignment, driver, runtime);
    cudaDevResource groups[3]{};
    cudaDevSmResourceGroupParams params[3]{};
    unsigned assigned = 0;
    for (unsigned i = 0; i < 3; ++i) {
      const unsigned left = all.sm.smCount - assigned;
      params[i].smCount = i == 2 ? left : ((left / (3 - i)) / all.sm.smCoscheduledAlignment) * all.sm.smCoscheduledAlignment;
      assigned += params[i].smCount;
    }
    check(cudaDevSmResourceSplit(groups, 3, &all, &remainder, 0, params), "split SM resources");
    DeviceBuffer<unsigned> seen(3 * 256), output(3 * 1024);
    ProbeLane lanes[3];
    for (unsigned j = 0; j < 3; ++j) {
      auto& lane = lanes[j];
      cudaDevResourceDesc_t descriptor{};
      check(cudaDevResourceGenerateDesc(&descriptor, groups + j, 1), "create descriptor");
      check(cudaGreenCtxCreate(&lane.context, descriptor, 0, 0), "create green context");
      check(cudaExecutionCtxStreamCreate(&lane.stream, lane.context, cudaStreamNonBlocking, 0), "create green stream");
      CUgreenCtx green{};
      driver_check(cuStreamGetGreenCtx(lane.stream, &green), "query stream context");
      driver_check(cuCtxFromGreenCtx(&lane.driver_context, green), "convert supported driver context");
      ContextScope scope(lane.driver_context);
      check(cudaMemsetAsync(seen.data + j * 256, 0, 256 * sizeof(unsigned), lane.stream), "clear SM observations");
      sample_sm<<<1024, 128, 0, lane.stream>>>(seen.data + j * 256, output.data + j * 1024);
      check(cudaGetLastError(), "launch observation kernel");
    }
    std::vector<unsigned> observations(3 * 256);
    for (unsigned j = 0; j < 3; ++j) {
      auto& lane = lanes[j];
      ContextScope scope(lane.driver_context);
      check(cudaMemcpyAsync(observations.data() + j * 256, seen.data + j * 256, 256 * sizeof(unsigned),
                            cudaMemcpyDeviceToHost, lane.stream), "copy SM observations");
      check(cudaStreamSynchronize(lane.stream), "wait for observation kernel");
      unsigned count = 0;
      std::printf("lane=%u assigned_sms=%u observed_smid=", j, groups[j].sm.smCount);
      for (unsigned sm = 0; sm < 256; ++sm) if (observations[j * 256 + sm]) { std::printf("%u,", sm); ++count; }
      std::printf("\n");
      if (count == 0 || count > groups[j].sm.smCount) throw std::runtime_error("observed SM count violates partition");
    }
    for (unsigned sm = 0; sm < 256; ++sm)
      if (observations[sm] + observations[256 + sm] + observations[512 + sm] > 1)
        throw std::runtime_error("SM partition overlap observed");
    std::puts("PASS: three disjoint Green Context SM partitions executed actual CUDA kernels");
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "probe error: %s\n", error.what()); return 1;
  }
}
