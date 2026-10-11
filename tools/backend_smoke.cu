// SPDX-License-Identifier: Unlicense
#include "gpu_backend.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <thread>
#include <vector>

static void require(bool truth, const char* message) {
  if (!truth) throw std::runtime_error(message);
}
template<class Fn> static void rejected(Fn action, const char* message) {
  bool did_throw = false;
  try { action(); } catch (const std::exception&) { did_throw = true; }
  require(did_throw, message);
}
static gpu::BatchCompletion finish(gpu::Backend& backend, unsigned lane) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(15);
  while (std::chrono::steady_clock::now() < deadline) {
    if (auto result = backend.poll(lane)) return *result;
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  throw std::runtime_error("CUDA completion timed out");
}
static void check_result(const gpu::BatchResult& result, unsigned seed, std::size_t n,
                         unsigned long long iterations) {
  require(result.sample_count == (n < 8 ? n : 8), "wrong sample count");
  double expected_sum = 0;
  for (unsigned i = 0; i < result.sample_count; ++i) {
    const float expected = gpu::reference_value(seed, i, iterations);
    if (result.sample_values[i] != expected) {
      std::fprintf(stderr,"seed=%u index=%u got=%.9g expected=%.9g\n", seed,i,result.sample_values[i],expected);
      throw std::runtime_error("CPU/GPU sample mismatch");
    }
    expected_sum += static_cast<double>(expected);
  }
  require(result.first == gpu::reference_value(seed, 0, iterations), "wrong first element");
  require(result.last == gpu::reference_value(seed, n-1, iterations), "wrong last element");
  require(result.sample_checksum == expected_sum, "wrong sample checksum");
  require(result.sm_mask != 0, "missing actual SM evidence");
}
int main() {
  try {
    gpu::BackendConfig config;
    config.slots_per_lane = 3;
    config.max_elements_per_slot = 8192;
    gpu::Backend backend(config);
    require(backend.info().mode == "green-contexts", "hardware SM isolation unavailable");
    std::printf("device=%s mode=%s gpu_pool_bytes=%zu pinned_bytes=%zu\n", backend.info().name.c_str(),
                backend.info().mode.c_str(),backend.info().pool_bytes,backend.info().pinned_bytes);
    rejected([&] { backend.allocate(4, 8, 1); }, "invalid lane accepted");
    rejected([&] { backend.allocate(0, 0, 1); }, "zero length accepted");
    rejected([&] { backend.allocate(0, 8193, 1); }, "oversized length accepted");
    std::vector<std::vector<int>> slots(4);
    const std::size_t sizes[3] = {1, 263, 8192};
    for(unsigned lane=0;lane<4;lane++) {
      for(unsigned j=0;j<3;j++) slots[lane].push_back(backend.allocate(lane,sizes[j],17+lane*10+j));
      rejected([&] { backend.allocate(lane, 3, 7); }, "slot capacity not enforced");
      rejected([&] { backend.submit(lane,{{slots[lane][0],3,false,false,1}}); }, "uninitialized compute accepted");
      rejected([&] { backend.submit(lane,{{slots[lane][0],3,true,false,1},{slots[lane][0],3,true,false,1}}); }, "duplicate slot accepted");
      rejected([&] { backend.submit(lane,{{slots[lane][0],4096,true,false,2}}); }, "unbounded slice accepted");
      std::vector<gpu::BatchItem> batch;
      for(auto slot:slots[lane]) batch.push_back({slot, 19, true, false, 3});
      backend.submit(lane,batch);
      require(backend.busy(lane),"submitted batch not owned");
      rejected([&] { backend.release(lane,slots[lane][0]); },"pending slot released");
      rejected([&] { backend.submit(lane,batch); },"overlapping batch accepted");
    }
    std::uint64_t masks[4]{};
    for(unsigned lane=0;lane<4;lane++) {
      auto complete=finish(backend,lane);
      require(complete.results.size()==3,"lost continuously batched job");
      for(unsigned j=0;j<3;j++) {
        check_result(complete.results[j],17+lane*10+j,sizes[j],57);
        masks[lane]|=complete.results[j].sm_mask;
      }
      std::printf("lane=%u assigned_sms=%u first_batch_ms=%.3f observed_sm_mask=0x%llx\n",lane,
                  backend.info().sm_counts[lane],complete.gpu_ms,(unsigned long long)masks[lane]);
      rejected([&] { backend.submit(lane,{{slots[lane][0],1,true,false,1}}); },"live slot reinitialization accepted");
      backend.release(lane,slots[lane][0]); // One user leaves while other jobs remain resident.
      int new_slot=backend.allocate(lane,97,93-lane); // A new user joins the same reservation.
      require(new_slot==slots[lane][0],"freed slot not reused");
      backend.submit(lane,{{new_slot,13,true,true,2},{slots[lane][2],11,false,true,4}});
      auto second=finish(backend,lane);
      check_result(second.results[0],93-lane,97,26);
      check_result(second.results[1],17+lane*10+2,8192,57+44);
      require(second.results[0].finalized && second.results[1].finalized,"finalization marker missing");
      rejected([&] { backend.submit(lane,{{new_slot,1,false,false,1}}); },"finalized slot computed again");
      for(auto slot:slots[lane]) backend.release(lane,slot);
      require(!backend.poll(lane).has_value(),"retired batch completed twice");
      rejected([&] { backend.release(lane,new_slot); },"double release accepted");
    }
    for(unsigned a=0;a<4;a++) for(unsigned b=a+1;b<4;b++) require((masks[a]&masks[b])==0,"actual SM sets overlap");
    // Repeated mixed-size churn exercises the same pool and event lifecycle.
    for(unsigned turn=0;turn<40;turn++) {
      unsigned lane=turn%4;
      auto n=static_cast<std::size_t>(1+(turn*113)%8192);
      auto seed=static_cast<unsigned>(turn*19);
      int slot=backend.allocate(lane,n,seed);
      backend.submit(lane,{{slot,7,true,true,5}});
      auto complete=finish(backend,lane);
      check_result(complete.results[0],seed,n,35);
      backend.release(lane,slot);
    }
    {
      gpu::BackendConfig shared = config; shared.require_green = false; shared.lanes = 2;
      gpu::Backend baseline(shared);
      require(baseline.info().mode == "streams-shared", "explicit shared-stream mode did not disable partitions");
      for (unsigned lane = 0; lane < shared.lanes; ++lane) {
        require(baseline.info().sm_counts[lane] == static_cast<unsigned>(baseline.info().sm_count), "shared stream falsely claims an SM partition");
        int slot = baseline.allocate(lane,263,44+lane);
        baseline.submit(lane,{{slot,11,true,true,2}});
        auto complete = finish(baseline,lane);
        check_result(complete.results[0],44+lane,263,22);
        baseline.release(lane,slot);
      }
    }
    std::puts("PASS: numerical reference, four disjoint SM lanes, async H2D/kernel/D2H, continuous batch join/leave, pool limits, lifecycle guards, 40 reuse cycles, explicit shared-stream baseline");
    return 0;
  } catch(const std::exception& e) {
    std::fprintf(stderr,"backend smoke failure: %s\n",e.what()); return 1;
  }
}
