#include "memory.hpp"
#include <nvbench/nvbench.hpp>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#ifdef USE_HIP
#include <thrust/system/hip/detail/execution_policy.h>
#else
#include <cuda.h>
#include <thrust/system/cuda/detail/execution_policy.h>
#endif

__device__ void do_work(int &data, int work) {
  int tmp;
  for (int i = 0; i < work; i++) {
#ifdef USE_HIP
    volatile int tmp = work + 1;
    work = tmp - 1;
#else
    asm volatile("add.s32 %0, %1, 1;" : "=r"(tmp) : "r"(data));
    asm volatile("sub.s32 %0, %1, 1;" : "=r"(data) : "r"(tmp));
#endif
  }
}

template <typename T> __device__ T imin(T a, T b) { return a < b ? a : b; }

__global__ void signal_propagation_tree_kernel(std::int32_t *flags, int degree,
                                               int size, int work,
                                               int nanosleep_time) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= size) {
    return;
  }
  const auto node = size - 1 - i;
  const auto child_begin = imin(node * degree + 1, size);
  const auto child_end = imin(child_begin + degree, size);
  auto cur_child = child_begin;
  std::int32_t sum{1};
#if !defined(__HIP_DEVICE_COMPILE__) && defined(__CUDA_ARCH__) &&              \
    (__CUDA_ARCH__ >= 700)
  for (auto cur_child = child_begin; cur_child < child_end; child++) {
    std::int32_t child_val{};
    while ((child_val = load_relaxed(flags + cur_child)) < 0) {
      __nanosleep(nanosleep_time);
    }
    do_work<work>(child_val);
    sum += child_val;
  }
  store_relaxed(flags + node, sum);
#else
  if (cur_child == child_end) {
    store_relaxed(flags + node, sum);
  }
  while (cur_child < child_end) {
    std::int32_t child_val{};
    if ((child_val = load_relaxed(flags + cur_child)) >= 0) {
      cur_child++;
      do_work(child_val, work);
      sum += child_val;
      if (cur_child == child_end) {
        store_relaxed(flags + node, sum);
      }
    }
  }
#endif
}

void signal_propagation_tree(nvbench::state &state) {
  // Allocate input data:
  const auto size = static_cast<std::size_t>(state.get_int64("size"));
  const auto degree = state.get_int64("degree");
  const auto work = state.get_int64("work");
  const auto sleep = state.get_int64("sleep");
#ifdef USE_HIP
  if (sleep > 0) {
    state.skip("HIP has no sleep support");
    return;
  }
#else
  int device_id;
  cudaGetDevice(&device_id);
  cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, device_id);
  if (sleep > 0 && device_prop.major < 7) {
    state.skip("Pascal or older has no sleep support.");
    return;
  }
#endif

  const auto blocksize = state.get_int64("blocksize");
  thrust::device_vector<std::int32_t> signal_flags(size, -1);
  const auto num_blocks = (size + blocksize - 1) / blocksize;

  state.exec(
      nvbench::exec_tag::timer, [&](nvbench::launch &launch, auto &timer) {
#ifdef USE_HIP
        auto policy = thrust::hip::par.on(launch.get_stream());
#else
        auto policy = thrust::cuda::par.on(launch.get_stream());
#endif

        thrust::fill(policy, signal_flags.begin(), signal_flags.end(), -1);
        timer.start();
        signal_propagation_tree_kernel<<<num_blocks, blocksize, 0,
                                         launch.get_stream()>>>(
            thrust::raw_pointer_cast(signal_flags.data()), degree, size, work,
            sleep);
        timer.stop();
      });
}
NVBENCH_BENCH(signal_propagation_tree)
    .add_int64_axis("size", {100, 1000, 10000, 100000})
    .add_int64_axis("work", {0, 16, 256})
    .add_int64_axis("degree", {1, 2, 3, 4, 16, 64})
    .add_int64_axis("blocksize", {64, 512})
    .add_int64_axis("sleep", {0, 10, 1000});