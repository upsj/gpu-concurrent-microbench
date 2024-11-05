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

__device__ void do_work(int &data, long work) {
  for (int i = 0; i < work; i++) {
    const auto answer1 = data / 2;
    const auto answer2 = 3 * data + 1;
    const bool is_even = data % 2 == 0;

    // do it like that to have uniform (timing) paths
    data = is_even * answer1 + !is_even * answer2;
  }
}

template <typename T> __device__ T imin(T a, T b) { return a < b ? a : b; }

// propagates signals down a tree with given degree and total node count size
//
// Call with at least size linear threads
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
  for (auto cur_child = child_begin; cur_child < child_end; cur_child++) {
    std::int32_t child_val{};
    while ((child_val = load_relaxed(flags + cur_child)) < 0) {
      __nanosleep(nanosleep_time);
    }
    do_work(child_val, work);
    sum += child_val;
  }
  // make sure we don't write "non-ready" values
  if (sum < 0) {
    sum = 1;
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
        // make sure we don't write "non-ready" values
        if (sum < 0) {
          sum = 1;
        }
        store_relaxed(flags + node, sum);
      }
    }
  }
#endif
}

// propagates signals along a 2D grid, depdnency directions going right and down
//
// Call with at least length * length linear threads
__global__ void signal_propagation_2dgrid_kernel(std::int32_t *flags,
                                                 int length, int work,
                                                 int nanosleep_time) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= length * length) {
    return;
  }
  const auto node = i;

  const auto has_left_dep = (i % length) != 0;
  const auto has_top_dep = i >= length;

  std::int32_t sum{1};

  if (has_left_dep) {
    std::int32_t left_dep_val{};
    while ((left_dep_val = load_relaxed(flags + i - 1)) < 0) {
      __nanosleep(nanosleep_time);
    }
    do_work(left_dep_val, work);
    sum += left_dep_val;
  }

  if (has_top_dep) {
    std::int32_t top_dep_val{};
    while ((top_dep_val = load_relaxed(flags + i - length)) < 0) {
      __nanosleep(nanosleep_time);
    }
    do_work(top_dep_val, work);
    sum += top_dep_val;
  }

  if (sum < 0) {
    sum = 1;
  }
  store_relaxed(flags + node, sum);
}

// propagate parallel lines of dependencies,
// with threads line_id + k * #lines responsible for
// the line "line_id"
//
// Call with at least num_lines * line_length linear threads
__global__ void
signal_propagation_lines_acrosswarps_kernel(std::int32_t *flags, int num_lines,
                                            int line_length, int work,
                                            int nanosleep_time) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i >= num_lines * line_length) {
    return;
  }

  const auto line = i % num_lines;

  std::int32_t sum{1};

  std::int32_t val{0};
  if (i >= num_lines) { // first threads are independent
    while ((val = load_relaxed(flags + i - num_lines)) < 0) {
      __nanosleep(nanosleep_time);
    }
    do_work(val, work);
  }

  sum += val;

  if (sum < 0) {
    sum = 1;
  }
  store_relaxed(flags + i, sum);
}

// propagate parallel lines of dependencies,
// with threads line_id * line_length + k responsible for
// the line "line_id"
//
// Call with at least num_lines * line_length linear threads
__global__ void
signal_propagation_lines_withinwarps_kernel(std::int32_t *flags, int num_lines,
                                            int line_length, int work,
                                            int nanosleep_time) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i >= num_lines * line_length) {
    return;
  }

  const auto line = i / line_length;

  std::int32_t sum{1};

  std::int32_t val{0};
  if (i % line_length != 0) { // first row for line is immediately ready
    while ((val = load_relaxed(flags + i - 1)) < 0) {
      __nanosleep(nanosleep_time);
    }
    do_work(val, work);
  }

  sum += val;

  if (sum < 0) {
    sum = 1;
  }
  store_relaxed(flags + i, sum);
}

// Only one thrad per warp is active, this thread computes an entire line by
// himself, thereby eleiminating the need for global/shared memory intermediate
// reads
//
// Call with at least warp_size * num_lines linear threads
template <bool should_store>
__global__ void
signal_propagation_lines_warpbased_kernel(std::int32_t *flags, int num_lines,
                                          int line_length, int work,
                                          int nanosleep_time) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  const auto tid = i % 32;
  const auto warp = i / 32;

  if (i >= num_lines) {
    return;
  }

  const auto line = i;

  if (tid > 0) {
    return;
  }

  std::int32_t sum{1};

  for (int idx = 0; idx < line_length; ++idx) {
    auto val = sum;
    do_work(val, work);
    sum += val;

    if (should_store) {
      store_relaxed(flags + line * line_length + idx, sum);
    }

    __nanosleep(nanosleep_time);
  }
}

// a basic sptrsv kernel, no manual shared memory
//
// call with at least n linear threads
__global__ void basic_sptrsv_kernel(std::int32_t *flags, std::int32_t *rowptrs,
                                    std::int32_t *colidxs, int n,
                                    int element_work, int diag_work,
                                    int nanosleep_time) {
  int gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid >= n) {
    return;
  }
  const auto node = gid;
  const auto dep_begin = rowptrs[node];
  const auto dep_end = rowptrs[node + 1];
  std::int32_t sum{1};

  for (auto i = dep_begin; i < dep_end; i++) {
    std::int32_t val{};
    const auto dep = colidxs[i];

    while ((val = load_relaxed(flags + dep)) < 0) {
      __nanosleep(nanosleep_time);
    }
    do_work(val, element_work);

    sum += val;
  }

  do_work(sum, diag_work);

  // make sure we don't write "non-ready" values
  if (sum < 0) {
    sum = 1;
  }
  store_relaxed(flags + node, sum);
}

// an sptrsv kernel which assigns thread_width threads to each row
//
// call with at least thread_width * n linear threads
__global__ void basic_sptrsv_wide_kernel(std::int32_t *flags,
                                         std::int32_t *rowptrs,
                                         std::int32_t *colidxs, int n,
                                         int element_work, int diag_work,
                                         int nanosleep_time, int thread_width) {
  const auto gid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto thread = threadIdx.x;

  if (gid >= n * thread_width) {
    return;
  }

  const auto vwarp = gid / thread_width;
  const auto vwarp_start = vwarp * thread_width;
  const auto vwarp_end = vwarp_start + thread_width;
  const auto row = vwarp;
  const int vthread = gid - vwarp_start;

  if (row >= n) {
    return;
  }

  const auto row_begin = rowptrs[row];
  const auto row_end = rowptrs[row + 1] - 1;
  const auto row_length = row_end - row_begin;

  auto sum = 0;
  int i = row_begin + vthread;
  for (;; i += thread_width) {

    if (i - row_begin >= row_length) {
      break;
    }

    const auto dep = colidxs[i];

    if (dep >= row) {
      break;
    }

    std::int32_t l{};

    while ((l = load_relaxed(flags + dep)) < 0) {
      __nanosleep(nanosleep_time);
    }

    do_work(l, element_work);

    sum += l;
  }

  do_work(sum, diag_work);

  uint32_t syncmask = ((1 << thread_width) - 1) << (vwarp_start & 31);

  int total = sum;
  for (int offset = 1; offset < thread_width; ++offset) {
    auto a = sum;
    const auto received_a = __shfl_down_sync(syncmask, a, offset);
    const auto should_add = (syncmask >> ((thread & 31) + offset)) & 1 == 1;
    total += should_add * received_a;
  }

  if (total < 0) {
    total = 1;
  }

  if (vthread == 0) {
    store_relaxed(flags + row, total);
  }
}