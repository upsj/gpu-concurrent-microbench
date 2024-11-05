#include "benchmark_kernels.gpu.hpp"

#include <random>
#include <set>

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

int ipow(int x, unsigned int p) {
  if (p == 0)
    return 1;
  if (p == 1)
    return x;

  int tmp = ipow(x, p / 2);
  if (p % 2 == 0)
    return tmp * tmp;
  else
    return x * tmp * tmp;
}

bool has_sleep_support() {
#ifdef USE_HIP
  return false;
#else
  int device_id;
  cudaGetDevice(&device_id);
  cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, device_id);
  return device_prop.major >= 7;
#endif
}

void signal_propagation_tree(nvbench::state &state) {

  const auto size = static_cast<std::size_t>(state.get_int64("size"));
  const auto degree = state.get_int64("degree");
  const auto work = state.get_int64("work");
  const auto sleep = state.get_int64("sleep");

  if (sleep > 0 && !has_sleep_support()) {
    state.skip("GPU has no sleep support");
    return;
  }

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

void signal_propagation_lines_32perwarp(nvbench::state &state) {

  const auto num_lines = state.get_int64("num_lines");
  const auto line_length = state.get_int64("line_length");
  const auto work = state.get_int64("work");
  const auto sleep = state.get_int64("sleep");

  if (sleep > 0 && !has_sleep_support()) {
    state.skip("GPU has no sleep support");
    return;
  }

  state.add_global_memory_reads<nvbench::int32_t>(num_lines *
                                                  (line_length - 1));
  state.add_global_memory_writes<nvbench::int32_t>(num_lines * line_length);

  const auto blocksize = state.get_int64("blocksize");
  thrust::device_vector<std::int32_t> signal_flags(num_lines * line_length, -1);
  const auto num_blocks = (num_lines * line_length + blocksize - 1) / blocksize;

  state.exec(
      nvbench::exec_tag::timer, [&](nvbench::launch &launch, auto &timer) {
#ifdef USE_HIP
        auto policy = thrust::hip::par.on(launch.get_stream());
#else
        auto policy = thrust::cuda::par.on(launch.get_stream());
#endif

        thrust::fill(policy, signal_flags.begin(), signal_flags.end(), -1);
        timer.start();
        signal_propagation_lines_acrosswarps_kernel<<<num_blocks, blocksize, 0,
                                                      launch.get_stream()>>>(
            thrust::raw_pointer_cast(signal_flags.data()), num_lines,
            line_length, work, sleep);
        timer.stop();
      });
}
NVBENCH_BENCH(signal_propagation_lines_32perwarp)
    .add_int64_axis("num_lines", {16, 128, 1024, 8 * 1024, 64 * 1024})
    .add_int64_axis("work", {0, 16, 256})
    .add_int64_axis("line_length", {1, 8, 32, 128, 512, 2048})
    .add_int64_axis("blocksize", {64, 512})
    // .add_int64_axis("sleep", {0, 10, 1000});
    .add_int64_axis("sleep", {0});

void signal_propagation_lines_1perwarp(nvbench::state &state) {

  const auto num_lines = state.get_int64("num_lines");
  const auto line_length = state.get_int64("line_length");
  const auto work = state.get_int64("work");
  const auto sleep = state.get_int64("sleep");

  if (sleep > 0 && !has_sleep_support()) {
    state.skip("GPU has no sleep support");
    return;
  }

  state.add_global_memory_reads<nvbench::int32_t>(num_lines *
                                                  (line_length - 1));
  state.add_global_memory_writes<nvbench::int32_t>(num_lines * line_length);

  const auto blocksize = state.get_int64("blocksize");
  thrust::device_vector<std::int32_t> signal_flags(num_lines * line_length, -1);
  const auto num_blocks = (num_lines * line_length + blocksize - 1) / blocksize;

  state.exec(
      nvbench::exec_tag::timer, [&](nvbench::launch &launch, auto &timer) {
#ifdef USE_HIP
        auto policy = thrust::hip::par.on(launch.get_stream());
#else
        auto policy = thrust::cuda::par.on(launch.get_stream());
#endif

        thrust::fill(policy, signal_flags.begin(), signal_flags.end(), -1);
        timer.start();
        signal_propagation_lines_withinwarps_kernel<<<num_blocks, blocksize, 0,
                                                      launch.get_stream()>>>(
            thrust::raw_pointer_cast(signal_flags.data()), num_lines,
            line_length, work, sleep);
        timer.stop();
      });
}
NVBENCH_BENCH(signal_propagation_lines_1perwarp)
    .add_int64_axis("num_lines", {16, 128, 1024, 8 * 1024, 64 * 1024})
    .add_int64_axis("work", {0, 16, 256})
    .add_int64_axis("line_length", {1, 8, 32, 128, 512, 2048})
    .add_int64_axis("blocksize", {64, 512})
    // .add_int64_axis("sleep", {0, 10, 1000});
    .add_int64_axis("sleep", {0});

void signal_propagation_lines_warpbased(nvbench::state &state) {

  const auto num_lines = state.get_int64("num_lines");
  const auto line_length = state.get_int64("line_length");
  const auto work = state.get_int64("work");
  const auto sleep = state.get_int64("sleep");

  if (sleep > 0 && !has_sleep_support()) {
    state.skip("GPU has no sleep support");
    return;
  }

  // state.add_global_memory_reads<nvbench::int32_t>(num_lines * (line_length -
  // 1));
  state.add_global_memory_writes<nvbench::int32_t>(num_lines * line_length);

  const auto blocksize = state.get_int64("blocksize");
  thrust::device_vector<std::int32_t> signal_flags(num_lines * line_length, -1);
  const auto num_blocks = (num_lines * line_length + blocksize - 1) / blocksize;

  state.exec(
      nvbench::exec_tag::timer, [&](nvbench::launch &launch, auto &timer) {
#ifdef USE_HIP
        auto policy = thrust::hip::par.on(launch.get_stream());
#else
        auto policy = thrust::cuda::par.on(launch.get_stream());
#endif

        thrust::fill(policy, signal_flags.begin(), signal_flags.end(), -1);
        timer.start();
        signal_propagation_lines_warpbased_kernel<true>
            <<<num_blocks, blocksize, 0, launch.get_stream()>>>(
                thrust::raw_pointer_cast(signal_flags.data()), num_lines,
                line_length, work, sleep);
        timer.stop();
      });
}
NVBENCH_BENCH(signal_propagation_lines_warpbased)
    .add_int64_axis("num_lines", {16, 128, 1024, 8 * 1024, 64 * 1024})
    .add_int64_axis("work", {0, 16, 256})
    .add_int64_axis("line_length", {1, 8, 32, 128, 512, 2048})
    .add_int64_axis("blocksize", {64, 512})
    // .add_int64_axis("sleep", {0, 10, 1000});
    .add_int64_axis("sleep", {0});

void signal_propagation_2dgrid(nvbench::state &state) {

  const auto length = static_cast<std::size_t>(state.get_int64("length"));
  const auto work = state.get_int64("work");
  const auto sleep = state.get_int64("sleep");

  if (sleep > 0 && !has_sleep_support()) {
    state.skip("GPU has no sleep support");
    return;
  }

  // throughput info
  state.add_global_memory_reads<nvbench::int32_t>(2 * length * length -
                                                  2 * length + 1);
  state.add_global_memory_writes<nvbench::int32_t>(length * length);

  const auto blocksize = state.get_int64("blocksize");
  thrust::device_vector<std::int32_t> signal_flags(length * length, -1);
  const auto num_blocks = (length * length + blocksize - 1) / blocksize;

  state.exec(
      nvbench::exec_tag::timer, [&](nvbench::launch &launch, auto &timer) {
#ifdef USE_HIP
        auto policy = thrust::hip::par.on(launch.get_stream());
#else
        auto policy = thrust::cuda::par.on(launch.get_stream());
#endif

        thrust::fill(policy, signal_flags.begin(), signal_flags.end(), -1);
        timer.start();
        signal_propagation_2dgrid_kernel<<<num_blocks, blocksize, 0,
                                           launch.get_stream()>>>(
            thrust::raw_pointer_cast(signal_flags.data()), length, work, sleep);
        timer.stop();
      });
}
NVBENCH_BENCH(signal_propagation_2dgrid)
    .add_int64_axis("length", {100, 300, 1000})
    .add_int64_axis("work", {0, 16, 256})
    .add_int64_axis("blocksize", {64, 512})
    .add_int64_axis("sleep", {0});

void fill_matrix_2dgrid_row(std::int32_t *rowptrs, std::int32_t *colidxs,
                            int length) {

  rowptrs[0] = 0;

  for (int i = 0; i < length * length; ++i) {
    const auto has_left_dep = (i % length) != 0;
    const auto has_top_dep = i >= length;

    auto rowptrsi = rowptrs[i];

    if (has_left_dep) {
      colidxs[rowptrsi++] = i - 1;
    }

    if (has_top_dep) {
      colidxs[rowptrsi++] = i - length;
    }

    rowptrs[i + 1] += rowptrsi;
  }
}

void fill_matrix_2dgrid_diag(std::int32_t *rowptrs, std::int32_t *colidxs,
                             int length) {

  auto perm = std::vector<int>(length * length);
  auto inv_perm = std::vector<int>(length * length);

  auto running_i = 0;
  for (int sum = 0; sum < 2 * length - 1; ++sum) {
    for (int i = 0; i <= sum; ++i) {
      const auto j = sum - i;

      if (!(0 <= j && j < length)) {
        continue; // diagonal out of grid
      }

      const auto flat_i = i * length + j;

      if (flat_i >= length * length) {
        continue; // diagonal out of grid
      }

      perm[flat_i] = running_i;
      inv_perm[running_i] = flat_i;

      running_i += 1;
    }
  }

  rowptrs[0] = 0;
  for (int i = 0; i < length * length; ++i) {

    const auto flat_i = inv_perm[i];
    const auto has_left_dep = (flat_i % length) != 0;
    const auto has_top_dep = flat_i >= length;

    auto rowptrsi = rowptrs[i];

    if (has_left_dep) {
      colidxs[rowptrsi++] = perm[flat_i - 1];
    }

    if (has_top_dep) {
      colidxs[rowptrsi++] = perm[flat_i - length];
    }

    rowptrs[i + 1] += rowptrsi;
  }
}

void generate_matrix_info_balanced_tree_updown(bool down, int degree,
                                               int height, int *n) {
  *n = 0; // Could also be evaluated via geometric series
  int t = 1;
  for (auto i = 0; i < height; ++i) {
    *n += t;
    t *= degree;
  }
}

// TODO: Debug in the up case
void fill_matrix_balanced_tree_updown(std::int32_t *rowptrs,
                                      std::int32_t *colidxs, bool down,
                                      int degree, int height, int n) {

  if (down) {
    rowptrs[0] = 0;
    rowptrs[1] = 0;

    for (int i = 1; i < n; ++i) {
      const auto parent = (i - 1) / degree;

      colidxs[rowptrs[i]] = parent;

      rowptrs[i + 1] = rowptrs[i] + 1;
    }
  } else {

    const auto num_leafs = ipow(degree, (unsigned int)height - 1);

    for (int i = 0; i < num_leafs; ++i) {
      rowptrs[i] = 0;
      rowptrs[i + 1] = 0;
    }

    int cur_level_size = ipow(degree, (unsigned int)std::max(height - 2, 0));
    int cur_level_i = 0;
    for (int i = num_leafs; i < n; ++i) {

      if (cur_level_i == cur_level_size) {
        cur_level_size /= degree;
        cur_level_i = 0;
      }

      const auto parent_start =
          i - cur_level_i - cur_level_size * degree + cur_level_i * degree;
      const auto parent_end = parent_start + degree;

      auto rowptrsi = rowptrs[i];

      for (auto parent = parent_start; i < parent_end; ++parent) {
        colidxs[rowptrsi++] = parent;
      }

      rowptrs[i + 1] = rowptrsi;

      cur_level_i += 1;
    }
  }
}

void fill_matrix_random_graph(std::int32_t *rowptrs, std::int32_t *colidxs,
                              double alpha, int n, int window) {

  std::random_device rd;
  std::mt19937 gen(rd());

  rowptrs[0] = 0;

  for (int i = 0; i < n; ++i) {

    auto rowptrsi = rowptrs[i];

    std::binomial_distribution<int> num_deps_dist(i, alpha);
    const auto num_deps = num_deps_dist(gen);

    const auto effect_window = std::max(num_deps, window);

    // DEBUG
    // if(effect_window > window){
    //   std::cout << "Enlarged effective window, more deps" << std::endl;
    // }

    auto numbers = std::set<int>();
    auto dep_dist = std::uniform_int_distribution<int>(
        std::max(0, i - 1 - effect_window), i - 1);

    while (numbers.size() < num_deps) {
      numbers.insert(dep_dist(gen));
    }

    // add dependencies
    // for(auto depi = 0; depi < num_deps; ++depi){
    //   colidxs[rowptrsi++] = *(numbers.begin() + depi);
    // }
    std::copy(numbers.begin(), numbers.end(), colidxs + rowptrsi);
    rowptrsi += num_deps;

    rowptrs[i + 1] += rowptrsi;
  }
}

void sptrsv_random_graph(nvbench::state &state) {

  const auto n = static_cast<std::size_t>(state.get_int64("n"));
  const auto element_work = state.get_int64("element_work");
  const auto diag_work = state.get_int64("diag_work");
  const auto sleep = state.get_int64("sleep");
  const auto window = state.get_int64("window");
  const auto avg_nnz = state.get_int64("avg_nnz");

  if (sleep > 0 && !has_sleep_support()) {
    state.skip("GPU has no sleep support");
    return;
  }

  std::vector<std::int32_t> rowptrs_h(n + 1, 0);
  std::vector<std::int32_t> colidxs_h(2 * avg_nnz * n,
                                      0); // this just an approx
  thrust::device_vector<std::int32_t> rowptrs(n + 1, 0);
  thrust::device_vector<std::int32_t> colidxs(2 * avg_nnz * n,
                                              0); // This just an estimate

  fill_matrix_random_graph(rowptrs_h.data(), colidxs_h.data(),
                           avg_nnz / (double)n, n, window);

  thrust::copy(rowptrs_h.begin(), rowptrs_h.end(), rowptrs.begin());
  thrust::copy(colidxs_h.begin(), colidxs_h.end(), colidxs.begin());

  // throughput info
  state.add_global_memory_reads<nvbench::int32_t>(avg_nnz * n);
  state.add_global_memory_writes<nvbench::int32_t>(n);

  const auto blocksize = state.get_int64("blocksize");
  thrust::device_vector<std::int32_t> signal_flags(n, -1);
  const auto num_blocks = (n + blocksize - 1) / blocksize;

  state.exec(
      nvbench::exec_tag::timer, [&](nvbench::launch &launch, auto &timer) {
#ifdef USE_HIP
        auto policy = thrust::hip::par.on(launch.get_stream());
#else
        auto policy = thrust::cuda::par.on(launch.get_stream());
#endif

        thrust::fill(policy, signal_flags.begin(), signal_flags.end(), -1);
        timer.start();
        basic_sptrsv_kernel<<<num_blocks, blocksize, 0, launch.get_stream()>>>(
            thrust::raw_pointer_cast(signal_flags.data()),
            thrust::raw_pointer_cast(rowptrs.data()),
            thrust::raw_pointer_cast(colidxs.data()), n, element_work,
            diag_work, sleep);
        timer.stop();
      });
}
NVBENCH_BENCH(sptrsv_random_graph)
    .add_int64_axis("n", {100, 10000, 1000000})
    .add_int64_axis("window", {10, 100, 1000, 10000, 100000})
    .add_int64_axis("avg_nnz", {5})
    .add_int64_axis("element_work", {0, 16, 256})
    .add_int64_axis("diag_work", {0, 16, 256})
    .add_int64_axis("blocksize", {64, 512})
    .add_int64_axis("sleep", {0});

void sptrsv_random_graph_wide(nvbench::state &state) {

  const auto n = static_cast<std::size_t>(state.get_int64("n"));
  const auto element_work = state.get_int64("element_work");
  const auto diag_work = state.get_int64("diag_work");
  const auto sleep = state.get_int64("sleep");
  const auto window = state.get_int64("window");
  const auto avg_nnz = state.get_int64("avg_nnz");
  const auto width = state.get_int64("width");

  if (sleep > 0 && !has_sleep_support()) {
    state.skip("GPU has no sleep support");
    return;
  }

  std::vector<std::int32_t> rowptrs_h(n + 1, 0);
  std::vector<std::int32_t> colidxs_h(2 * avg_nnz * n,
                                      0); // this just an approx
  thrust::device_vector<std::int32_t> rowptrs(n + 1, 0);
  thrust::device_vector<std::int32_t> colidxs(2 * avg_nnz * n, 0);

  fill_matrix_random_graph(rowptrs_h.data(), colidxs_h.data(),
                           avg_nnz / (double)n, n, window);

  thrust::copy(rowptrs_h.begin(), rowptrs_h.end(), rowptrs.begin());
  thrust::copy(colidxs_h.begin(), colidxs_h.end(), colidxs.begin());

  // throughput info
  state.add_global_memory_reads<nvbench::int32_t>(avg_nnz * n);
  state.add_global_memory_writes<nvbench::int32_t>(n);

  const auto blocksize = state.get_int64("blocksize");
  thrust::device_vector<std::int32_t> signal_flags(n, -1);
  const auto num_blocks = (width * n + blocksize - 1) / blocksize;

  state.exec(nvbench::exec_tag::timer, [&](nvbench::launch &launch,
                                           auto &timer) {
#ifdef USE_HIP
    auto policy = thrust::hip::par.on(launch.get_stream());
#else
        auto policy = thrust::cuda::par.on(launch.get_stream());
#endif

    thrust::fill(policy, signal_flags.begin(), signal_flags.end(), -1);
    timer.start();
    basic_sptrsv_wide_kernel<<<num_blocks, blocksize, 0, launch.get_stream()>>>(
        thrust::raw_pointer_cast(signal_flags.data()),
        thrust::raw_pointer_cast(rowptrs.data()),
        thrust::raw_pointer_cast(colidxs.data()), n, element_work, diag_work,
        sleep, width);
    timer.stop();
  });
}
NVBENCH_BENCH(sptrsv_random_graph_wide)
    .add_int64_axis("n", {100, 10000, 1000000})
    .add_int64_axis("window", {10, 100, 1000, 10000, 100000})
    .add_int64_axis("avg_nnz", {5})
    .add_int64_axis("element_work", {0, 16, 256})
    .add_int64_axis("diag_work", {0})
    .add_int64_axis("blocksize", {64})
    .add_int64_axis("width", {1, 4, 16, 32})
    .add_int64_axis("sleep", {0});

void sptrsv_propagation_2dgrid_row(nvbench::state &state) {

  const auto length = static_cast<std::size_t>(state.get_int64("length"));
  const auto element_work = state.get_int64("element_work");
  const auto diag_work = state.get_int64("diag_work");
  const auto sleep = state.get_int64("sleep");

  if (sleep > 0 && !has_sleep_support()) {
    state.skip("GPU has no sleep support");
    return;
  }

  const auto n = length * length;
  std::vector<std::int32_t> rowptrs_h(n + 1, 0);
  std::vector<std::int32_t> colidxs_h(2 * n - 2 * length + 1, 0);
  thrust::device_vector<std::int32_t> rowptrs(n + 1, 0);
  thrust::device_vector<std::int32_t> colidxs(2 * n - 2 * length + 1, 0);

  fill_matrix_2dgrid_row(rowptrs_h.data(), colidxs_h.data(), length);

  thrust::copy(rowptrs_h.begin(), rowptrs_h.end(), rowptrs.begin());
  thrust::copy(colidxs_h.begin(), colidxs_h.end(), colidxs.begin());

  const auto blocksize = state.get_int64("blocksize");
  thrust::device_vector<std::int32_t> signal_flags(length * length, -1);
  const auto num_blocks = (length * length + blocksize - 1) / blocksize;

  state.exec(
      nvbench::exec_tag::timer, [&](nvbench::launch &launch, auto &timer) {
#ifdef USE_HIP
        auto policy = thrust::hip::par.on(launch.get_stream());
#else
        auto policy = thrust::cuda::par.on(launch.get_stream());
#endif

        thrust::fill(policy, signal_flags.begin(), signal_flags.end(), -1);
        timer.start();
        basic_sptrsv_kernel<<<num_blocks, blocksize, 0, launch.get_stream()>>>(
            thrust::raw_pointer_cast(signal_flags.data()),
            thrust::raw_pointer_cast(rowptrs.data()),
            thrust::raw_pointer_cast(colidxs.data()), n, element_work,
            diag_work, sleep);
        timer.stop();
      });
}
NVBENCH_BENCH(sptrsv_propagation_2dgrid_row)
    .add_int64_axis("length", {100, 300, 1000})
    .add_int64_axis("element_work", {0, 16, 256})
    .add_int64_axis("diag_work", {0, 16, 256})
    .add_int64_axis("blocksize", {64, 512})
    .add_int64_axis("sleep", {0});

void sptrsv_propagation_2dgrid_diag(nvbench::state &state) {

  const auto length = static_cast<std::size_t>(state.get_int64("length"));
  const auto element_work = state.get_int64("element_work");
  const auto diag_work = state.get_int64("diag_work");
  const auto sleep = state.get_int64("sleep");

  if (sleep > 0 && !has_sleep_support()) {
    state.skip("GPU has no sleep support");
    return;
  }

  const auto n = length * length;
  std::vector<std::int32_t> rowptrs_h(n + 1, 0);
  std::vector<std::int32_t> colidxs_h(2 * n - 2 * length + 1, 0);
  thrust::device_vector<std::int32_t> rowptrs(n + 1, 0);
  thrust::device_vector<std::int32_t> colidxs(2 * n - 2 * length + 1, 0);

  fill_matrix_2dgrid_diag(rowptrs_h.data(), colidxs_h.data(), length);

  thrust::copy(rowptrs_h.begin(), rowptrs_h.end(), rowptrs.begin());
  thrust::copy(colidxs_h.begin(), colidxs_h.end(), colidxs.begin());

  const auto blocksize = state.get_int64("blocksize");
  thrust::device_vector<std::int32_t> signal_flags(length * length, -1);
  const auto num_blocks = (length * length + blocksize - 1) / blocksize;

  state.exec(
      nvbench::exec_tag::timer, [&](nvbench::launch &launch, auto &timer) {
#ifdef USE_HIP
        auto policy = thrust::hip::par.on(launch.get_stream());
#else
        auto policy = thrust::cuda::par.on(launch.get_stream());
#endif

        thrust::fill(policy, signal_flags.begin(), signal_flags.end(), -1);
        timer.start();
        basic_sptrsv_kernel<<<num_blocks, blocksize, 0, launch.get_stream()>>>(
            thrust::raw_pointer_cast(signal_flags.data()),
            thrust::raw_pointer_cast(rowptrs.data()),
            thrust::raw_pointer_cast(colidxs.data()), n, element_work,
            diag_work, sleep);
        timer.stop();
      });
}
NVBENCH_BENCH(sptrsv_propagation_2dgrid_diag)
    .add_int64_axis("length", {100, 300, 1000})
    .add_int64_axis("element_work", {0, 16, 256})
    .add_int64_axis("diag_work", {0, 16, 256})
    .add_int64_axis("blocksize", {64, 512})
    .add_int64_axis("sleep", {0});

void sptrsv_propagation_balanced_tree_down(nvbench::state &state) {

  const auto element_work = state.get_int64("element_work");
  const auto diag_work = state.get_int64("diag_work");
  const auto sleep = state.get_int64("sleep");
  const auto degree = state.get_int64("degree");
  const auto height = state.get_int64("height");

  if (sleep > 0 && !has_sleep_support()) {
    state.skip("GPU has no sleep support");
    return;
  }

  int n = 0;
  bool down = true;
  generate_matrix_info_balanced_tree_updown(down, degree, height, &n);
  const auto nnz = n - 1;

  std::vector<std::int32_t> rowptrs_h(n + 1, 0);
  std::vector<std::int32_t> colidxs_h(nnz, 0);
  thrust::device_vector<std::int32_t> rowptrs(n + 1, 0);
  thrust::device_vector<std::int32_t> colidxs(nnz, 0);

  fill_matrix_balanced_tree_updown(rowptrs_h.data(), colidxs_h.data(), down,
                                   degree, height, n);

  thrust::copy(rowptrs_h.begin(), rowptrs_h.end(), rowptrs.begin());
  thrust::copy(colidxs_h.begin(), colidxs_h.end(), colidxs.begin());

  // throughput info
  state.add_global_memory_reads<nvbench::int32_t>(nnz);
  state.add_global_memory_writes<nvbench::int32_t>(n);
  state.add_element_count(diag_work * n, "DiagonalWork");
  state.add_element_count(element_work * nnz, "ElementWork");

  const auto blocksize = state.get_int64("blocksize");
  thrust::device_vector<std::int32_t> signal_flags(n, -1);
  const auto num_blocks = (n + blocksize - 1) / blocksize;

  state.exec(
      nvbench::exec_tag::timer, [&](nvbench::launch &launch, auto &timer) {
#ifdef USE_HIP
        auto policy = thrust::hip::par.on(launch.get_stream());
#else
        auto policy = thrust::cuda::par.on(launch.get_stream());
#endif

        thrust::fill(policy, signal_flags.begin(), signal_flags.end(), -1);
        timer.start();
        basic_sptrsv_kernel<<<num_blocks, blocksize, 0, launch.get_stream()>>>(
            thrust::raw_pointer_cast(signal_flags.data()),
            thrust::raw_pointer_cast(rowptrs.data()),
            thrust::raw_pointer_cast(colidxs.data()), n, element_work,
            diag_work, sleep);
        timer.stop();
      });
}
NVBENCH_BENCH(sptrsv_propagation_balanced_tree_down)
    .add_int64_axis("height", {1, 2, 3, 4, 5, 6})
    .add_int64_axis("degree", {1, 2, 4, 8, 16})
    .add_int64_axis("element_work", {0, 16, 256})
    .add_int64_axis("diag_work", {0, 16, 256})
    // .add_int64_axis("blocksize", {64, 512})
    .add_int64_axis("blocksize", {64})
    .add_int64_axis("sleep", {0});

void sptrsv_propagation_balanced_tree_up(nvbench::state &state) {

  const auto element_work = state.get_int64("element_work");
  const auto diag_work = state.get_int64("diag_work");
  const auto sleep = state.get_int64("sleep");
  const auto degree = state.get_int64("degree");
  const auto height = state.get_int64("height");

  if (sleep > 0 && !has_sleep_support()) {
    state.skip("GPU has no sleep support");
    return;
  }

  int n = 0;
  bool down = false;
  generate_matrix_info_balanced_tree_updown(down, degree, height, &n);
  const auto nnz = n - 1;

  std::vector<std::int32_t> rowptrs_h(n + 1, 0);
  std::vector<std::int32_t> colidxs_h(nnz, 0);
  thrust::device_vector<std::int32_t> rowptrs(n + 1, 0);
  thrust::device_vector<std::int32_t> colidxs(nnz, 0);

  fill_matrix_balanced_tree_updown(rowptrs_h.data(), colidxs_h.data(), down,
                                   degree, height, n);

  thrust::copy(rowptrs_h.begin(), rowptrs_h.end(), rowptrs.begin());
  thrust::copy(colidxs_h.begin(), colidxs_h.end(), colidxs.begin());

  const auto blocksize = state.get_int64("blocksize");
  thrust::device_vector<std::int32_t> signal_flags(n, -1);
  const auto num_blocks = (n + blocksize - 1) / blocksize;

  state.exec(
      nvbench::exec_tag::timer, [&](nvbench::launch &launch, auto &timer) {
#ifdef USE_HIP
        auto policy = thrust::hip::par.on(launch.get_stream());
#else
        auto policy = thrust::cuda::par.on(launch.get_stream());
#endif

        thrust::fill(policy, signal_flags.begin(), signal_flags.end(), -1);
        timer.start();
        basic_sptrsv_kernel<<<num_blocks, blocksize, 0, launch.get_stream()>>>(
            thrust::raw_pointer_cast(signal_flags.data()),
            thrust::raw_pointer_cast(rowptrs.data()),
            thrust::raw_pointer_cast(colidxs.data()), n, element_work,
            diag_work, sleep);
        timer.stop();
      });
}
NVBENCH_BENCH(sptrsv_propagation_balanced_tree_up)
    .add_int64_axis("height", {1, 2, 3, 4, 8})
    .add_int64_axis("degree", {1, 2, 4, 8})
    .add_int64_axis("element_work", {0, 16, 256})
    .add_int64_axis("diag_work", {0, 16, 256})
    .add_int64_axis("blocksize", {64})
    .add_int64_axis("sleep", {0});