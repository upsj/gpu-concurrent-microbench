#include <cstdint>
#include <cstring>
#include <thrust/complex.h>

#ifdef __CUDA_ARCH__
/**
 * Transforms a generic CUDA pointer pointing to shared memory to a
 * shared memory pointer for use in PTX assembly.
 * CUDA PTX assembly uses 32bit pointers for shared memory addressing.
 * The result is undefined for a generic pointer pointing to anything but
 * shared memory.
 */
__device__ __forceinline__ std::uint32_t
convert_generic_ptr_to_smem_ptr(void *ptr) {
// see
// https://github.com/NVIDIA/cutlass/blob/
//     6fc5008803fe4e81b81a836fcd3a88258f4e5bbf/
//     include/cutlass/arch/memory_sm75.h#L90
// for reasoning behind this implementation
#if (!defined(__clang__) && __CUDACC_VER_MAJOR__ >= 11)
  return static_cast<std::uint32_t>(__cvta_generic_to_shared(ptr));
#elif (!defined(__clang__) && CUDACC_VER_MAJOR__ == 10 &&                      \
       __CUDACC_VER_MINOR__ >= 2)
  return __nvvm_get_smem_pointer(ptr);
#else
  std::uint32_t smem_ptr;
  asm("{{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 "
      "%0, smem_ptr; }}"
      : "=r"(smem_ptr)
      : "l"(ptr));
  return smem_ptr;
#endif
}

__device__ __forceinline__ void membar_acq_rel() {
#if __CUDA_ARCH__ < 700
  asm volatile("membar.gl;" ::: "memory");
#else
  asm volatile("fence.acq_rel.gpu;" ::: "memory");
#endif
}

__device__ __forceinline__ void membar_acq_rel_shared() {
#if __CUDA_ARCH__ < 700
  asm volatile("membar.cta;" ::: "memory");
#else
  asm volatile("fence.acq_rel.cta;" ::: "memory");
#endif
}

__device__ __forceinline__ void membar_acq_rel_local() {
#if __CUDA_ARCH__ < 700
  asm volatile("membar.cta;" ::: "memory");
#else
  asm volatile("fence.acq_rel.cta;" ::: "memory");
#endif
}

__device__ __forceinline__ std::int32_t
load_relaxed_shared(const std::int32_t *ptr) {
  std::int32_t result;
#if __CUDA_ARCH__ < 700
  asm volatile(
      "ld.volatile.shared.s32 %0, [%1];"
      : "=r"(result)
      : "r"(convert_generic_ptr_to_smem_ptr(const_cast<std::int32_t *>(ptr)))
      : "memory");
#else
  asm volatile(
      "ld.relaxed.cta.shared.s32 %0, [%1];"
      : "=r"(result)
      : "r"(convert_generic_ptr_to_smem_ptr(const_cast<std::int32_t *>(ptr)))
      : "memory");
#endif

  return result;
}

__device__ __forceinline__ void store_relaxed_shared(std::int32_t *ptr,
                                                     std::int32_t result) {
#if __CUDA_ARCH__ < 700
  asm volatile("st.volatile.shared.s32 [%0], %1;" ::"r"(
                   convert_generic_ptr_to_smem_ptr(ptr)),
               "r"(result)
               : "memory");
#else
  asm volatile("st.relaxed.cta.shared.s32 [%0], %1;" ::"r"(
                   convert_generic_ptr_to_smem_ptr(ptr)),
               "r"(result)
               : "memory");
#endif
}

__device__ __forceinline__ std::int64_t
load_relaxed_shared(const std::int64_t *ptr) {
  std::int64_t result;
#if __CUDA_ARCH__ < 700
  asm volatile(
      "ld.volatile.shared.s64 %0, [%1];"
      : "=l"(result)
      : "r"(convert_generic_ptr_to_smem_ptr(const_cast<std::int64_t *>(ptr)))
      : "memory");
#else
  asm volatile(
      "ld.relaxed.cta.shared.s64 %0, [%1];"
      : "=l"(result)
      : "r"(convert_generic_ptr_to_smem_ptr(const_cast<std::int64_t *>(ptr)))
      : "memory");
#endif

  return result;
}

__device__ __forceinline__ void store_relaxed_shared(std::int64_t *ptr,
                                                     std::int64_t result) {
#if __CUDA_ARCH__ < 700
  asm volatile("st.volatile.shared.s64 [%0], %1;" ::"r"(
                   convert_generic_ptr_to_smem_ptr(ptr)),
               "l"(result)
               : "memory");
#else
  asm volatile("st.relaxed.cta.shared.s64 [%0], %1;" ::"r"(
                   convert_generic_ptr_to_smem_ptr(ptr)),
               "l"(result)
               : "memory");
#endif
}

__device__ __forceinline__ float load_relaxed_shared(const float *ptr) {
  float result;
#if __CUDA_ARCH__ < 700
  asm volatile("ld.volatile.shared.f32 %0, [%1];"
               : "=f"(result)
               : "r"(convert_generic_ptr_to_smem_ptr(const_cast<float *>(ptr)))
               : "memory");
#else
  asm volatile("ld.relaxed.cta.shared.f32 %0, [%1];"
               : "=f"(result)
               : "r"(convert_generic_ptr_to_smem_ptr(const_cast<float *>(ptr)))
               : "memory");
#endif

  return result;
}

__device__ __forceinline__ void store_relaxed_shared(float *ptr, float result) {
#if __CUDA_ARCH__ < 700
  asm volatile("st.volatile.shared.f32 [%0], %1;" ::"r"(
                   convert_generic_ptr_to_smem_ptr(ptr)),
               "f"(result)
               : "memory");
#else
  asm volatile("st.relaxed.cta.shared.f32 [%0], %1;" ::"r"(
                   convert_generic_ptr_to_smem_ptr(ptr)),
               "f"(result)
               : "memory");
#endif
}

__device__ __forceinline__ double load_relaxed_shared(const double *ptr) {
  double result;
#if __CUDA_ARCH__ < 700
  asm volatile("ld.volatile.shared.f64 %0, [%1];"
               : "=d"(result)
               : "r"(convert_generic_ptr_to_smem_ptr(const_cast<double *>(ptr)))
               : "memory");
#else
  asm volatile("ld.relaxed.cta.shared.f64 %0, [%1];"
               : "=d"(result)
               : "r"(convert_generic_ptr_to_smem_ptr(const_cast<double *>(ptr)))
               : "memory");
#endif

  return result;
}

__device__ __forceinline__ void store_relaxed_shared(double *ptr,
                                                     double result) {
#if __CUDA_ARCH__ < 700
  asm volatile("st.volatile.shared.f64 [%0], %1;" ::"r"(
                   convert_generic_ptr_to_smem_ptr(ptr)),
               "d"(result)
               : "memory");
#else
  asm volatile("st.relaxed.cta.shared.f64 [%0], %1;" ::"r"(
                   convert_generic_ptr_to_smem_ptr(ptr)),
               "d"(result)
               : "memory");
#endif
}

__device__ __forceinline__ std::int32_t
load_acquire_shared(const std::int32_t *ptr) {
  std::int32_t result;
#if __CUDA_ARCH__ < 700
  asm volatile(
      "ld.volatile.shared.s32 %0, [%1];"
      : "=r"(result)
      : "r"(convert_generic_ptr_to_smem_ptr(const_cast<std::int32_t *>(ptr)))
      : "memory");
#else
  asm volatile(
      "ld.acquire.cta.shared.s32 %0, [%1];"
      : "=r"(result)
      : "r"(convert_generic_ptr_to_smem_ptr(const_cast<std::int32_t *>(ptr)))
      : "memory");
#endif
  membar_acq_rel_shared();
  return result;
}

__device__ __forceinline__ void store_release_shared(std::int32_t *ptr,
                                                     std::int32_t result) {
  membar_acq_rel_shared();
#if __CUDA_ARCH__ < 700
  asm volatile("st.volatile.shared.s32 [%0], %1;" ::"r"(
                   convert_generic_ptr_to_smem_ptr(ptr)),
               "r"(result)
               : "memory");
#else
  asm volatile("st.release.cta.shared.s32 [%0], %1;" ::"r"(
                   convert_generic_ptr_to_smem_ptr(ptr)),
               "r"(result)
               : "memory");
#endif
}

__device__ __forceinline__ std::int64_t
load_acquire_shared(const std::int64_t *ptr) {
  std::int64_t result;
#if __CUDA_ARCH__ < 700
  asm volatile(
      "ld.volatile.shared.s64 %0, [%1];"
      : "=l"(result)
      : "r"(convert_generic_ptr_to_smem_ptr(const_cast<std::int64_t *>(ptr)))
      : "memory");
#else
  asm volatile(
      "ld.acquire.cta.shared.s64 %0, [%1];"
      : "=l"(result)
      : "r"(convert_generic_ptr_to_smem_ptr(const_cast<std::int64_t *>(ptr)))
      : "memory");
#endif
  membar_acq_rel_shared();
  return result;
}

__device__ __forceinline__ void store_release_shared(std::int64_t *ptr,
                                                     std::int64_t result) {
  membar_acq_rel_shared();
#if __CUDA_ARCH__ < 700
  asm volatile("st.volatile.shared.s64 [%0], %1;" ::"r"(
                   convert_generic_ptr_to_smem_ptr(ptr)),
               "l"(result)
               : "memory");
#else
  asm volatile("st.release.cta.shared.s64 [%0], %1;" ::"r"(
                   convert_generic_ptr_to_smem_ptr(ptr)),
               "l"(result)
               : "memory");
#endif
}

__device__ __forceinline__ float load_acquire_shared(const float *ptr) {
  float result;
#if __CUDA_ARCH__ < 700
  asm volatile("ld.volatile.shared.f32 %0, [%1];"
               : "=f"(result)
               : "r"(convert_generic_ptr_to_smem_ptr(const_cast<float *>(ptr)))
               : "memory");
#else
  asm volatile("ld.acquire.cta.shared.f32 %0, [%1];"
               : "=f"(result)
               : "r"(convert_generic_ptr_to_smem_ptr(const_cast<float *>(ptr)))
               : "memory");
#endif
  membar_acq_rel_shared();
  return result;
}

__device__ __forceinline__ void store_release_shared(float *ptr, float result) {
  membar_acq_rel_shared();
#if __CUDA_ARCH__ < 700
  asm volatile("st.volatile.shared.f32 [%0], %1;" ::"r"(
                   convert_generic_ptr_to_smem_ptr(ptr)),
               "f"(result)
               : "memory");
#else
  asm volatile("st.release.cta.shared.f32 [%0], %1;" ::"r"(
                   convert_generic_ptr_to_smem_ptr(ptr)),
               "f"(result)
               : "memory");
#endif
}

__device__ __forceinline__ double load_acquire_shared(const double *ptr) {
  double result;
#if __CUDA_ARCH__ < 700
  asm volatile("ld.volatile.shared.f64 %0, [%1];"
               : "=d"(result)
               : "r"(convert_generic_ptr_to_smem_ptr(const_cast<double *>(ptr)))
               : "memory");
#else
  asm volatile("ld.acquire.cta.shared.f64 %0, [%1];"
               : "=d"(result)
               : "r"(convert_generic_ptr_to_smem_ptr(const_cast<double *>(ptr)))
               : "memory");
#endif
  membar_acq_rel_shared();
  return result;
}

__device__ __forceinline__ void store_release_shared(double *ptr,
                                                     double result) {
  membar_acq_rel_shared();
#if __CUDA_ARCH__ < 700
  asm volatile("st.volatile.shared.f64 [%0], %1;" ::"r"(
                   convert_generic_ptr_to_smem_ptr(ptr)),
               "d"(result)
               : "memory");
#else
  asm volatile("st.release.cta.shared.f64 [%0], %1;" ::"r"(
                   convert_generic_ptr_to_smem_ptr(ptr)),
               "d"(result)
               : "memory");
#endif
}

__device__ __forceinline__ std::int32_t
load_relaxed_local(const std::int32_t *ptr) {
  std::int32_t result;
#if __CUDA_ARCH__ < 700
  asm volatile("ld.volatile.s32 %0, [%1];"
               : "=r"(result)
               : "l"(const_cast<std::int32_t *>(ptr))
               : "memory");
#else
  asm volatile("ld.relaxed.cta.s32 %0, [%1];"
               : "=r"(result)
               : "l"(const_cast<std::int32_t *>(ptr))
               : "memory");
#endif

  return result;
}

__device__ __forceinline__ void store_relaxed_local(std::int32_t *ptr,
                                                    std::int32_t result) {
#if __CUDA_ARCH__ < 700
  asm volatile("st.volatile.s32 [%0], %1;" ::"l"(ptr), "r"(result) : "memory");
#else
  asm volatile("st.relaxed.cta.s32 [%0], %1;" ::"l"(ptr), "r"(result)
               : "memory");
#endif
}

__device__ __forceinline__ std::int64_t
load_relaxed_local(const std::int64_t *ptr) {
  std::int64_t result;
#if __CUDA_ARCH__ < 700
  asm volatile("ld.volatile.s64 %0, [%1];"
               : "=l"(result)
               : "l"(const_cast<std::int64_t *>(ptr))
               : "memory");
#else
  asm volatile("ld.relaxed.cta.s64 %0, [%1];"
               : "=l"(result)
               : "l"(const_cast<std::int64_t *>(ptr))
               : "memory");
#endif

  return result;
}

__device__ __forceinline__ void store_relaxed_local(std::int64_t *ptr,
                                                    std::int64_t result) {
#if __CUDA_ARCH__ < 700
  asm volatile("st.volatile.s64 [%0], %1;" ::"l"(ptr), "l"(result) : "memory");
#else
  asm volatile("st.relaxed.cta.s64 [%0], %1;" ::"l"(ptr), "l"(result)
               : "memory");
#endif
}

__device__ __forceinline__ float load_relaxed_local(const float *ptr) {
  float result;
#if __CUDA_ARCH__ < 700
  asm volatile("ld.volatile.f32 %0, [%1];"
               : "=f"(result)
               : "l"(const_cast<float *>(ptr))
               : "memory");
#else
  asm volatile("ld.relaxed.cta.f32 %0, [%1];"
               : "=f"(result)
               : "l"(const_cast<float *>(ptr))
               : "memory");
#endif

  return result;
}

__device__ __forceinline__ void store_relaxed_local(float *ptr, float result) {
#if __CUDA_ARCH__ < 700
  asm volatile("st.volatile.f32 [%0], %1;" ::"l"(ptr), "f"(result) : "memory");
#else
  asm volatile("st.relaxed.cta.f32 [%0], %1;" ::"l"(ptr), "f"(result)
               : "memory");
#endif
}

__device__ __forceinline__ double load_relaxed_local(const double *ptr) {
  double result;
#if __CUDA_ARCH__ < 700
  asm volatile("ld.volatile.f64 %0, [%1];"
               : "=d"(result)
               : "l"(const_cast<double *>(ptr))
               : "memory");
#else
  asm volatile("ld.relaxed.cta.f64 %0, [%1];"
               : "=d"(result)
               : "l"(const_cast<double *>(ptr))
               : "memory");
#endif

  return result;
}

__device__ __forceinline__ void store_relaxed_local(double *ptr,
                                                    double result) {
#if __CUDA_ARCH__ < 700
  asm volatile("st.volatile.f64 [%0], %1;" ::"l"(ptr), "d"(result) : "memory");
#else
  asm volatile("st.relaxed.cta.f64 [%0], %1;" ::"l"(ptr), "d"(result)
               : "memory");
#endif
}

__device__ __forceinline__ std::int32_t
load_acquire_local(const std::int32_t *ptr) {
  std::int32_t result;
#if __CUDA_ARCH__ < 700
  asm volatile("ld.volatile.s32 %0, [%1];"
               : "=r"(result)
               : "l"(const_cast<std::int32_t *>(ptr))
               : "memory");
#else
  asm volatile("ld.acquire.cta.s32 %0, [%1];"
               : "=r"(result)
               : "l"(const_cast<std::int32_t *>(ptr))
               : "memory");
#endif
  membar_acq_rel_local();
  return result;
}

__device__ __forceinline__ void store_release_local(std::int32_t *ptr,
                                                    std::int32_t result) {
  membar_acq_rel_local();
#if __CUDA_ARCH__ < 700
  asm volatile("st.volatile.s32 [%0], %1;" ::"l"(ptr), "r"(result) : "memory");
#else
  asm volatile("st.release.cta.s32 [%0], %1;" ::"l"(ptr), "r"(result)
               : "memory");
#endif
}

__device__ __forceinline__ std::int64_t
load_acquire_local(const std::int64_t *ptr) {
  std::int64_t result;
#if __CUDA_ARCH__ < 700
  asm volatile("ld.volatile.s64 %0, [%1];"
               : "=l"(result)
               : "l"(const_cast<std::int64_t *>(ptr))
               : "memory");
#else
  asm volatile("ld.acquire.cta.s64 %0, [%1];"
               : "=l"(result)
               : "l"(const_cast<std::int64_t *>(ptr))
               : "memory");
#endif
  membar_acq_rel_local();
  return result;
}

__device__ __forceinline__ void store_release_local(std::int64_t *ptr,
                                                    std::int64_t result) {
  membar_acq_rel_local();
#if __CUDA_ARCH__ < 700
  asm volatile("st.volatile.s64 [%0], %1;" ::"l"(ptr), "l"(result) : "memory");
#else
  asm volatile("st.release.cta.s64 [%0], %1;" ::"l"(ptr), "l"(result)
               : "memory");
#endif
}

__device__ __forceinline__ float load_acquire_local(const float *ptr) {
  float result;
#if __CUDA_ARCH__ < 700
  asm volatile("ld.volatile.f32 %0, [%1];"
               : "=f"(result)
               : "l"(const_cast<float *>(ptr))
               : "memory");
#else
  asm volatile("ld.acquire.cta.f32 %0, [%1];"
               : "=f"(result)
               : "l"(const_cast<float *>(ptr))
               : "memory");
#endif
  membar_acq_rel_local();
  return result;
}

__device__ __forceinline__ void store_release_local(float *ptr, float result) {
  membar_acq_rel_local();
#if __CUDA_ARCH__ < 700
  asm volatile("st.volatile.f32 [%0], %1;" ::"l"(ptr), "f"(result) : "memory");
#else
  asm volatile("st.release.cta.f32 [%0], %1;" ::"l"(ptr), "f"(result)
               : "memory");
#endif
}

__device__ __forceinline__ double load_acquire_local(const double *ptr) {
  double result;
#if __CUDA_ARCH__ < 700
  asm volatile("ld.volatile.f64 %0, [%1];"
               : "=d"(result)
               : "l"(const_cast<double *>(ptr))
               : "memory");
#else
  asm volatile("ld.acquire.cta.f64 %0, [%1];"
               : "=d"(result)
               : "l"(const_cast<double *>(ptr))
               : "memory");
#endif
  membar_acq_rel_local();
  return result;
}

__device__ __forceinline__ void store_release_local(double *ptr,
                                                    double result) {
  membar_acq_rel_local();
#if __CUDA_ARCH__ < 700
  asm volatile("st.volatile.f64 [%0], %1;" ::"l"(ptr), "d"(result) : "memory");
#else
  asm volatile("st.release.cta.f64 [%0], %1;" ::"l"(ptr), "d"(result)
               : "memory");
#endif
}

__device__ __forceinline__ std::int32_t load_relaxed(const std::int32_t *ptr) {
  std::int32_t result;
#if __CUDA_ARCH__ < 700
  asm volatile("ld.volatile.s32 %0, [%1];"
               : "=r"(result)
               : "l"(const_cast<std::int32_t *>(ptr))
               : "memory");
#else
  asm volatile("ld.relaxed.gpu.s32 %0, [%1];"
               : "=r"(result)
               : "l"(const_cast<std::int32_t *>(ptr))
               : "memory");
#endif

  return result;
}

__device__ __forceinline__ void store_relaxed(std::int32_t *ptr,
                                              std::int32_t result) {
#if __CUDA_ARCH__ < 700
  asm volatile("st.volatile.s32 [%0], %1;" ::"l"(ptr), "r"(result) : "memory");
#else
  asm volatile("st.relaxed.gpu.s32 [%0], %1;" ::"l"(ptr), "r"(result)
               : "memory");
#endif
}

__device__ __forceinline__ std::int64_t load_relaxed(const std::int64_t *ptr) {
  std::int64_t result;
#if __CUDA_ARCH__ < 700
  asm volatile("ld.volatile.s64 %0, [%1];"
               : "=l"(result)
               : "l"(const_cast<std::int64_t *>(ptr))
               : "memory");
#else
  asm volatile("ld.relaxed.gpu.s64 %0, [%1];"
               : "=l"(result)
               : "l"(const_cast<std::int64_t *>(ptr))
               : "memory");
#endif

  return result;
}

__device__ __forceinline__ void store_relaxed(std::int64_t *ptr,
                                              std::int64_t result) {
#if __CUDA_ARCH__ < 700
  asm volatile("st.volatile.s64 [%0], %1;" ::"l"(ptr), "l"(result) : "memory");
#else
  asm volatile("st.relaxed.gpu.s64 [%0], %1;" ::"l"(ptr), "l"(result)
               : "memory");
#endif
}

__device__ __forceinline__ float load_relaxed(const float *ptr) {
  float result;
#if __CUDA_ARCH__ < 700
  asm volatile("ld.volatile.f32 %0, [%1];"
               : "=f"(result)
               : "l"(const_cast<float *>(ptr))
               : "memory");
#else
  asm volatile("ld.relaxed.gpu.f32 %0, [%1];"
               : "=f"(result)
               : "l"(const_cast<float *>(ptr))
               : "memory");
#endif

  return result;
}

__device__ __forceinline__ void store_relaxed(float *ptr, float result) {
#if __CUDA_ARCH__ < 700
  asm volatile("st.volatile.f32 [%0], %1;" ::"l"(ptr), "f"(result) : "memory");
#else
  asm volatile("st.relaxed.gpu.f32 [%0], %1;" ::"l"(ptr), "f"(result)
               : "memory");
#endif
}

__device__ __forceinline__ double load_relaxed(const double *ptr) {
  double result;
#if __CUDA_ARCH__ < 700
  asm volatile("ld.volatile.f64 %0, [%1];"
               : "=d"(result)
               : "l"(const_cast<double *>(ptr))
               : "memory");
#else
  asm volatile("ld.relaxed.gpu.f64 %0, [%1];"
               : "=d"(result)
               : "l"(const_cast<double *>(ptr))
               : "memory");
#endif

  return result;
}

__device__ __forceinline__ void store_relaxed(double *ptr, double result) {
#if __CUDA_ARCH__ < 700
  asm volatile("st.volatile.f64 [%0], %1;" ::"l"(ptr), "d"(result) : "memory");
#else
  asm volatile("st.relaxed.gpu.f64 [%0], %1;" ::"l"(ptr), "d"(result)
               : "memory");
#endif
}

__device__ __forceinline__ std::int32_t load_acquire(const std::int32_t *ptr) {
  std::int32_t result;
#if __CUDA_ARCH__ < 700
  asm volatile("ld.volatile.s32 %0, [%1];"
               : "=r"(result)
               : "l"(const_cast<std::int32_t *>(ptr))
               : "memory");
#else
  asm volatile("ld.acquire.gpu.s32 %0, [%1];"
               : "=r"(result)
               : "l"(const_cast<std::int32_t *>(ptr))
               : "memory");
#endif
  membar_acq_rel();
  return result;
}

__device__ __forceinline__ void store_release(std::int32_t *ptr,
                                              std::int32_t result) {
  membar_acq_rel();
#if __CUDA_ARCH__ < 700
  asm volatile("st.volatile.s32 [%0], %1;" ::"l"(ptr), "r"(result) : "memory");
#else
  asm volatile("st.release.gpu.s32 [%0], %1;" ::"l"(ptr), "r"(result)
               : "memory");
#endif
}

__device__ __forceinline__ std::int64_t load_acquire(const std::int64_t *ptr) {
  std::int64_t result;
#if __CUDA_ARCH__ < 700
  asm volatile("ld.volatile.s64 %0, [%1];"
               : "=l"(result)
               : "l"(const_cast<std::int64_t *>(ptr))
               : "memory");
#else
  asm volatile("ld.acquire.gpu.s64 %0, [%1];"
               : "=l"(result)
               : "l"(const_cast<std::int64_t *>(ptr))
               : "memory");
#endif
  membar_acq_rel();
  return result;
}

__device__ __forceinline__ void store_release(std::int64_t *ptr,
                                              std::int64_t result) {
  membar_acq_rel();
#if __CUDA_ARCH__ < 700
  asm volatile("st.volatile.s64 [%0], %1;" ::"l"(ptr), "l"(result) : "memory");
#else
  asm volatile("st.release.gpu.s64 [%0], %1;" ::"l"(ptr), "l"(result)
               : "memory");
#endif
}

__device__ __forceinline__ float load_acquire(const float *ptr) {
  float result;
#if __CUDA_ARCH__ < 700
  asm volatile("ld.volatile.f32 %0, [%1];"
               : "=f"(result)
               : "l"(const_cast<float *>(ptr))
               : "memory");
#else
  asm volatile("ld.acquire.gpu.f32 %0, [%1];"
               : "=f"(result)
               : "l"(const_cast<float *>(ptr))
               : "memory");
#endif
  membar_acq_rel();
  return result;
}

__device__ __forceinline__ void store_release(float *ptr, float result) {
  membar_acq_rel();
#if __CUDA_ARCH__ < 700
  asm volatile("st.volatile.f32 [%0], %1;" ::"l"(ptr), "f"(result) : "memory");
#else
  asm volatile("st.release.gpu.f32 [%0], %1;" ::"l"(ptr), "f"(result)
               : "memory");
#endif
}

__device__ __forceinline__ double load_acquire(const double *ptr) {
  double result;
#if __CUDA_ARCH__ < 700
  asm volatile("ld.volatile.f64 %0, [%1];"
               : "=d"(result)
               : "l"(const_cast<double *>(ptr))
               : "memory");
#else
  asm volatile("ld.acquire.gpu.f64 %0, [%1];"
               : "=d"(result)
               : "l"(const_cast<double *>(ptr))
               : "memory");
#endif
  membar_acq_rel();
  return result;
}

__device__ __forceinline__ void store_release(double *ptr, double result) {
  membar_acq_rel();
#if __CUDA_ARCH__ < 700
  asm volatile("st.volatile.f64 [%0], %1;" ::"l"(ptr), "d"(result) : "memory");
#else
  asm volatile("st.release.gpu.f64 [%0], %1;" ::"l"(ptr), "d"(result)
               : "memory");
#endif
}

__device__ __forceinline__ thrust::complex<float>
load_relaxed_shared(const thrust::complex<float> *ptr) {
  float real_result;
  float imag_result;
#if __CUDA_ARCH__ < 700
  asm volatile("ld.volatile.shared.v2.f32 {%0, %1}, [%2];"
               : "=f"(real_result), "=f"(imag_result)
               : "r"(convert_generic_ptr_to_smem_ptr(
                   const_cast<thrust::complex<float> *>(ptr)))
               : "memory");
#else
  asm volatile("ld.relaxed.cta.shared.v2.f32 {%0, %1}, [%2];"
               : "=f"(real_result), "=f"(imag_result)
               : "r"(convert_generic_ptr_to_smem_ptr(
                   const_cast<thrust::complex<float> *>(ptr)))
               : "memory");
#endif
  return thrust::complex<float>{real_result, imag_result};
}

__device__ __forceinline__ void
store_relaxed_shared(thrust::complex<float> *ptr,
                     thrust::complex<float> result) {
  auto real_result = result.real();
  auto imag_result = result.imag();
#if __CUDA_ARCH__ < 700
  asm volatile("st.volatile.shared.v2.f32 [%0], {%1, %2};" ::"r"(
                   convert_generic_ptr_to_smem_ptr(ptr)),
               "f"(real_result), "f"(imag_result)
               : "memory");
#else
  asm volatile("st.relaxed.cta.shared.v2.f32 [%0], {%1, %2};" ::"r"(
                   convert_generic_ptr_to_smem_ptr(ptr)),
               "f"(real_result), "f"(imag_result)
               : "memory");
#endif
}

__device__ __forceinline__ thrust::complex<double>
load_relaxed_shared(const thrust::complex<double> *ptr) {
  double real_result;
  double imag_result;
#if __CUDA_ARCH__ < 700
  asm volatile("ld.volatile.shared.v2.f64 {%0, %1}, [%2];"
               : "=d"(real_result), "=d"(imag_result)
               : "r"(convert_generic_ptr_to_smem_ptr(
                   const_cast<thrust::complex<double> *>(ptr)))
               : "memory");
#else
  asm volatile("ld.relaxed.cta.shared.v2.f64 {%0, %1}, [%2];"
               : "=d"(real_result), "=d"(imag_result)
               : "r"(convert_generic_ptr_to_smem_ptr(
                   const_cast<thrust::complex<double> *>(ptr)))
               : "memory");
#endif
  return thrust::complex<double>{real_result, imag_result};
}

__device__ __forceinline__ void
store_relaxed_shared(thrust::complex<double> *ptr,
                     thrust::complex<double> result) {
  auto real_result = result.real();
  auto imag_result = result.imag();
#if __CUDA_ARCH__ < 700
  asm volatile("st.volatile.shared.v2.f64 [%0], {%1, %2};" ::"r"(
                   convert_generic_ptr_to_smem_ptr(ptr)),
               "d"(real_result), "d"(imag_result)
               : "memory");
#else
  asm volatile("st.relaxed.cta.shared.v2.f64 [%0], {%1, %2};" ::"r"(
                   convert_generic_ptr_to_smem_ptr(ptr)),
               "d"(real_result), "d"(imag_result)
               : "memory");
#endif
}

__device__ __forceinline__ thrust::complex<float>
load_relaxed_local(const thrust::complex<float> *ptr) {
  float real_result;
  float imag_result;
#if __CUDA_ARCH__ < 700
  asm volatile("ld.volatile.v2.f32 {%0, %1}, [%2];"
               : "=f"(real_result), "=f"(imag_result)
               : "l"(const_cast<thrust::complex<float> *>(ptr))
               : "memory");
#else
  asm volatile("ld.relaxed.cta.v2.f32 {%0, %1}, [%2];"
               : "=f"(real_result), "=f"(imag_result)
               : "l"(const_cast<thrust::complex<float> *>(ptr))
               : "memory");
#endif
  return thrust::complex<float>{real_result, imag_result};
}

__device__ __forceinline__ void
store_relaxed_local(thrust::complex<float> *ptr,
                    thrust::complex<float> result) {
  auto real_result = result.real();
  auto imag_result = result.imag();
#if __CUDA_ARCH__ < 700
  asm volatile("st.volatile.v2.f32 [%0], {%1, %2};" ::"l"(ptr),
               "f"(real_result), "f"(imag_result)
               : "memory");
#else
  asm volatile("st.relaxed.cta.v2.f32 [%0], {%1, %2};" ::"l"(ptr),
               "f"(real_result), "f"(imag_result)
               : "memory");
#endif
}

__device__ __forceinline__ thrust::complex<double>
load_relaxed_local(const thrust::complex<double> *ptr) {
  double real_result;
  double imag_result;
#if __CUDA_ARCH__ < 700
  asm volatile("ld.volatile.v2.f64 {%0, %1}, [%2];"
               : "=d"(real_result), "=d"(imag_result)
               : "l"(const_cast<thrust::complex<double> *>(ptr))
               : "memory");
#else
  asm volatile("ld.relaxed.cta.v2.f64 {%0, %1}, [%2];"
               : "=d"(real_result), "=d"(imag_result)
               : "l"(const_cast<thrust::complex<double> *>(ptr))
               : "memory");
#endif
  return thrust::complex<double>{real_result, imag_result};
}

__device__ __forceinline__ void
store_relaxed_local(thrust::complex<double> *ptr,
                    thrust::complex<double> result) {
  auto real_result = result.real();
  auto imag_result = result.imag();
#if __CUDA_ARCH__ < 700
  asm volatile("st.volatile.v2.f64 [%0], {%1, %2};" ::"l"(ptr),
               "d"(real_result), "d"(imag_result)
               : "memory");
#else
  asm volatile("st.relaxed.cta.v2.f64 [%0], {%1, %2};" ::"l"(ptr),
               "d"(real_result), "d"(imag_result)
               : "memory");
#endif
}

__device__ __forceinline__ thrust::complex<float>
load_relaxed(const thrust::complex<float> *ptr) {
  float real_result;
  float imag_result;
#if __CUDA_ARCH__ < 700
  asm volatile("ld.volatile.v2.f32 {%0, %1}, [%2];"
               : "=f"(real_result), "=f"(imag_result)
               : "l"(const_cast<thrust::complex<float> *>(ptr))
               : "memory");
#else
  asm volatile("ld.relaxed.gpu.v2.f32 {%0, %1}, [%2];"
               : "=f"(real_result), "=f"(imag_result)
               : "l"(const_cast<thrust::complex<float> *>(ptr))
               : "memory");
#endif
  return thrust::complex<float>{real_result, imag_result};
}

__device__ __forceinline__ void store_relaxed(thrust::complex<float> *ptr,
                                              thrust::complex<float> result) {
  auto real_result = result.real();
  auto imag_result = result.imag();
#if __CUDA_ARCH__ < 700
  asm volatile("st.volatile.v2.f32 [%0], {%1, %2};" ::"l"(ptr),
               "f"(real_result), "f"(imag_result)
               : "memory");
#else
  asm volatile("st.relaxed.gpu.v2.f32 [%0], {%1, %2};" ::"l"(ptr),
               "f"(real_result), "f"(imag_result)
               : "memory");
#endif
}

__device__ __forceinline__ thrust::complex<double>
load_relaxed(const thrust::complex<double> *ptr) {
  double real_result;
  double imag_result;
#if __CUDA_ARCH__ < 700
  asm volatile("ld.volatile.v2.f64 {%0, %1}, [%2];"
               : "=d"(real_result), "=d"(imag_result)
               : "l"(const_cast<thrust::complex<double> *>(ptr))
               : "memory");
#else
  asm volatile("ld.relaxed.gpu.v2.f64 {%0, %1}, [%2];"
               : "=d"(real_result), "=d"(imag_result)
               : "l"(const_cast<thrust::complex<double> *>(ptr))
               : "memory");
#endif
  return thrust::complex<double>{real_result, imag_result};
}

__device__ __forceinline__ void store_relaxed(thrust::complex<double> *ptr,
                                              thrust::complex<double> result) {
  auto real_result = result.real();
  auto imag_result = result.imag();
#if __CUDA_ARCH__ < 700
  asm volatile("st.volatile.v2.f64 [%0], {%1, %2};" ::"l"(ptr),
               "d"(real_result), "d"(imag_result)
               : "memory");
#else
  asm volatile("st.relaxed.gpu.v2.f64 [%0], {%1, %2};" ::"l"(ptr),
               "d"(real_result), "d"(imag_result)
               : "memory");
#endif
}
#else
/*Used to map primitive types to an equivalently -
    sized / aligned type that can be *used in atomic intrinsics.*/
template <typename T> struct gcc_atomic_intrinsic_type_map {};

template <> struct gcc_atomic_intrinsic_type_map<std::int32_t> {
  using type = std::int32_t;
};

template <> struct gcc_atomic_intrinsic_type_map<float> {
  using type = std::int32_t;
};

template <> struct gcc_atomic_intrinsic_type_map<std::int64_t> {
  using type = std::int64_t;
};

template <> struct gcc_atomic_intrinsic_type_map<double> {
  using type = std::int64_t;
};

#if HIP_VERSION >= 50100000
// These intrinsics can be found used in clang/test/SemaCUDA/atomic-ops.cu
// in the LLVM source code

#define HIP_ATOMIC_LOAD(ptr, memorder, scope)                                  \
  __hip_atomic_load(ptr, memorder, scope)
#define HIP_ATOMIC_STORE(ptr, value, memorder, scope)                          \
  __hip_atomic_store(ptr, value, memorder, scope)
#define HIP_SCOPE_GPU __HIP_MEMORY_SCOPE_AGENT
#define HIP_SCOPE_THREADBLOCK __HIP_MEMORY_SCOPE_WORKGROUP
#else
#define HIP_ATOMIC_LOAD(ptr, memorder, scope) __atomic_load_n(ptr, memorder)
#define HIP_ATOMIC_STORE(ptr, value, memorder, scope)                          \
  __atomic_store_n(ptr, value, memorder)
#define HIP_SCOPE_GPU -1
#define HIP_SCOPE_THREADBLOCK -1

#endif

/**
 * Loads a value from memory using an atomic operation.
 *
 * @tparam memorder  The GCC memory ordering type
 * (https://gcc.gnu.org/onlinedocs/gcc/_005f_005fatomic-Builtins.html) to use
 * for this atomic operation.
 * @tparam scope  The visibility of this operation, i.e. which threads may have
 * written to this memory location before. HIP_SCOPE_GPU means that we want to
 * observe writes from all threads on this device, HIP_SCOPE_THREADBLOCK means
 * we want to observe only writes from within the same threadblock.
 */
template <int memorder, int scope, typename ValueType>
__device__ __forceinline__ ValueType load_generic(const ValueType *ptr) {
  using atomic_type = typename gcc_atomic_intrinsic_type_map<ValueType>::type;
  static_assert(sizeof(atomic_type) == sizeof(ValueType), "invalid map");
  static_assert(alignof(atomic_type) == alignof(ValueType), "invalid map");
  auto cast_value = HIP_ATOMIC_LOAD(reinterpret_cast<const atomic_type *>(ptr),
                                    memorder, scope);
  ValueType result{};
  memcpy(&result, &cast_value, sizeof(ValueType));
  return result;
}

/**
 * Stores a value to memory using an atomic operation.
 *
 * @tparam memorder  The GCC memory ordering type
 * (https://gcc.gnu.org/onlinedocs/gcc/_005f_005fatomic-Builtins.html) to use
 * for this atomic operation.
 * @tparam scope  The visibility of this operation, i.e. which threads may
 * observe the write to this memory location. HIP_SCOPE_GPU means that we want
 * to all threads on this device to observe it, HIP_SCOPE_THREADBLOCK means we
 * want only threads within the same threadblock to observe it.
 */
template <int memorder, int scope, typename ValueType>
__device__ __forceinline__ void store_generic(ValueType *ptr, ValueType value) {
  using atomic_type = typename gcc_atomic_intrinsic_type_map<ValueType>::type;
  static_assert(sizeof(atomic_type) == sizeof(ValueType), "invalid map");
  static_assert(alignof(atomic_type) == alignof(ValueType), "invalid map");
  atomic_type cast_value{};
  memcpy(&cast_value, &value, sizeof(ValueType));
  HIP_ATOMIC_STORE(reinterpret_cast<atomic_type *>(ptr), cast_value, memorder,
                   scope);
}

template <typename ValueType>
__device__ __forceinline__ ValueType load_relaxed(const ValueType *ptr) {
  return load_generic<__ATOMIC_RELAXED, HIP_SCOPE_GPU>(ptr);
}

template <typename ValueType>
__device__ __forceinline__ ValueType load_relaxed_shared(const ValueType *ptr) {
  return load_generic<__ATOMIC_RELAXED, HIP_SCOPE_THREADBLOCK>(ptr);
}

template <typename ValueType>
__device__ __forceinline__ ValueType load_relaxed_local(const ValueType *ptr) {
  return load_generic<__ATOMIC_RELAXED, HIP_SCOPE_THREADBLOCK>(ptr);
}

template <typename ValueType>
__device__ __forceinline__ ValueType load_acquire(const ValueType *ptr) {
  return load_generic<__ATOMIC_ACQUIRE, HIP_SCOPE_GPU>(ptr);
}

template <typename ValueType>
__device__ __forceinline__ ValueType load_acquire_shared(const ValueType *ptr) {
  return load_generic<__ATOMIC_ACQUIRE, HIP_SCOPE_THREADBLOCK>(ptr);
}

template <typename ValueType>
__device__ __forceinline__ ValueType load_acquire_local(const ValueType *ptr) {
  return load_generic<__ATOMIC_ACQUIRE, HIP_SCOPE_THREADBLOCK>(ptr);
}

template <typename ValueType>
__device__ __forceinline__ void store_relaxed(ValueType *ptr, ValueType value) {
  store_generic<__ATOMIC_RELAXED, HIP_SCOPE_GPU>(ptr, value);
}

template <typename ValueType>
__device__ __forceinline__ void store_relaxed_shared(ValueType *ptr,
                                                     ValueType value) {
  store_generic<__ATOMIC_RELAXED, HIP_SCOPE_THREADBLOCK>(ptr, value);
}

template <typename ValueType>
__device__ __forceinline__ void store_relaxed_local(ValueType *ptr,
                                                    ValueType value) {
  store_generic<__ATOMIC_RELAXED, HIP_SCOPE_THREADBLOCK>(ptr, value);
}

template <typename ValueType>
__device__ __forceinline__ void store_release(ValueType *ptr, ValueType value) {
  store_generic<__ATOMIC_RELEASE, HIP_SCOPE_GPU>(ptr, value);
}

template <typename ValueType>
__device__ __forceinline__ void store_release_shared(ValueType *ptr,
                                                     ValueType value) {
  store_generic<__ATOMIC_RELEASE, HIP_SCOPE_THREADBLOCK>(ptr, value);
}

template <typename ValueType>
__device__ __forceinline__ void store_release_local(ValueType *ptr,
                                                    ValueType value) {
  store_generic<__ATOMIC_RELEASE, HIP_SCOPE_THREADBLOCK>(ptr, value);
}

template <typename ValueType>
__device__ __forceinline__ thrust::complex<ValueType>
load_relaxed(const thrust::complex<ValueType> *ptr) {
  auto real_ptr = reinterpret_cast<const ValueType *>(ptr);
  auto real = load_relaxed(real_ptr);
  auto imag = load_relaxed(real_ptr + 1);
  return {real, imag};
}

template <typename ValueType>
__device__ __forceinline__ thrust::complex<ValueType>
load_relaxed_shared(const thrust::complex<ValueType> *ptr) {
  auto real_ptr = reinterpret_cast<const ValueType *>(ptr);
  auto real = load_relaxed_shared(real_ptr);
  auto imag = load_relaxed_shared(real_ptr + 1);
  return {real, imag};
}

template <typename ValueType>
__device__ __forceinline__ thrust::complex<ValueType>
load_relaxed_local(const thrust::complex<ValueType> *ptr) {
  auto real_ptr = reinterpret_cast<const ValueType *>(ptr);
  auto real = load_relaxed_local(real_ptr);
  auto imag = load_relaxed_local(real_ptr + 1);
  return {real, imag};
}

template <typename ValueType>
__device__ __forceinline__ void
store_relaxed(thrust::complex<ValueType> *ptr,
              thrust::complex<ValueType> value) {
  auto real_ptr = reinterpret_cast<ValueType *>(ptr);
  store_relaxed(real_ptr, value.real());
  store_relaxed(real_ptr + 1, value.imag());
}

template <typename ValueType>
__device__ __forceinline__ void
store_relaxed_shared(thrust::complex<ValueType> *ptr,
                     thrust::complex<ValueType> value) {
  auto real_ptr = reinterpret_cast<ValueType *>(ptr);
  store_relaxed_shared(real_ptr, value.real());
  store_relaxed_shared(real_ptr + 1, value.imag());
}

template <typename ValueType>
__device__ __forceinline__ void
store_relaxed_local(thrust::complex<ValueType> *ptr,
                    thrust::complex<ValueType> value) {
  auto real_ptr = reinterpret_cast<ValueType *>(ptr);
  store_relaxed_local(real_ptr, value.real());
  store_relaxed_local(real_ptr + 1, value.imag());
}

#undef HIP_ATOMIC_LOAD
#undef HIP_ATOMIC_STORE
#undef HIP_SCOPE_GPU
#undef HIP_SCOPE_THREADBLOCK

#endif