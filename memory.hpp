#include <cstdint>

#ifdef __CUDA_ARCH__
__device__ __forceinline__ std::int32_t load_relaxed_shared(std::int32_t *ptr) {
  std::int32_t result;
#if __CUDA_ARCH__ < 700
  asm volatile("ld.shared.volatile.b32 %0, [%1];"
               : "=r"(result)
               : "r"(static_cast<std::uint32_t>(__cvta_generic_to_shared(ptr)))
               : "memory");
#else
  asm volatile("ld.shared.relaxed.b32.cta %0, [%1];"
               : "=r"(result)
               : "r"(static_cast<std::uint32_t>(__cvta_generic_to_shared(ptr)))
               : "memory");
#endif
  return result;
}

__device__ __forceinline__ void store_relaxed_shared(std::int32_t *ptr,
                                                     std::int32_t result) {
#if __CUDA_ARCH__ < 700
  asm volatile("st.shared.volatile.b32 [%0], %1;" ::"r"(
                   static_cast<std::uint32_t>(__cvta_generic_to_shared(ptr))),
               "r"(result)
               : "memory");
#else
  asm volatile("st.shared.relaxed.b32.cta [%0], %1;" ::"r"(
                   static_cast<std::uint32_t>(__cvta_generic_to_shared(ptr))),
               "r"(result)
               : "memory");
#endif
}

__device__ __forceinline__ std::int64_t load_relaxed_shared(std::int64_t *ptr) {
  std::int64_t result;
#if __CUDA_ARCH__ < 700
  asm volatile("ld.shared.volatile.b64 %0, [%1];"
               : "=l"(result)
               : "r"(static_cast<std::uint32_t>(__cvta_generic_to_shared(ptr)))
               : "memory");
#else
  asm volatile("ld.shared.relaxed.b64.cta %0, [%1];"
               : "=l"(result)
               : "r"(static_cast<std::uint32_t>(__cvta_generic_to_shared(ptr)))
               : "memory");
#endif
  return result;
}

__device__ __forceinline__ void store_relaxed_shared(std::int64_t *ptr,
                                                     std::int64_t result) {
#if __CUDA_ARCH__ < 700
  asm volatile("st.shared.volatile.b64 [%0], %1;" ::"r"(
                   static_cast<std::uint32_t>(__cvta_generic_to_shared(ptr))),
               "l"(result)
               : "memory");
#else
  asm volatile("st.shared.relaxed.b64.cta [%0], %1;" ::"r"(
                   static_cast<std::uint32_t>(__cvta_generic_to_shared(ptr))),
               "l"(result)
               : "memory");
#endif
}

__device__ __forceinline__ float load_relaxed_shared(float *ptr) {
  float result;
#if __CUDA_ARCH__ < 700
  asm volatile("ld.shared.volatile.f32 %0, [%1];"
               : "=f"(result)
               : "r"(static_cast<std::uint32_t>(__cvta_generic_to_shared(ptr)))
               : "memory");
#else
  asm volatile("ld.shared.relaxed.f32.cta %0, [%1];"
               : "=f"(result)
               : "r"(static_cast<std::uint32_t>(__cvta_generic_to_shared(ptr)))
               : "memory");
#endif
  return result;
}

__device__ __forceinline__ void store_relaxed_shared(float *ptr, float result) {
#if __CUDA_ARCH__ < 700
  asm volatile("st.shared.volatile.f32 [%0], %1;" ::"r"(
                   static_cast<std::uint32_t>(__cvta_generic_to_shared(ptr))),
               "f"(result)
               : "memory");
#else
  asm volatile("st.shared.relaxed.f32.cta [%0], %1;" ::"r"(
                   static_cast<std::uint32_t>(__cvta_generic_to_shared(ptr))),
               "f"(result)
               : "memory");
#endif
}

__device__ __forceinline__ double load_relaxed_shared(double *ptr) {
  double result;
#if __CUDA_ARCH__ < 700
  asm volatile("ld.shared.volatile.f64 %0, [%1];"
               : "=d"(result)
               : "r"(static_cast<std::uint32_t>(__cvta_generic_to_shared(ptr)))
               : "memory");
#else
  asm volatile("ld.shared.relaxed.f64.cta %0, [%1];"
               : "=d"(result)
               : "r"(static_cast<std::uint32_t>(__cvta_generic_to_shared(ptr)))
               : "memory");
#endif
  return result;
}

__device__ __forceinline__ void store_relaxed_shared(double *ptr,
                                                     double result) {
#if __CUDA_ARCH__ < 700
  asm volatile("st.shared.volatile.f64 [%0], %1;" ::"r"(
                   static_cast<std::uint32_t>(__cvta_generic_to_shared(ptr))),
               "d"(result)
               : "memory");
#else
  asm volatile("st.shared.relaxed.f64.cta [%0], %1;" ::"r"(
                   static_cast<std::uint32_t>(__cvta_generic_to_shared(ptr))),
               "d"(result)
               : "memory");
#endif
}

__device__ __forceinline__ std::int32_t load_acquire_shared(std::int32_t *ptr) {
  std::int32_t result;
#if __CUDA_ARCH__ < 700
  asm volatile("ld.shared.volatile.b32 %0, [%1];"
               : "=r"(result)
               : "r"(static_cast<std::uint32_t>(__cvta_generic_to_shared(ptr)))
               : "memory");
  asm volatile("membar.cta;" ::: "memory");
#else
  asm volatile("ld.shared.acquire.b32.cta %0, [%1];"
               : "=r"(result)
               : "r"(static_cast<std::uint32_t>(__cvta_generic_to_shared(ptr)))
               : "memory");
#endif
  return result;
}

__device__ __forceinline__ void store_release_shared(std::int32_t *ptr,
                                                     std::int32_t result) {
#if __CUDA_ARCH__ < 700
  asm volatile("membar.cta;" ::: "memory");
  asm volatile("st.shared.volatile.b32 [%0], %1;" ::"r"(
                   static_cast<std::uint32_t>(__cvta_generic_to_shared(ptr))),
               "r"(result)
               : "memory");
#else
  asm volatile("st.shared.release.b32.cta [%0], %1;" ::"r"(
                   static_cast<std::uint32_t>(__cvta_generic_to_shared(ptr))),
               "r"(result)
               : "memory");
#endif
}

__device__ __forceinline__ std::int64_t load_acquire_shared(std::int64_t *ptr) {
  std::int64_t result;
#if __CUDA_ARCH__ < 700
  asm volatile("ld.shared.volatile.b64 %0, [%1];"
               : "=l"(result)
               : "r"(static_cast<std::uint32_t>(__cvta_generic_to_shared(ptr)))
               : "memory");
  asm volatile("membar.cta;" ::: "memory");
#else
  asm volatile("ld.shared.acquire.b64.cta %0, [%1];"
               : "=l"(result)
               : "r"(static_cast<std::uint32_t>(__cvta_generic_to_shared(ptr)))
               : "memory");
#endif
  return result;
}

__device__ __forceinline__ void store_release_shared(std::int64_t *ptr,
                                                     std::int64_t result) {
#if __CUDA_ARCH__ < 700
  asm volatile("membar.cta;" ::: "memory");
  asm volatile("st.shared.volatile.b64 [%0], %1;" ::"r"(
                   static_cast<std::uint32_t>(__cvta_generic_to_shared(ptr))),
               "l"(result)
               : "memory");
#else
  asm volatile("st.shared.release.b64.cta [%0], %1;" ::"r"(
                   static_cast<std::uint32_t>(__cvta_generic_to_shared(ptr))),
               "l"(result)
               : "memory");
#endif
}

__device__ __forceinline__ float load_acquire_shared(float *ptr) {
  float result;
#if __CUDA_ARCH__ < 700
  asm volatile("ld.shared.volatile.f32 %0, [%1];"
               : "=f"(result)
               : "r"(static_cast<std::uint32_t>(__cvta_generic_to_shared(ptr)))
               : "memory");
  asm volatile("membar.cta;" ::: "memory");
#else
  asm volatile("ld.shared.acquire.f32.cta %0, [%1];"
               : "=f"(result)
               : "r"(static_cast<std::uint32_t>(__cvta_generic_to_shared(ptr)))
               : "memory");
#endif
  return result;
}

__device__ __forceinline__ void store_release_shared(float *ptr, float result) {
#if __CUDA_ARCH__ < 700
  asm volatile("membar.cta;" ::: "memory");
  asm volatile("st.shared.volatile.f32 [%0], %1;" ::"r"(
                   static_cast<std::uint32_t>(__cvta_generic_to_shared(ptr))),
               "f"(result)
               : "memory");
#else
  asm volatile("st.shared.release.f32.cta [%0], %1;" ::"r"(
                   static_cast<std::uint32_t>(__cvta_generic_to_shared(ptr))),
               "f"(result)
               : "memory");
#endif
}

__device__ __forceinline__ double load_acquire_shared(double *ptr) {
  double result;
#if __CUDA_ARCH__ < 700
  asm volatile("ld.shared.volatile.f64 %0, [%1];"
               : "=d"(result)
               : "r"(static_cast<std::uint32_t>(__cvta_generic_to_shared(ptr)))
               : "memory");
  asm volatile("membar.cta;" ::: "memory");
#else
  asm volatile("ld.shared.acquire.f64.cta %0, [%1];"
               : "=d"(result)
               : "r"(static_cast<std::uint32_t>(__cvta_generic_to_shared(ptr)))
               : "memory");
#endif
  return result;
}

__device__ __forceinline__ void store_release_shared(double *ptr,
                                                     double result) {
#if __CUDA_ARCH__ < 700
  asm volatile("membar.cta;" ::: "memory");
  asm volatile("st.shared.volatile.f64 [%0], %1;" ::"r"(
                   static_cast<std::uint32_t>(__cvta_generic_to_shared(ptr))),
               "d"(result)
               : "memory");
#else
  asm volatile("st.shared.release.f64.cta [%0], %1;" ::"r"(
                   static_cast<std::uint32_t>(__cvta_generic_to_shared(ptr))),
               "d"(result)
               : "memory");
#endif
}

__device__ __forceinline__ std::int32_t load_relaxed(std::int32_t *ptr) {
  std::int32_t result;
#if __CUDA_ARCH__ < 700
  asm volatile("ld.volatile.b32 %0, [%1];"
               : "=r"(result)
               : "l"(ptr)
               : "memory");
#else
  asm volatile("ld.relaxed.b32.cta %0, [%1];"
               : "=r"(result)
               : "l"(ptr)
               : "memory");
#endif
  return result;
}

__device__ __forceinline__ void store_relaxed(std::int32_t *ptr,
                                              std::int32_t result) {
#if __CUDA_ARCH__ < 700
  asm volatile("st.volatile.b32 [%0], %1;" ::"l"(ptr), "r"(result) : "memory");
#else
  asm volatile("st.relaxed.b32.cta [%0], %1;" ::"l"(ptr), "r"(result)
               : "memory");
#endif
}

__device__ __forceinline__ std::int64_t load_relaxed(std::int64_t *ptr) {
  std::int64_t result;
#if __CUDA_ARCH__ < 700
  asm volatile("ld.volatile.b64 %0, [%1];"
               : "=l"(result)
               : "l"(ptr)
               : "memory");
#else
  asm volatile("ld.relaxed.b64.cta %0, [%1];"
               : "=l"(result)
               : "l"(ptr)
               : "memory");
#endif
  return result;
}

__device__ __forceinline__ void store_relaxed(std::int64_t *ptr,
                                              std::int64_t result) {
#if __CUDA_ARCH__ < 700
  asm volatile("st.volatile.b64 [%0], %1;" ::"l"(ptr), "l"(result) : "memory");
#else
  asm volatile("st.relaxed.b64.cta [%0], %1;" ::"l"(ptr), "l"(result)
               : "memory");
#endif
}

__device__ __forceinline__ float load_relaxed(float *ptr) {
  float result;
#if __CUDA_ARCH__ < 700
  asm volatile("ld.volatile.f32 %0, [%1];"
               : "=f"(result)
               : "l"(ptr)
               : "memory");
#else
  asm volatile("ld.relaxed.f32.cta %0, [%1];"
               : "=f"(result)
               : "l"(ptr)
               : "memory");
#endif
  return result;
}

__device__ __forceinline__ void store_relaxed(float *ptr, float result) {
#if __CUDA_ARCH__ < 700
  asm volatile("st.volatile.f32 [%0], %1;" ::"l"(ptr), "f"(result) : "memory");
#else
  asm volatile("st.relaxed.f32.cta [%0], %1;" ::"l"(ptr), "f"(result)
               : "memory");
#endif
}

__device__ __forceinline__ double load_relaxed(double *ptr) {
  double result;
#if __CUDA_ARCH__ < 700
  asm volatile("ld.volatile.f64 %0, [%1];"
               : "=d"(result)
               : "l"(ptr)
               : "memory");
#else
  asm volatile("ld.relaxed.f64.cta %0, [%1];"
               : "=d"(result)
               : "l"(ptr)
               : "memory");
#endif
  return result;
}

__device__ __forceinline__ void store_relaxed(double *ptr, double result) {
#if __CUDA_ARCH__ < 700
  asm volatile("st.volatile.f64 [%0], %1;" ::"l"(ptr), "d"(result) : "memory");
#else
  asm volatile("st.relaxed.f64.cta [%0], %1;" ::"l"(ptr), "d"(result)
               : "memory");
#endif
}

__device__ __forceinline__ std::int32_t load_acquire(std::int32_t *ptr) {
  std::int32_t result;
#if __CUDA_ARCH__ < 700
  asm volatile("ld.volatile.b32 %0, [%1];"
               : "=r"(result)
               : "l"(ptr)
               : "memory");
  asm volatile("membar.cta;" ::: "memory");
#else
  asm volatile("ld.acquire.b32.cta %0, [%1];"
               : "=r"(result)
               : "l"(ptr)
               : "memory");
#endif
  return result;
}

__device__ __forceinline__ void store_release(std::int32_t *ptr,
                                              std::int32_t result) {
#if __CUDA_ARCH__ < 700
  asm volatile("membar.cta;" ::: "memory");
  asm volatile("st.volatile.b32 [%0], %1;" ::"l"(ptr), "r"(result) : "memory");
#else
  asm volatile("st.release.b32.cta [%0], %1;" ::"l"(ptr), "r"(result)
               : "memory");
#endif
}

__device__ __forceinline__ std::int64_t load_acquire(std::int64_t *ptr) {
  std::int64_t result;
#if __CUDA_ARCH__ < 700
  asm volatile("ld.volatile.b64 %0, [%1];"
               : "=l"(result)
               : "l"(ptr)
               : "memory");
  asm volatile("membar.cta;" ::: "memory");
#else
  asm volatile("ld.acquire.b64.cta %0, [%1];"
               : "=l"(result)
               : "l"(ptr)
               : "memory");
#endif
  return result;
}

__device__ __forceinline__ void store_release(std::int64_t *ptr,
                                              std::int64_t result) {
#if __CUDA_ARCH__ < 700
  asm volatile("membar.cta;" ::: "memory");
  asm volatile("st.volatile.b64 [%0], %1;" ::"l"(ptr), "l"(result) : "memory");
#else
  asm volatile("st.release.b64.cta [%0], %1;" ::"l"(ptr), "l"(result)
               : "memory");
#endif
}

__device__ __forceinline__ float load_acquire(float *ptr) {
  float result;
#if __CUDA_ARCH__ < 700
  asm volatile("ld.volatile.f32 %0, [%1];"
               : "=f"(result)
               : "l"(ptr)
               : "memory");
  asm volatile("membar.cta;" ::: "memory");
#else
  asm volatile("ld.acquire.f32.cta %0, [%1];"
               : "=f"(result)
               : "l"(ptr)
               : "memory");
#endif
  return result;
}

__device__ __forceinline__ void store_release(float *ptr, float result) {
#if __CUDA_ARCH__ < 700
  asm volatile("membar.cta;" ::: "memory");
  asm volatile("st.volatile.f32 [%0], %1;" ::"l"(ptr), "f"(result) : "memory");
#else
  asm volatile("st.release.f32.cta [%0], %1;" ::"l"(ptr), "f"(result)
               : "memory");
#endif
}

__device__ __forceinline__ double load_acquire(double *ptr) {
  double result;
#if __CUDA_ARCH__ < 700
  asm volatile("ld.volatile.f64 %0, [%1];"
               : "=d"(result)
               : "l"(ptr)
               : "memory");
  asm volatile("membar.cta;" ::: "memory");
#else
  asm volatile("ld.acquire.f64.cta %0, [%1];"
               : "=d"(result)
               : "l"(ptr)
               : "memory");
#endif
  return result;
}

__device__ __forceinline__ void store_release(double *ptr, double result) {
#if __CUDA_ARCH__ < 700
  asm volatile("membar.cta;" ::: "memory");
  asm volatile("st.volatile.f64 [%0], %1;" ::"l"(ptr), "d"(result) : "memory");
#else
  asm volatile("st.release.f64.cta [%0], %1;" ::"l"(ptr), "d"(result)
               : "memory");
#endif
}
#else
template <typename T> __device__ T load_relaxed(volatile T *ptr) {
  return *ptr;
}

template <typename T> __device__ void store_relaxed(volatile T *ptr, T result) {
  *ptr = result;
}
#endif