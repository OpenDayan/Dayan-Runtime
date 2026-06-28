#ifndef _NPU_RUNTIME_H_
#define _NPU_RUNTIME_H_

#include <NPUDriver.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NPU_MAX_KERNEL_ARGS 30
#define NPU_CORE_ID_AUTO (-1)
#define NPU_CORE_ID_AUTO (-1)
#define NPU_PRINTCHAR_BUF_SIZE 0x1000000

// Device Kernel Set Description Structure
struct NPUDeviceKernelSetDesc {
  uint32_t magic;    // Reserve for future use
  uint32_t flags;    // Reserve for future use
  uint32_t version;  // Reserve for future use
  uint32_t reserve1; // Reserved for future use

  uint64_t load_start; // Load address of the binary
  uint64_t load_size;  // Size of the binary

  uint64_t kernel_count;  // Number of kernels in the set
  uint64_t general_entry; // General entry point for the device
  uint64_t args_ptr;      // Pointer to the arguments for the device

  uint8_t *data;                 // Pointer to the binary
  struct NPUKernelDesc *kernels; // Pointer to an array of kernel descriptions
  uint64_t heap;                 // Pointer to the start address of heap
  uint64_t perf;                 // Pointer to the start address of perf counter
};

struct NPUKernelELFImage {
  const uint8_t *data;
  size_t size;
};

struct NPUKernelIRBCImage {
  const uint8_t *data;
  size_t size;
};

struct NPUFatBinaryEntry {
  // Optional logical id/module name for diagnostics.
  const char *id;
  // ISA version tag used for runtime variant selection (e.g. "v0.1").
  const char *isa_version;
  // Embedded ELF image produced at compile time.
  const struct NPUKernelELFImage *image;
  // Optional LLVM bitcode image used as JIT fallback.
  const struct NPUKernelIRBCImage *ir_bc_image;
};

// Device Runtime Stateful Structure
struct NPURuntimeState {
  // TODO
  struct NPUDeviceState dev_state;
  struct NPUDeviceKernelSetDesc kernel_set_desc;
  void *kernelset_so_handle;
  uint8_t *owned_kernelset_data;
  struct NPUKernelDesc *owned_kernel_descs;
  char **owned_kernel_names;
  size_t owned_kernel_name_count;
  
  // Printchar buffer
  uint64_t printchar_base;
  uint64_t printchar_size;

  // Memory Management Context
  void *mm;

  // Optional nvcc-like embedded fatbinary registry (non-owning pointers).
  const struct NPUFatBinaryEntry *fatbin_entries;
  size_t fatbin_entry_count;
  // Device ISA version used to pick the best fatbin variant.
  const char *device_isa_version;
};

struct GeneralEntryArgs {
  uint64_t kernel_entry;
  uint32_t ret;
  uint32_t arg_count;
  uint64_t putchar_buf;
  uint64_t putchar_buflength;
  uint64_t putchar_wptr;
  uint64_t args[NPU_MAX_KERNEL_ARGS];
};

// Kernel Description Structure
struct NPUKernelDesc {
  uint64_t entry;   // Entry point of the kernel
  const char *name; // Name of the kernel binary
};

typedef enum {
  NPU_SUCCESS,
  NPU_ERROR_INVALID_ARGUMENT,
  NPU_ERROR_OUT_OF_MEMORY,
  NPU_ERROR_NOT_INITIALIZED,
  NPU_ERROR_KERNEL_NOT_FOUND,
  NPU_ERROR_DEVICE_FAILURE
} NPURuntimeError_t;

// Define Basic Runtime API
ssize_t npu_runtime_init(struct NPURuntimeState *runtime);
void npu_runtime_deinit(struct NPURuntimeState *runtime);

ssize_t npu_runtime_load_kernelset(
    struct NPURuntimeState *runtime,
    const char *so_path);
// Load kernel set from a shared object path.
// - so_path: explicit .so path. If NULL/empty, runtime performs no eager load
//            and relies on lazy per-kernel loading in npu_runtime_get_kernel_desc.

ssize_t npu_runtime_load_kernelset_image(
    struct NPURuntimeState *runtime,
    const struct NPUKernelELFImage *image);
// Load kernel set from an embedded ELF image. This enables nvcc-like flow:
// compile-time embedding + runtime registration/loading without dlopen().

ssize_t npu_runtime_load_kernelset_from_memory(
    struct NPURuntimeState *runtime,
    const uint8_t *elf_data,
    size_t elf_size);
// Convenience wrapper of npu_runtime_load_kernelset_image for raw [ptr,size].

ssize_t npu_runtime_kernel_start(const struct NPURuntimeState *runtime,
    const struct NPUKernelDesc *kernel_desc,
    int* kernel_ret, size_t arg_count, ...);

ssize_t npu_runtime_multi_kernel_start_ex(
    const struct NPURuntimeState *runtime,
    size_t core_count,
    const int core_ids[],
    const struct NPUKernelDesc *kernel_descs[],
    int *kernel_rets,
    const size_t arg_counts[],
    const uint64_t *per_core_args[]);
ssize_t npu_runtime_multi_kernel_start_auto(
    const struct NPURuntimeState *runtime,
    size_t core_count,
    const struct NPUKernelDesc *kernel_descs[],
    int *kernel_rets,
    const size_t arg_counts[],
    const uint64_t *per_core_args[],
    int *allocated_core_ids);

// Device Memory Management
ssize_t npu_mm_init(void** mm, uint64_t heap_start, uint64_t heap_end);
ssize_t npu_mm_deinit(void** mm);
ssize_t npu_runtime_malloc(const struct NPURuntimeState *runtime,
                                     uint64_t* dev_ptr, size_t size);
ssize_t npu_runtime_free(const struct NPURuntimeState *runtime,
                                     uint64_t  dev_ptr);
ssize_t npu_runtime_reserve(const struct NPURuntimeState *runtime,
                                     uint64_t  dev_ptr, size_t size);
ssize_t npu_runtime_get_meminfo(const struct NPURuntimeState *runtime,
        size_t * free, size_t * total);
ssize_t npu_runtime_get_kernel_desc(const struct NPURuntimeState *runtime,
  const char *kernel_name,
  struct NPUKernelDesc **kernel_desc_out);
// If not found in currently loaded kernelset, runtime will try redirect-loading
// "<NPU_KERNEL_SO_DIR>/<kernel_name>.so" (default dir: /workspace/npu-kernels/bin).
ssize_t npu_runtime_memset(const struct NPURuntimeState *runtime,
                                     uint64_t  dev_ptr, int c, size_t n);

__attribute__((always_inline)) static inline uint32_t float_as_u32(float i) {
  union {
    float f;
    uint32_t i;
  } t;
  t.f = i;
  return t.i;
}

__attribute__((always_inline)) static inline float u32_as_float(uint32_t i) {
  union {
    float f;
    uint32_t i;
  } t;
  t.i = i;
  return t.f;
}

#ifdef __cplusplus
}
#endif

#endif
