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
};

// Device Runtime Stateful Structure
struct NPURuntimeState {
  // TODO
  struct NPUDeviceState dev_state;
  struct NPUDeviceKernelSetDesc kernel_set_desc;
};

struct GeneralEntryArgs {
  uint64_t kernel_entry;
  uint32_t ret;
  uint32_t arg_count;
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
    const struct NPUDeviceKernelSetDesc *kernel_set_desc);

ssize_t npu_runtime_kernel_start(const struct NPURuntimeState *runtime,
    const struct NPUKernelDesc *kernel_desc,
    int* kernel_ret, size_t arg_count, ...);

// Device Memory Management
ssize_t npu_runtime_malloc(const struct NPURuntimeState *runtime,
                                     void **dev_ptr, size_t size);
ssize_t npu_runtime_free(const struct NPURuntimeState *runtime,
                                   void *dev_ptr);
ssize_t npu_runtime_reserve(const struct NPURuntimeState *runtime,
                                      size_t size, uint64_t start_address);
ssize_t npu_runtime_get_kernel_desc(const struct NPURuntimeState *runtime,
  const char *kernel_name,
  struct NPUKernelDesc **kernel_desc_out);

#ifdef __cplusplus
}
#endif

#endif
