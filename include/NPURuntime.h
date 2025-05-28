#ifndef _NPU_RUNTIME_H_
#define _NPU_RUNTIME_H_

#include <NPUDriver.h>
#include <stdint.h>
#include <stdlib.h>

// Device Runtime Stateful Structure
struct NPURuntimeState {
    // TODO
    struct NPUDeviceState *dev_state;
};

// Device Kernel Set Description Structure
struct NPUDeviceKernelSetDesc {
    uint32_t magic;        // Reserve for future use
    uint32_t flags;        // Reserve for future use
    uint32_t version;      // Reserve for future use
    uint32_t reserve1;     // Reserved for future use

    uint64_t load_start;   // Load address of the binary
    uint64_t load_size;    // Size of the binary

    uint64_t kernel_count;  // Number of kernels in the set
    uint64_t general_entry; // General entry point for the device
    uint64_t args_ptr;      // Pointer to the arguments for the device
    
    uint8_t *data;         // Pointer to the binary
    struct NPUKernelDesc *kernels; // Pointer to an array of kernel descriptions
};

// Kernel Description Structure
struct NPUKernelDesc {
    uint64_t   entry; // Entry point of the kernel
    const char *name; // Name of the kernel binary
};

// Define Basic Runtime API
void npu_runtime_init(struct NPURuntimeState *runtime);
void npu_runtime_deinit(struct NPURuntimeState *runtime);

void npu_runtime_load_kernelset(struct NPURuntimeState *runtime, struct NPUDeviceKernelSetDesc *kernel_set_desc);

void npu_runtime_kernel_start(struct NPURuntimeState *runtime, struct NPUKernelDesc *kernel_desc, uint64_t *args, size_t arg_count);

// Device Memory Management
long npu_runtime_malloc(struct NPURuntimeState *runtime, size_t size, uint64_t *ptr);
long npu_runtime_free(struct NPURuntimeState *runtime, uint64_t *ptr);
long npu_runtime_reserve(struct NPURuntimeState *runtime, size_t size, uint64_t start_address);

#endif

