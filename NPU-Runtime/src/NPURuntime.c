#include <string.h>
#include <sys/types.h>

#include "NPUDriver.h"

#include "NPURuntime.h"

#ifdef __cplusplus
extern "C" {
#endif

NPURuntimeError_t npu_runtime_init(struct NPURuntimeState *runtime) {
  ssize_t ret;
  if (runtime == NULL) {
    return NPU_ERROR_INVALID_ARGUMENT;
  }

  ret = npu_device_init(&runtime->dev_state);
  if (ret < 0) {
    return NPU_ERROR_DEVICE_FAILURE;
  }

  return NPU_SUCCESS;
}

void npu_runtime_deinit(struct NPURuntimeState *runtime) {}

NPURuntimeError_t npu_runtime_load_kernelset(
    struct NPURuntimeState *runtime,
    const struct NPUDeviceKernelSetDesc *kernel_set_desc) {
  if (runtime == NULL || kernel_set_desc == NULL) {
    return NPU_ERROR_INVALID_ARGUMENT;
  }

  runtime->kernel_set_desc = *kernel_set_desc;

  ssize_t ret;

  // load bin
  ret = npu_memcpy_to_dev(&runtime->dev_state,
                          runtime->kernel_set_desc.load_start,
                          kernel_set_desc->data, kernel_set_desc->load_size);
  if (ret < 0) {
    return NPU_ERROR_DEVICE_FAILURE;
  }

  return NPU_SUCCESS;
}

NPURuntimeError_t
npu_runtime_kernel_start(const struct NPURuntimeState *runtime,
                         const struct NPUKernelDesc *kernel_desc,
                         uint64_t *args, size_t arg_count) {
  if (runtime == NULL || arg_count > NPU_MAX_KERNEL_ARGS) {
    return NPU_ERROR_INVALID_ARGUMENT;
  }

  if (kernel_desc == NULL || kernel_desc->entry == 0) {
    return NPU_ERROR_KERNEL_NOT_FOUND;
  }

  ssize_t ret;

  struct GeneralEntryArgs *general_entry_args =
      (struct GeneralEntryArgs *)malloc(
          sizeof(struct GeneralEntryArgs) +
          (arg_count * sizeof(uint64_t))); // Allocate memory for args

  if (general_entry_args == NULL) {
    return NPU_ERROR_OUT_OF_MEMORY;
  }

  general_entry_args->kernel_entry = kernel_desc->entry;
  for (size_t i = 0; i < arg_count; i++) {
    general_entry_args->args[i] = args[i];
  }

  ret =
      npu_memcpy_to_dev(&runtime->dev_state, runtime->kernel_set_desc.args_ptr,
                        &general_entry_args, arg_count * sizeof(uint64_t));

  free(general_entry_args); // Free the allocated memory

  if (ret < 0) {
    return NPU_ERROR_DEVICE_FAILURE;
  }

  ret = npu_reset(&runtime->dev_state);
  if (ret < 0) {
    return NPU_ERROR_DEVICE_FAILURE;
  }

  ret = npu_start(&runtime->dev_state, runtime->kernel_set_desc.general_entry);
  if (ret < 0) {
    return NPU_ERROR_DEVICE_FAILURE;
  }

  return NPU_SUCCESS;
}

// Device Memory Management
NPURuntimeError_t npu_runtime_malloc(const struct NPURuntimeState *runtime,
                                     void **dev_ptr, size_t size) {
  return NPU_SUCCESS;
}

NPURuntimeError_t npu_runtime_free(const struct NPURuntimeState *runtime,
                                   void *dev_ptr) {
  return NPU_SUCCESS;
}

NPURuntimeError_t npu_runtime_reserve(const struct NPURuntimeState *runtime,
                                      size_t size, uint64_t start_address) {
  return NPU_SUCCESS;
}

#ifdef __cplusplus
}
#endif
