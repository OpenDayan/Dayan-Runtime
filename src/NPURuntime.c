#include <string.h>
#include <sys/types.h>
#include <stdarg.h>
#include <unistd.h>

#include "NPUDriver.h"

#include "NPURuntime.h"

#ifdef __cplusplus
extern "C" {
#endif

ssize_t npu_runtime_init(struct NPURuntimeState *runtime) {
  ssize_t ret;
  if (runtime == NULL) {
    return NPU_ERROR_INVALID_ARGUMENT;
  }

  ret = npu_device_init(&runtime->dev_state);
  if (ret < 0) {
    printf("Failed to initialize NPU device: %zd\n", ret);
    return NPU_ERROR_DEVICE_FAILURE;
  }

  return NPU_SUCCESS;
}

void npu_runtime_deinit(struct NPURuntimeState *runtime) {}

ssize_t npu_runtime_load_kernelset(
    struct NPURuntimeState *runtime,
    const struct NPUDeviceKernelSetDesc *kernel_set_desc) {
  if (runtime == NULL || kernel_set_desc == NULL) {
    return NPU_ERROR_INVALID_ARGUMENT;
  }

  runtime->kernel_set_desc = *kernel_set_desc;

  ssize_t ret;

  ret = npu_abort(&runtime->dev_state);
  if (ret < 0) {
    printf("Failed to abort NPU device: %zd\n", ret);
    return NPU_ERROR_DEVICE_FAILURE;
  }

  // load bin
  ret = npu_memcpy_to_dev(&runtime->dev_state,
                          runtime->kernel_set_desc.load_start,
                          kernel_set_desc->data, kernel_set_desc->load_size);
  if (ret < 0) {
    return NPU_ERROR_DEVICE_FAILURE;
  }

  return NPU_SUCCESS;
}

// Blocking until kernel exit
ssize_t npu_runtime_kernel_start(const struct NPURuntimeState *runtime,
                         const struct NPUKernelDesc *kernel_desc,
                         int* kernel_ret, size_t arg_count, ...) {
  if (runtime == NULL || arg_count > NPU_MAX_KERNEL_ARGS) {
    return NPU_ERROR_INVALID_ARGUMENT;
  }

  if (kernel_desc == NULL || kernel_desc->entry == 0) {
    return NPU_ERROR_KERNEL_NOT_FOUND;
  }

  ssize_t ret;

  ret = npu_check(&runtime->dev_state);
  if(ret != -NPU_DEVICE_ERRNO_EMPTY) {
    return ret;
  }

  struct GeneralEntryArgs general_entry_args;

  general_entry_args.kernel_entry = kernel_desc->entry;
  general_entry_args.arg_count = arg_count;
  general_entry_args.ret = 0xdeadbeaf;
  general_entry_args.putchar_buf = 0x80040000; // 0x80040000 - 0x81000000 is putchar buf
  general_entry_args.putchar_buflength = 0x81000000 - 0x80040000;
  general_entry_args.putchar_wptr = 0;

  va_list args;
  va_start(args, arg_count);
  for(int i = 0 ; i < arg_count; i++) {
    general_entry_args.args[i] = va_arg(args, uint64_t);
  }
  va_end(args);

  ret = npu_memcpy_to_dev(&runtime->dev_state, runtime->kernel_set_desc.args_ptr,
                          &general_entry_args, sizeof(general_entry_args) + (arg_count - 30) * sizeof(uint64_t));

  if (ret < 0) {
    printf("Failed to copy arguments to device: %zd\n", ret);
    return NPU_ERROR_DEVICE_FAILURE;
  }

  ret = npu_start(&runtime->dev_state, runtime->kernel_set_desc.general_entry);
  if (ret < 0) {
    printf("Failed to start NPU device: %zd\n", ret);
    return NPU_ERROR_DEVICE_FAILURE;
  }

  long t0 = 0;
  while(1) {
    ret = npu_check(&runtime->dev_state);
    if(ret == -NPU_DEVICE_ERRNO_EMPTY) {
      ret = npu_memcpy_from_dev(&runtime->dev_state, &general_entry_args,
        runtime->kernel_set_desc.args_ptr, sizeof(general_entry_args) - 30 * sizeof(uint64_t));
      if(kernel_ret) {
        *kernel_ret = general_entry_args.ret;
      }
      static char printbuf[0x1000000];
      npu_memcpy_from_dev(&runtime->dev_state, printbuf, general_entry_args.putchar_buf, general_entry_args.putchar_wptr);
      for(long i = 0 ; i < general_entry_args.putchar_wptr ; i ++) {
        putchar(printbuf[i]);
      }
      // printf("Exit using %ldus printbuf %d\n",t0 * 100, general_entry_args.putchar_wptr);
      return NPU_SUCCESS;
    }
    t0++;
    usleep(100); // 0.1ms
  }
}

// Device Memory Management
ssize_t npu_runtime_malloc(const struct NPURuntimeState *runtime,
                                     void **dev_ptr, size_t size) {
  return NPU_SUCCESS;
}

ssize_t npu_runtime_free(const struct NPURuntimeState *runtime,
                                   void *dev_ptr) {
  return NPU_SUCCESS;
}

ssize_t npu_runtime_reserve(const struct NPURuntimeState *runtime,
                                      size_t size, uint64_t start_address) {
  return NPU_SUCCESS;
}

ssize_t npu_runtime_get_kernel_desc(const struct NPURuntimeState *runtime,
                            const char *kernel_name,
                            struct NPUKernelDesc **kernel_desc_out) {
  if (runtime == NULL || kernel_name == NULL || kernel_desc_out == NULL) {
    return NPU_ERROR_INVALID_ARGUMENT;
  }

  for (size_t i = 0; i < runtime->kernel_set_desc.kernel_count; i++) {
    if (strcmp(runtime->kernel_set_desc.kernels[i].name, kernel_name) == 0) {
      *kernel_desc_out = &runtime->kernel_set_desc.kernels[i];
      return NPU_SUCCESS;
    }
  }

  return NPU_ERROR_KERNEL_NOT_FOUND;
}

#ifdef __cplusplus
}
#endif
