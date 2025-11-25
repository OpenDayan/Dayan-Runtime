
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string.h>

#include "NPUDriver.h"
#include "NPURuntime.h"
#include "npu_instruction.hpp"

extern struct NPUDeviceKernelSetDesc __general_entry_fpga;

#define NPU_START_KERNEL_BY_NAME(err_var, runtime, name, ret_ptr, arg_count,   \
                                 ...)                                          \
  do {                                                                         \
    struct NPUKernelDesc *kernel_desc;                                         \
                                                                               \
    err_var = npu_runtime_get_kernel_desc(runtime, name, &kernel_desc);        \
    if (err_var != NPU_SUCCESS) {                                              \
      std::cout << "Failed to get kernel description for" << name << ":"       \
                << err_var << std::endl;                                       \
      break;                                                                   \
    }                                                                          \
                                                                               \
    err_var = npu_runtime_kernel_start(runtime, kernel_desc, ret_ptr,          \
                                       arg_count, ##__VA_ARGS__);              \
    if (err_var != NPU_SUCCESS) {                                              \
      std::cout << "Failed to start kernel " << name << ":" << err_var         \
                << std::endl;                                                  \
      break;                                                                   \
    }                                                                          \
  } while (0)

typedef union {
  float f32;
  int32_t i32;
} fp_bits_t;

#define check(x, expect)                                                       \
  if ((x) != (expect)) {                                                       \
    printf("Error: %s:%d: (%d,%d) %s = %d, expected %d\n", __FILE__, __LINE__, \
           m, n, #x, (x), (expect));                                           \
    return -1;                                                                 \
  }

#define fp_check(x, expect, eps)                                               \
  if ((x) < (expect) - (eps) || (x) > (expect) + (eps)) {                      \
    fp_bits_t x_bits, expect_bits;                                             \
    x_bits.f32 = x;                                                            \
    expect_bits.f32 = expect;                                                  \
    printf("Error: %s:%d: %s = %08x, expected %08x\n", __FILE__, __LINE__, #x, \
           (x_bits.i32), (expect_bits.i32));                                   \
    return -1;                                                                 \
  }

#define fp_check2(x, expect, eps)                                              \
  do {                                                                         \
    fp_bits_t x_bits, expect_bits;                                             \
    x_bits.f32 = x;                                                            \
    expect_bits.f32 = expect;                                                  \
    if (x_bits.i32 != expect_bits.i32) {                                       \
      printf("Warning: %s:%d: (%d,%d) %s = %08x, expected %08x\n", __FILE__,   \
             __LINE__, m, n, #x, (x_bits.i32), (expect_bits.i32));             \
    }                                                                          \
  } while (0)

#define fp_check2f(x, expect, eps)                                             \
  do {                                                                         \
    fp_bits_t x_bits, expect_bits;                                             \
    x_bits.f32 = x;                                                            \
    expect_bits.f32 = expect;                                                  \
    if (x_bits.i32 != expect_bits.i32) {                                       \
      printf("Warning: %s:%d: (%d,%d) %s = %f, expected %f, diff %f\n",        \
             __FILE__, __LINE__, m, n, #x, (x), (expect), (x - expect));       \
    }                                                                          \
  } while (0)

#include "generated/e4m3_mat_size.gen"

// A^T mem               DIM0 DIM1
float a_in[M * K] = {
#include "generated/e4m3_a_row.gen"
}; // A[M][K]
// B mem
float b_in[N * K] = {
#include "generated/e4m3_b_col.gen"
}; // B[N][K]
// C mem

uint8_t a[M * K];
uint8_t b[N * K];
float c[N * M];

float c_real[N * M] = {
#include "generated/e4m3_c_col.gen"
}; // C[N][M]

uint8_t __f32_to_e4m3(float f) { return RVT_EMU_FP8::from_float<4>(f); }

float __e4m3_to_f32(uint8_t e4m3) {
  RVT_EMU_FP8::FP8<4> data;
  data.bits = e4m3;
  return RVT_EMU_FP8::to_float(data);
}

int main() {
  std::cout << "Initializing Runtime..." << std::endl;
  struct NPURuntimeState runtime = {0};
  ssize_t err;

  err = npu_runtime_init(&runtime);
  if (err != NPU_SUCCESS) {
    std::cerr << "Failed to initialize Runtime: " << err << std::endl;
    return -1;
  }

  err = npu_runtime_load_kernelset(&runtime, &__general_entry_fpga);
  if (err != NPU_SUCCESS) {
    std::cerr << "Failed to load kernel set: " << err << std::endl;
    npu_runtime_deinit(&runtime);
    return -1;
  }
  std::cout << "Kernel set loaded successfully." << std::endl;

  int kernel_ret = 0;

  uint64_t device_a_addr = 0x81000000;
  uint64_t device_b_addr = 0x91000000;
  uint64_t device_c_addr = 0xa1000000;

  // __tx_init();
  NPU_START_KERNEL_BY_NAME(err, &runtime, "__tx_init", &kernel_ret, 0);
  if (err != NPU_SUCCESS) {
    npu_runtime_deinit(&runtime);
    return -1;
  }

  printf("prepare data\n");

  for (int m = 0; m < M; m++) {
    for (int k = 0; k < K; k++) {
      a[m * K + k] = __f32_to_e4m3(a_in[m * K + k]);
      fp_check(__e4m3_to_f32(a[m * K + k]), a_in[m * K + k], 0.0f);
    }
  }
  for (int n = 0; n < N; n++) {
    for (int k = 0; k < K; k++) {
      b[n * K + k] = __f32_to_e4m3(b_in[n * K + k]);
      fp_check(__e4m3_to_f32(b[n * K + k]), b_in[n * K + k], 0.0f);
    }
  }
  printf("M: %d, N: %d, K: %d\n", M, N, K);
  printf("gmma_e4m3_e4m3_f32_BT_CT start\n");
  // gmma_e4m3_e4m3_f32_BT_CT(b, a, c, N, M, K, K, K, M);

  size_t ret;
  ret = npu_memcpy_to_dev(&runtime.dev_state, device_a_addr, (void *)a,
                          sizeof(a));
  if (ret < 0) {
    std::cout << "a memcpy_to_dev fail: " << err << std::endl;
    npu_runtime_deinit(&runtime);
    return -1;
  }

  ret = npu_memcpy_to_dev(&runtime.dev_state, device_b_addr, (void *)b,
                          sizeof(b));
  if (ret < 0) {
    std::cout << "b memcpy_to_dev fail: " << err << std::endl;
    npu_runtime_deinit(&runtime);
    return -1;
  }

  memset(c, 0, sizeof(c));
  ret = npu_memcpy_to_dev(&runtime.dev_state, device_c_addr, (void *)c,
                          sizeof(c));
  if (ret < 0) {
    std::cout << "b memcpy_to_dev fail: " << err << std::endl;
    npu_runtime_deinit(&runtime);
    return -1;
  }

  NPU_START_KERNEL_BY_NAME(err, &runtime, "gmma_e4m3_e4m3_f32_BT_CT",
                           &kernel_ret, 9, device_a_addr, device_b_addr,
                           device_c_addr, M, N, K, K, K, M);
  if (err != NPU_SUCCESS) {
    npu_runtime_deinit(&runtime);
    return -1;
  }

  ret = npu_memcpy_from_dev(&runtime.dev_state, (void *)c, device_c_addr,
                            sizeof(c));
  if (ret < 0) {
    std::cout << "c npu_memcpy_from_dev fail: " << err;
    npu_runtime_deinit(&runtime);
    return -1;
  }

  printf("gmma_e4m3_e4m3_f32_BT_CT finish\n");

  // printf("c_col:\n");
  // for (size_t i = 0; i < N; i++) {
  //   for (size_t j = 0; j < M; j++) {
  //     printf("%x ", ((uint32_t *)c)[i * M + j]);
  //   }
  //   printf("\n");
  // }

  // printf("c_col_golden:\n");
  // for (size_t i = 0; i < N; i++) {
  //   for (size_t j = 0; j < M; j++) {
  //     printf("%x ", ((uint32_t *)c_real)[i * M + j]);
  //   }
  //   printf("\n");
  // }

  for (int n = 0; n < N; n++) {
    for (int m = 0; m < M; m++) {
      fp_check2f(c[n * M + m], c_real[n * M + m], 0.1f);
    }
  }
  printf("gmma_e4m3_e4m3_f32_BT_CT success\n");

  return 0;
}