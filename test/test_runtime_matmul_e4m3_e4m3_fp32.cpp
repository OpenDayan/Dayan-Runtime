#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>

#include "NPUDriver.h"
#include "NPURuntime.h"
#include "npu_instruction.hpp"

typedef uint8_t fp8;

#define check(x, expect)                                                       \
  if (isnan(x) || (x) != (expect)) {                                           \
    printf("Error: %s:%d: (%d,%d) %s = %d, expected %d\n", __FILE__, __LINE__, \
           m, n, #x, (x), (expect));                                           \
    exit(-1);                                                                  \
  }

#define fp_check(x, expect, eps)                                               \
  if (isnan(x) || (x) < (expect) - (eps) || (x) > (expect) + (eps)) {          \
    printf("Error: %s:%d: %s = %f, expected %f\n", __FILE__, __LINE__, #x,     \
           (x), (expect));                                                     \
    exit(-1);                                                                  \
  }

#define fp_checkr(m, n, x, expect, eps)                                        \
  if (isnan(x) || (x) < (expect) - (eps) || (x) > (expect) + (eps)) {          \
    printf("Error: %s:%d: (%d,%d) %s = %x, expected %x\n", __FILE__, __LINE__, \
           m, n, #x, *(int *)&(x), *(int *)&(expect));                         \
  }

#include "../npu-kernels/testdata_gen/generated/e4m3_mat_size.gen"

float a[M * K] = {
#include "../npu-kernels/testdata_gen/generated/e4m3_a_row.gen"
}; // A[M][K]

float b[K * N] = {
#include "../npu-kernels/testdata_gen/generated/e4m3_b_row.gen"
}; // B[K][N]

float c_real[M * N] = {
#include "../npu-kernels/testdata_gen/generated/e4m3_c_row.gen"
}; // C[N][M]

fp8 a_e4m3[M * K];
fp8 b_e4m3[K * N];

float c[M * N];

extern struct NPUDeviceKernelSetDesc __general_entry_fpga;

static struct NPURuntimeState runtime;
static struct NPUKernelDesc *kernel_desc;

static uint64_t src_0_addr = 0x81000000;
static uint64_t src_1_addr = 0x82000000;
static uint64_t dst_0_addr = 0x83000000;

static fp8 __f32_to_e4m3(float f) { return RVT_EMU_FP8::from_float<4>(f); }

static float __e4m3_to_f32(fp8 e4m3) {
  RVT_EMU_FP8::FP8<4> data;
  data.bits = e4m3;
  return RVT_EMU_FP8::to_float(data);
}

static void trans_f32_to_e4m3() {
  for (int m = 0; m < M; m++) {
    for (int k = 0; k < K; k++) {
      a_e4m3[m * K + k] = __f32_to_e4m3(a[m * K + k]);
      fp_check(__e4m3_to_f32(a_e4m3[m * K + k]), a[m * K + k], 0.0f);
    }
  }
  for (int k = 0; k < K; k++) {
    for (int n = 0; n < N; n++) {
      b_e4m3[k * N + n] = __f32_to_e4m3(b[k * N + n]);
      fp_check(__e4m3_to_f32(b_e4m3[k * N + n]), b[k * N + n], 0.0f);
    }
  }
}

static void testrt_npu_init() {
  ssize_t err = 0;
  printf("[INFO] Initializing Runtime..\n");
  err = npu_runtime_init(&runtime);
  if (err != NPU_SUCCESS) {
    printf("[ERROR] Failed to initialize Runtime: err=%ld\n", err);
    exit(-1);
  }
  printf("[INFO] Runtime initialized successfully.\n");
}

static void testrt_npu_load_kernel_set() {
  ssize_t err = 0;
  err = npu_runtime_load_kernelset(&runtime, &__general_entry_fpga);
  if (err != NPU_SUCCESS) {
    printf("Failed to load kernel set: err=%ld\n", err);
    npu_runtime_deinit(&runtime);
    exit(-1);
  }
  printf("[INFO] Kernel set loaded successfully.\n");
}

static void testrt_run_tx_init_kernel() {
  ssize_t err = 0;
  struct NPUKernelDesc *tx_init_kernel_desc;
  err =
      npu_runtime_get_kernel_desc(&runtime, "__tx_init", &tx_init_kernel_desc);
  if (err != NPU_SUCCESS) {
    printf("Failed to get kernel description for __tx_init: err=%ld\n", err);
    npu_runtime_deinit(&runtime);
    exit(-1);
  }
  printf("[INFO] Loaded __tx_init kernel desc successfully.\n");
  int kernel_ret;
  printf("[INFO] Start kernel __tx_init ... \n");
  err = npu_runtime_kernel_start(&runtime, tx_init_kernel_desc, &kernel_ret, 0);
  printf("[INFO] Kernel ret=%d\n", kernel_ret);
  if (err != NPU_SUCCESS) {
    printf("[ERROR] Failed to start kernel : err=%ld\n", err);
    npu_runtime_deinit(&runtime);
    exit(-1);
  }
  printf("[INFO] Kernel __tx_init run successfully.\n");
}

static void testrt_npu_get_kernel_desc() {
  ssize_t err = 0;
  err = npu_runtime_get_kernel_desc(&runtime, "matmul_e4m3_e4m3_fp32",
                                    &kernel_desc);
  if (err != NPU_SUCCESS) {
    printf("Failed to get kernel description: err=%ld\n", err);
    npu_runtime_deinit(&runtime);
    exit(-1);
  }
  printf("[INFO] Loaded matmul_fp32_fp32_fp32 kernel desc successfully.\n");
}

static void testrt_npu_memcpy_in() {
  printf("[INFO] Copy data to device ... \n");
  size_t ret = 0;
  ret = npu_memcpy_to_dev(&runtime.dev_state, src_0_addr, (void *)a_e4m3,
                          M * K * sizeof(fp8));
  if (ret < 0)
    printf("[ERROR] Failed to write src0 to device at 0x%lx: err=%ld\n",
           src_0_addr, ret);
  printf("[INFO] Copied src0(%ld Bytes) to device at 0x%lx\n",
         M * K * sizeof(fp8), src_0_addr);

  ret = npu_memcpy_to_dev(&runtime.dev_state, src_1_addr, (void *)b_e4m3,
                          N * K * sizeof(fp8));
  if (ret < 0)
    printf("[ERROR] Failed to write src1 to device at 0x%lx: err=%ld\n",
           src_1_addr, ret);
  printf("[INFO] Copied src1(%ld Bytes) to device at 0x%lx\n",
         N * K * sizeof(fp8), src_1_addr);

  ret = npu_memcpy_to_dev(&runtime.dev_state, dst_0_addr, (void *)c,
                          N * M * sizeof(float));
  if (ret < 0)
    printf("[ERROR] Failed to write dst0 to device: at 0x%lx err=%ld\n",
           dst_0_addr, ret);
  printf("[INFO] Copied dst0(%ld Bytes) to device at 0x%lx\n",
         N * M * sizeof(float), dst_0_addr);
}

static void testrt_npu_memcpy_out() {
  printf("[INFO] Copy data from device ... \n");
  size_t ret = 0;
  ret = npu_memcpy_from_dev(&runtime.dev_state, (void *)c, dst_0_addr,
                            N * M * sizeof(float));
  if (ret < 0)
    printf("[ERROR] Failed to read dst0 from device: at 0x%lx: err=%ld\n",
           dst_0_addr, ret);
  printf("[INFO] Copied dst0(%ld Bytes) from device at 0x%lx\n",
         N * M * sizeof(float), dst_0_addr);
}

void check_result() {
  printf("[INFO] Check result ...\n");
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      fp_checkr(m, n, c[m * N + n], c_real[m * N + n], 0.1f);
    }
  }
  printf("[INFO] Result check successfully!\n");
}

int main(int argc, char **argv) {
  testrt_npu_init();
  testrt_npu_load_kernel_set();
  testrt_run_tx_init_kernel();

  testrt_npu_get_kernel_desc();

  printf("[INFO] M=%d, N=%d, K=%d\n", M, N, K);
  trans_f32_to_e4m3();

  // Copy in
  testrt_npu_memcpy_in();

  int kernel_ret;
  ssize_t err = 0;
  printf("[INFO] Start kernel matmul_e4m3_e4m3_fp32 ... \n");
  err = npu_runtime_kernel_start(&runtime, kernel_desc, &kernel_ret, 6,
                                 src_0_addr, src_1_addr, dst_0_addr, M, N, K);
  printf("[INFO] Kernel ret=%d\n", kernel_ret);
  if (err != NPU_SUCCESS) {
    printf("[ERROR] Failed to start kernel : err=%ld\n", err);
    npu_runtime_deinit(&runtime);
    return -1;
  }
  printf("[INFO] Kernel matmul_e4m3_e4m3_fp32 run successfully.\n");

  // Copy out
  testrt_npu_memcpy_out();

  // End
  npu_runtime_deinit(&runtime);

  // Check result
  check_result();

  return 0;
}
