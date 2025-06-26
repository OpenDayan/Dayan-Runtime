#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>
#include <math.h>

#include "NPUDriver.h"
#include "NPURuntime.h"
#include "npu_instruction.hpp"


#define check(x, expect)                                                                                     \
    if (isnan(x) || (x) != (expect))                                                                                     \
    {                                                                                                        \
        printf("Error: %s:%d: (%d,%d) %s = %d, expected %d\n", __FILE__, __LINE__, m, n, #x, (x), (expect)); \
        exit(-1);                                                                                      \
    }

#define fp_check(x, expect, eps)                                                               \
    if (isnan(x) || (x) < (expect) - (eps) || (x) > (expect) + (eps))                                      \
    {                                                                                          \
        printf("Error: %s:%d: %s = %f, expected %f\n", __FILE__, __LINE__, #x, (x), (expect)); \
        exit(-1);                                                                             \
    }

#define fp_checkr(m, n, x, expect, eps)                                                                      \
    if (isnan(x) || (x) < (expect) - (eps) || (x) > (expect) + (eps))                                                    \
    {                                                                                                        \
        printf("Error: %s:%d: (%d,%d) %s = %x, expected %x\n", __FILE__, __LINE__, m, n, #x, *(int*)&(x), *(int*)&(expect)); \
    }

const int M = 2;
const int K = 3;
const int N = 4;

float a_in[M * K] = {
    1.0f, 2.0f, 3.0f,
    4.0f, 5.0f, 6.0f
};

float b_in[K * N] = {
    1.0f,  2.0f,  3.0f,  4.0f,
    5.0f,  6.0f,  7.0f,  8.0f,
    9.0f, 10.0f, 11.0f, 12.0f
};

float c[M * N];

float c_real[M * N] = {
    // C = A * B
    38.0f,  44.0f,  50.0f,  56.0f, 
    83.0f,  98.0f, 113.0f, 128.0f
};


extern struct NPUDeviceKernelSetDesc __general_entry_fpga;

static struct NPURuntimeState runtime;
static struct NPUKernelDesc *kernel_desc;

static uint64_t src_0_addr = 0x81000000;
static uint64_t src_1_addr = 0x82000000;
static uint64_t dst_0_addr = 0x83000000;

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
  err = npu_runtime_get_kernel_desc(&runtime, "__tx_init", &tx_init_kernel_desc);
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
  err = npu_runtime_get_kernel_desc(&runtime, "matmul_fp32_fp32_fp32", &kernel_desc);
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
  ret = npu_memcpy_to_dev(&runtime.dev_state, src_0_addr, (void *)a_in, M  * K * sizeof(float));
  if (ret < 0)
    printf("[ERROR] Failed to write src0 to device at 0x%lx: err=%ld\n", src_0_addr, ret);
  printf("[INFO] Copied src0(%ld Bytes) to device at 0x%lx\n", M  * K * sizeof(float), src_0_addr);

  ret = npu_memcpy_to_dev(&runtime.dev_state, src_1_addr, (void *)b_in, N * K * sizeof(float));
  if (ret < 0)
    printf("[ERROR] Failed to write src1 to device at 0x%lx: err=%ld\n", src_1_addr, ret);
  printf("[INFO] Copied src1(%ld Bytes) to device at 0x%lx\n", N * K * sizeof(float), src_1_addr);

  ret = npu_memcpy_to_dev(&runtime.dev_state, dst_0_addr, (void *)c, N * M  * sizeof(float));
  if (ret < 0)
    printf("[ERROR] Failed to write dst0 to device: at 0x%lx err=%ld\n", dst_0_addr, ret);
  printf("[INFO] Copied dst0(%ld Bytes) to device at 0x%lx\n", N * M * sizeof(float), dst_0_addr);
}

static void testrt_npu_memcpy_out() {
  printf("[INFO] Copy data from device ... \n");
  size_t ret = 0;
  ret = npu_memcpy_from_dev(&runtime.dev_state, (void *)c, dst_0_addr, N * M * sizeof(float));
  if (ret < 0)
    printf("[ERROR] Failed to read dst0 from device: at 0x%lx: err=%ld\n", dst_0_addr, ret);
  printf("[INFO] Copied dst0(%ld Bytes) from device at 0x%lx\n", N * M * sizeof(float), dst_0_addr);
}

void check_result() {
  printf("[INFO] Check result ...\n");
  for (int n = 0; n < N; n++)
  {
      for (int m = 0; m < M; m++)
      {
          fp_checkr(m, n, c[n * M + m], c_real[n * M + m], 0.1f);
      }
  }
  printf("[INFO] Result check successfully!\n");
}

int main(int argc, char** argv) {
  testrt_npu_init();
  testrt_npu_load_kernel_set();
  testrt_run_tx_init_kernel();

  testrt_npu_get_kernel_desc();

  printf("[INFO] M=%d, N=%d, K=%d\n", M, N, K);

  // Copy in
  testrt_npu_memcpy_in();

  int kernel_ret;
  ssize_t err = 0;
  printf("[INFO] Start kernel matmul_fp32_fp32_fp32 ... \n");
  err = npu_runtime_kernel_start(&runtime, kernel_desc, &kernel_ret, 6,
                                 src_0_addr, src_1_addr, dst_0_addr,
                                 M, N, K
                                 );
  printf("[INFO] Kernel ret=%d\n", kernel_ret);
  if (err != NPU_SUCCESS) {
    printf("[ERROR] Failed to start kernel : err=%ld\n", err);
    npu_runtime_deinit(&runtime);
    return -1;
  }
  printf("[INFO] Kernel matmul_fp32_fp32_fp32 run successfully.\n");

  // Copy out
  testrt_npu_memcpy_out();

  // End
  npu_runtime_deinit(&runtime);

  // Check result
  check_result();

  return 0;
}
