#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>

#include "NPUDriver.h"
#include "NPURuntime.h"
#include "npu_instruction.hpp"


#define check(x, expect)                                                                                     \
    if ((x) != (expect))                                                                                     \
    {                                                                                                        \
        printf("Error: %s:%d: (%d,%d) %s = %d, expected %d\n", __FILE__, __LINE__, m, n, #x, (x), (expect)); \
        exit(-1);                                                                                      \
    }

#define fp_check(x, expect, eps)                                                               \
    if ((x) < (expect) - (eps) || (x) > (expect) + (eps))                                      \
    {                                                                                          \
        printf("Error: %s:%d: %s = %f, expected %f\n", __FILE__, __LINE__, #x, (x), (expect)); \
        exit(-1);                                                                             \
    }

#define fp_checkr(m, n, x, expect, eps)                                                                      \
    if ((x) < (expect) - (eps) || (x) > (expect) + (eps))                                                    \
    {                                                                                                        \
        printf("Error: %s:%d: (%d,%d) %s = %f, expected %f\n", __FILE__, __LINE__, m, n, #x, (x), (expect)); \
        exit(-1);                                                                                            \
    }

#include "generated/e4m3q_mat_size.gen"

// A^T mem               DIM0 DIM1
float a_in[M * K] = {
#include "generated/e4m3q_a_row.gen"
}; // A[M][K]
// B mem
float b_in[N * K] = {
#include "generated/e4m3q_b_col.gen"
}; // B[N][K]
// C mem

// A^T mem               DIM0 DIM1
float as_in[M * K / QK] = {
#include "generated/e4m3q_as_row.gen"
}; // A[M][K]
// B mem
float bs_in[N * K / QK] = {
#include "generated/e4m3q_bs_col.gen"
}; // B[N][K]
// C mem

struct blk_e4m3q
{
    float scale;
    uint8_t v[QK];
};

struct blk_e4m3q a[M * K / QK];
struct blk_e4m3q b[N * K / QK];

float c[N * M];

float c_real[N * M] = {
#include "generated/e4m3q_c_col.gen"
}; // C[N][M]

uint8_t __f32_to_e4m3(float f)
{
    return RVT_EMU_FP8::from_float<4>(f);
}

float __e4m3_to_f32(uint8_t e4m3)
{
    RVT_EMU_FP8::FP8<4> data;
    data.bits = e4m3;
    return RVT_EMU_FP8::to_float(data);
}

extern struct NPUDeviceKernelSetDesc __general_entry_fpga;

void trans_f32_to_e4m3() {
  for (int m = 0; m < M; m++)
  {
      for (int n = 0; n < N; n++)
      {
          c[m * N + n] = 0.0;
      }
  }
  for (int m = 0; m < M; m++)
  {
      for (int k = 0; k < (K / QK); k++)
      {
          a[m * K / QK + k].scale = as_in[m * K / QK + k];
      }
  }
  for (int n = 0; n < N; n++)
  {
      for (int k = 0; k < (K / QK); k++)
      {
          b[n * K / QK + k].scale = bs_in[n * K / QK + k];
      }
  }
  for (int n = 0; n < N; n++)
  {
      for (int k = 0; k < K; k++)
      {
          b[(n * K + k) / QK].v[k % QK] = __f32_to_e4m3(b_in[n * K + k]);
          fp_check(__e4m3_to_f32(b[(n * K + k) / QK].v[k % QK]), b_in[n * K + k], 0.0f);
      }
  }
  for (int m = 0; m < M; m++)
  {
      for (int k = 0; k < K; k++)
      {
          a[(m * K + k) / QK].v[k % QK] = __f32_to_e4m3(a_in[m * K + k]);
          fp_check(__e4m3_to_f32(a[(m * K + k) / QK].v[k % QK]), a_in[m * K + k], 0.0f);
      }
  }
  fp_check(a[1].scale, ((float *)a)[1 * (QK / sizeof(float) + 1)], 0.0f);
}

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
    printf("[ERROR] Failed to initialize Runtime: err=%d\n", err);
    exit(-1);
  }
  printf("[INFO] Runtime initialized successfully.\n");
}

static void testrt_npu_load_kernel_set() {
  ssize_t err = 0;
  err = npu_runtime_load_kernelset(&runtime, &__general_entry_fpga);
  if (err != NPU_SUCCESS) {
    printf("Failed to load kernel set: err=%d\n", err);
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
    printf("Failed to get kernel description for __tx_init: err=%d\n", err);
    npu_runtime_deinit(&runtime);
    exit(-1);
  }
  printf("[INFO] Loaded __tx_init kernel desc successfully.\n");
  int kernel_ret;
  printf("[INFO] Start kernel __tx_init ... \n");
  err = npu_runtime_kernel_start(&runtime, tx_init_kernel_desc, &kernel_ret, 0);
  printf("[INFO] Kernel ret=%ld\n", kernel_ret);
  if (err != NPU_SUCCESS) {
    printf("[ERROR] Failed to start kernel : err=%d\n", err);
    npu_runtime_deinit(&runtime);
    exit(-1);
  }
  printf("[INFO] Kernel __tx_init run successfully.\n");
}


static void testrt_npu_get_kernel_desc() {
  ssize_t err = 0;
  err = npu_runtime_get_kernel_desc(&runtime, "gmma_e4m3q_fp32_AT_QK32", &kernel_desc);
  if (err != NPU_SUCCESS) {
    printf("Failed to get kernel description: err=%d\n", err);
    npu_runtime_deinit(&runtime);
    exit(-1);
  }
  printf("[INFO] Loaded gmma_e4m3q_fp32_AT_QK32 kernel desc successfully.\n");
}


static void testrt_npu_memcpy_in(uint64_t lda, uint64_t ldb, uint64_t ldc) {
  printf("[INFO] Copy data to device ... \n");
  size_t ret = 0;
  ret = npu_memcpy_to_dev(&runtime.dev_state, src_0_addr, (void *)a, M * lda * sizeof(uint8_t));
  if (ret < 0)
    printf("[ERROR] Failed to write src0 to device at 0x%lx: err=%d\n", src_0_addr, ret);
  printf("[INFO] Copied src0(%ld Bytes) to device at 0x%lx\n", M * lda, src_0_addr);

  ret = npu_memcpy_to_dev(&runtime.dev_state, src_1_addr, (void *)b, N * ldb * sizeof(uint8_t));
  if (ret < 0)
    printf("[ERROR] Failed to write src1 to device at 0x%lx: err=%d\n", src_1_addr, ret);
  printf("[INFO] Copied src1(%ld Bytes) to device at 0x%lx\n", N * ldb, src_1_addr);

  ret = npu_memcpy_to_dev(&runtime.dev_state, dst_0_addr, (void *)c, M * ldc * sizeof(float));
  if (ret < 0)
    printf("[ERROR] Failed to write dst0 to device: at 0x%lx err=%d\n", dst_0_addr, ret);
  printf("[INFO] Copied dst0(%ld Bytes) to device at 0x%lx\n", M * ldc * sizeof(float), dst_0_addr);
}

static void testrt_npu_memcpy_out(uint64_t ldc) {
  printf("[INFO] Copy data from device ... \n");
  size_t ret = 0;
  ret = npu_memcpy_from_dev(&runtime.dev_state, (void *)c, dst_0_addr, M * ldc * sizeof(float));
  if (ret < 0)
    printf("[ERROR] Failed to read dst0 from device: at 0x%lx: err=%d\n", dst_0_addr, ret);
  printf("[INFO] Copied dst0(%ld Bytes) from device at 0x%lx\n", M * ldc * sizeof(float), dst_0_addr);
}

void check_result() {
  printf("[INFO] Check result ...\n");
  for (int n = 0; n < N; n++)
  {
      for (int m = 0; m < M; m++)
      {
          fp_checkr(m, n, c[n * M + m], c_real[n * M + m], 0.01f);
      }
  }
  printf("[INFO] Result check successfully!\n");
}

int main(int argc, char** argv) {
  trans_f32_to_e4m3();
  testrt_npu_init();
  testrt_npu_load_kernel_set();
  testrt_run_tx_init_kernel();

  testrt_npu_get_kernel_desc();

  uint64_t lda = K + K * 4 / QK;
  uint64_t ldb = K + K * 4 / QK;
  uint64_t ldc = M;
  printf("[INFO] M=%ld, N=%ld, K=%ld\n", M, N, K);
  printf("[INFO] lda=%ld, ldb=%ld, ldc=%ld\n", lda, ldb, ldc);

  // Copy in
  testrt_npu_memcpy_in(lda, ldb, ldc);

  int kernel_ret;
  ssize_t err = 0;
  printf("[INFO] Start kernel gmma_e4m3q_fp32_AT_QK32 ... \n");
  err = npu_runtime_kernel_start(&runtime, kernel_desc, &kernel_ret, 9,
                                 src_0_addr, src_1_addr, dst_0_addr,
                                 (uint64_t)M, (uint64_t)N, (uint64_t)K,
                                 lda, ldb, ldc);
  printf("[INFO] Kernel ret=%ld\n", kernel_ret);
  if (err != NPU_SUCCESS) {
    printf("[ERROR] Failed to start kernel : err=%d\n", err);
    npu_runtime_deinit(&runtime);
    return -1;
  }
  printf("[INFO] Kernel gmma_e4m3q_fp32_AT_QK32 run successfully.\n");

  // Copy out
  testrt_npu_memcpy_out(ldc);

  // End
  npu_runtime_deinit(&runtime);

  // Check result
  check_result();

  return 0;
}
