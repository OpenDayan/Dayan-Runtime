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



#define fp_check(x, expect, eps)                                               \
  do {                                                                         \
    float _x = (x);                                                            \
    float _expect = (float)(expect);                                           \
    if (isnan(_x) || (_x) < (_expect) - (eps) || (_x) > (_expect) + (eps)) {   \
      printf("Error: %s:%d: %s = %f, expected %f\n", __FILE__, __LINE__, #x,   \
             _x, _expect);                                                     \
      exit(-1);                                                                \
    }                                                                          \
  } while (0)

#define fp_checkr(m, n, x, expect, eps)                                        \
  if (isnan(x) || (x) < (expect) - (eps) || (x) > (expect) + (eps)) {          \
    printf("Error: %s:%d: (%d,%d) %s = %x, expected %x\n", __FILE__, __LINE__, \
           m, n, #x, *(int *)&(x), *(int *)&(expect));                         \
  }

#include "generated/elementwise_int_size.gen"

float c_fp32[N * 1] = {0};

int c_s32[N * 1] = {0};

float a_fp32_in[N * M] = {
#include "generated/elementwise_float_a_row.gen"
};

int a_s32_in[N * M] = {
#include "generated/elementwise_int_a_row.gen"
};

float c_s32_max_real[N * 1] = {
#include "generated/elementwise_int_c_max_row.gen"
};

float c_s32_min_real[N * 1] = {
#include "generated/elementwise_int_c_min_row.gen"
};

float c_s32_sum_real[N * 1] = {
#include "generated/elementwise_int_c_sum_row.gen"
};

float c_fp32_max_real[N * 1] = {
#include "generated/elementwise_float_c_max_row.gen"
};

float c_fp32_min_real[N * 1] = {
#include "generated/elementwise_float_c_min_row.gen"
};

float c_fp32_sum_real[N * 1] = {
#include "generated/elementwise_float_c_sum_row.gen"
};

extern struct NPUDeviceKernelSetDesc __general_entry_fpga;

static struct NPURuntimeState runtime;
static struct NPUKernelDesc *kernel_desc;

static uint64_t src_0_addr = 0x81000000;
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
  err = npu_runtime_load_kernelset(&runtime, nullptr);
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

static void testrt_npu_get_kernel_desc(const char *kernel_name) {
  ssize_t err = 0;
  err = npu_runtime_get_kernel_desc(&runtime, kernel_name, &kernel_desc);
  if (err != NPU_SUCCESS) {
    printf("Failed to get kernel description for '%s': err=%ld\n", kernel_name,
           err);
    npu_runtime_deinit(&runtime);
    exit(-1);
  }
  printf("[INFO] Loaded '%s' kernel desc successfully.\n", kernel_name);
}

static void testrt_npu_memcpy_in(void *input0, void *output) {
  printf("[INFO] Copy data to device ... \n");
  size_t ret = 0;
  ret = npu_memcpy_to_dev(&runtime.dev_state, src_0_addr, (void *)input0,
                          M * N * sizeof(int32_t));
  if (ret < 0)
    printf("[ERROR] Failed to write src0 to device at 0x%lx: err=%ld\n",
           src_0_addr, ret);
  printf("[INFO] Copied src0(%ld Bytes) to device at 0x%lx\n",
         M * N * sizeof(int32_t), src_0_addr);


  ret = npu_memcpy_to_dev(&runtime.dev_state, dst_0_addr, (void *)output,
                          1 * N * sizeof(int32_t));
  if (ret < 0)
    printf("[ERROR] Failed to write dst0 to device: at 0x%lx err=%ld\n",
           dst_0_addr, ret);
  printf("[INFO] Copied dst0(%ld Bytes) to device at 0x%lx\n",
         1 * N * sizeof(int32_t), dst_0_addr);
}

static void testrt_npu_memcpy_out(void *output) {
  printf("[INFO] Copy data from device ... \n");
  size_t ret = 0;
  ret = npu_memcpy_from_dev(&runtime.dev_state, (void *)output, dst_0_addr,
                            1 * N * sizeof(int32_t));
  if (ret < 0)
    printf("[ERROR] Failed to read dst0 from device: at 0x%lx: err=%ld\n",
           dst_0_addr, ret);
  printf("[INFO] Copied dst0(%ld Bytes) from device at 0x%lx\n",
         1 * N * sizeof(int32_t), dst_0_addr);
}

void check_fp32_result(float *output_real) {
  printf("[INFO] Check result ...\n");
  for (int n = 0; n < N; n++) {
    fp_check(c_fp32[n], output_real[n], 0);
  }
  printf("[INFO] Result check successfully!\n");
}

void check_s32_result(float *output_real) {
  printf("[INFO] Check result ...\n");
  for (int n = 0; n < N; n++) {
    fp_check(c_s32[n], output_real[n], 0);
  }
  printf("[INFO] Result check successfully!\n");
}


int run_kenrel(const char *kernel_name, void *input0, 
               void *output, int axis) {
  // Get kernel
  testrt_npu_get_kernel_desc(kernel_name);

  printf("[INFO] M=%d, N=%d\n", M, N);

  // Copy in
  testrt_npu_memcpy_in(input0, output);

  int kernel_ret;
  ssize_t err = 0;
  printf("[INFO] Start kernel %s ... \n", kernel_name);
  err = npu_runtime_kernel_start(&runtime, kernel_desc, &kernel_ret, 5,
                                 src_0_addr, dst_0_addr, M, N, axis);
  printf("[INFO] Kernel ret=%d\n", kernel_ret);
  if (err != NPU_SUCCESS) {
    printf("[ERROR] Failed to start kernel : err=%ld\n", err);
    npu_runtime_deinit(&runtime);
    return -1;
  }
  printf("[INFO] Kernel %s run successfully.\n", kernel_name);

  // Copy out
  testrt_npu_memcpy_out(output);
  return 0;
}

int main(int argc, char **argv) {
  testrt_npu_init();
  testrt_npu_load_kernel_set();
  testrt_run_tx_init_kernel();

  
  run_kenrel("tx_fp_reduce_max_fp32", a_fp32_in, c_fp32, 1);
  check_fp32_result(c_fp32_max_real);

  run_kenrel("tx_fp_reduce_min_fp32", a_fp32_in, c_fp32, 1);
  check_fp32_result(c_fp32_min_real);

  run_kenrel("tx_fp_reduce_add_fp32", a_fp32_in, c_fp32, 1);
  check_fp32_result(c_fp32_sum_real);

  run_kenrel("tx_int_reduce_max_s32", a_s32_in, c_s32, 1);
  check_s32_result(c_s32_max_real);

  run_kenrel("tx_int_reduce_min_s32", a_s32_in, c_s32, 1);
  check_s32_result(c_s32_min_real);

  run_kenrel("tx_int_reduce_add_s32", a_s32_in, c_s32, 1);
  check_s32_result(c_s32_sum_real);

  // End
  npu_runtime_deinit(&runtime);

  return 0;
}
