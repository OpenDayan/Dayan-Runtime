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

#define check(x, expect)                                                       \
  if (isnan(x) || (x) != (expect)) {                                           \
    printf("Error: %s:%d: (%d,%d) %s = %d, expected %d\n", __FILE__, __LINE__, \
           m, n, #x, (x), (expect));                                           \
    exit(-1);                                                                  \
  }

#define fp_checkr(m, n, x, expect, eps)                                        \
  if (isnan(x) || (x) < (expect) - (eps) || (x) > (expect) + (eps)) {          \
    printf("Error: %s:%d: (%d,%d) %s = %f, expected %f\n", __FILE__, __LINE__, \
           m, n, #x, (x), (expect));                                           \
    exit(-1);                                                                  \
  }

const int input_dim0 = 4;
const int table_rows = 8;
const int log2Length = 3;
const int width = 2;

int a[input_dim0] = {0, 1, 2, 4};

int lut_table[table_rows * width] = {1, 2,  3,  4,  5,  6,  7,  8,
                                     9, 10, 11, 12, 13, 14, 15, 16};

int c[input_dim0 * width] = {0};

int c_real[input_dim0 * width] = {
    1, 2, 3, 4, 5, 6, 9, 10};

extern struct NPUDeviceKernelSetDesc __general_entry_fpga;

static struct NPURuntimeState runtime;
static struct NPUKernelDesc *kernel_desc;

static uint64_t src_addr = 0x81000000;
static uint64_t lut_addr = 0x82000000;
static uint64_t dst_addr = 0x83000000;
static uint64_t tcsr_addr = 0x84000000;
static TCFG tcsr[3];

void init_tcfg(const long dims[3][4], const long strides[3][4]) {
  for (int i = 0; i < 3; ++i) {
    tcsr[i] = {.dim0 = dims[i][0],
               .stride0 = strides[i][0],
               .dim1 = dims[i][1],
               .stride1 = strides[i][1],
               .dim2 = dims[i][2],
               .stride2 = strides[i][2],
               .dim3 = dims[i][3],
               .stride3 = strides[i][3]};
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

void check_result(int *c) {
  printf("[INFO] Check result ...\n");
  for (int n = 0; n < input_dim0; n++) {
    for (int m = 0; m < width; m++) {
      check(c[n * width + m], c_real[n * width + m]);
    }
  }
  printf("[INFO] Result check successfully!\n");
}

static void testrt_npu_memcpy_in(void *input0, void *input1, void *output) {
  printf("[INFO] Copy data to device ... \n");
  size_t ret = 0;
  ret = npu_memcpy_to_dev(&runtime.dev_state, src_addr, (void *)input0,
                          input_dim0 * sizeof(int32_t));
  if (ret < 0)
    printf("[ERROR] Failed to write src0 to device at 0x%lx: err=%ld\n",
           src_addr, ret);
  printf("[INFO] Copied src0(%ld Bytes) to device at 0x%lx\n",
         input_dim0 * sizeof(int32_t), src_addr);

  ret = npu_memcpy_to_dev(&runtime.dev_state, lut_addr, (void *)input1,
                          table_rows * width * sizeof(int32_t));
  if (ret < 0)
    printf("[ERROR] Failed to write src1 to device at 0x%lx: err=%ld\n",
           lut_addr, ret);
  printf("[INFO] Copied src0(%ld Bytes) to device at 0x%lx\n",
        table_rows * width * sizeof(int32_t), lut_addr);

  ret = npu_memcpy_to_dev(&runtime.dev_state, dst_addr, (void *)output,
                          input_dim0 * width * sizeof(int32_t));
  if (ret < 0)
    printf("[ERROR] Failed to write dst0 to device: at 0x%lx err=%ld\n",
           dst_addr, ret);
  printf("[INFO] Copied dst0(%ld Bytes) to device at 0x%lx\n",
         input_dim0 * width * sizeof(int32_t), dst_addr);
}

static void testrt_npu_memcpy_out(void *output) {
  printf("[INFO] Copy data from device ... \n");
  size_t ret = 0;
  ret = npu_memcpy_from_dev(&runtime.dev_state, (void *)output, dst_addr,
                            input_dim0 * width * sizeof(int32_t));
  if (ret < 0)
    printf("[ERROR] Failed to read dst0 from device: at 0x%lx: err=%ld\n",
           dst_addr, ret);
  printf("[INFO] Copied dst0(%ld Bytes) from device at 0x%lx\n",
        input_dim0 * width * sizeof(int32_t), dst_addr);
}

void testrt_npu_config_tcsrs(const long dims[3][4], const long strides[3][4]) {
  int err = 0;

  init_tcfg(dims, strides);
  err = npu_memcpy_to_dev(&runtime.dev_state, tcsr_addr, (void *)tcsr,
                          sizeof(tcsr));
  printf("[INFO] Copy tcsr to device with size:%ld and ret:%d\n", sizeof(tcsr),
         err);
}

int run_kenrel(const char *kernel_name, void *input0, void *input1,
               void *output) {
  // Get kernel
  testrt_npu_get_kernel_desc(kernel_name);

  // Copy in
  testrt_npu_memcpy_in(input0, input1, output);
  const long dims[3][4] = {{input_dim0, 1, 1, 1}, {width, table_rows, 1, 1}, {width, input_dim0, 1, 1}};
  const long strides[3][4] = {{1, input_dim0, input_dim0,input_dim0}, {1, width, width * table_rows, width * table_rows}, {1, width, width*input_dim0, width*input_dim0}};
  testrt_npu_config_tcsrs(dims, strides);

  int kernel_ret;
  ssize_t err = 0;
  printf("[INFO] Start kernel %s ... \n", kernel_name);
  err = npu_runtime_kernel_start(&runtime, kernel_desc, &kernel_ret, 6,
                                 src_addr, lut_addr, dst_addr, tcsr_addr, width, log2Length);
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

  run_kenrel("tx_lut_int_s32", a, lut_table, c);
  check_result(c);
  // End
  npu_runtime_deinit(&runtime);

  return 0;
}
