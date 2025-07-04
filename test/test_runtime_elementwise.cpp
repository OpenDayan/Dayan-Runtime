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
  if ((x) != (expect)) {                                                       \
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

#include "../npu-kernels/testdata_gen/generated/elementwise_int_size.gen"

int32_t a_in[M * N] = {
#include "../npu-kernels/testdata_gen/generated/elementwise_int_a_row.gen"
};

int32_t b_in[M * N] = {
#include "../npu-kernels/testdata_gen/generated/elementwise_int_b_row.gen"
};

int32_t shift_in[M * N] = {
#include "../npu-kernels/testdata_gen/generated/elementwise_int_shift_row.gen"
};

int32_t c_add_real[M * N] = {
#include "../npu-kernels/testdata_gen/generated/elementwise_int_c_add_row.gen"
};

int32_t c_sub_real[M * N] = {
#include "../npu-kernels/testdata_gen/generated/elementwise_int_c_sub_row.gen"
};

int32_t c_mul_real[M * N] = {
#include "../npu-kernels/testdata_gen/generated/elementwise_int_c_mul_row.gen"
};

int32_t c_srl_real[M * N] = {
#include "../npu-kernels/testdata_gen/generated/elementwise_int_c_srl_row.gen"
};

int32_t c_sra_real[M * N] = {
#include "../npu-kernels/testdata_gen/generated/elementwise_int_c_sra_row.gen"
};

int32_t c_sll_real[M * N] = {
#include "../npu-kernels/testdata_gen/generated/elementwise_int_c_sll_row.gen"
};

int32_t c_and_real[M * N] = {
#include "../npu-kernels/testdata_gen/generated/elementwise_int_c_and_row.gen"
};

int32_t c_or_real[M * N] = {
#include "../npu-kernels/testdata_gen/generated/elementwise_int_c_or_row.gen"
};

int32_t c_xor_real[M * N] = {
#include "../npu-kernels/testdata_gen/generated/elementwise_int_c_xor_row.gen"
};

float a_fp32_in[M * N] = {
#include "../npu-kernels/testdata_gen/generated/elementwise_float_a_row.gen"
};

float b_fp32_in[M * N] = {
#include "../npu-kernels/testdata_gen/generated/elementwise_float_b_row.gen"
};

float c_fp32_add_real[M * N] = {
#include "../npu-kernels/testdata_gen/generated/elementwise_float_c_add_row.gen"
};

float c_fp32_sub_real[M * N] = {
#include "../npu-kernels/testdata_gen/generated/elementwise_float_c_sub_row.gen"
};

float c_fp32_mul_real[M * N] = {
#include "../npu-kernels/testdata_gen/generated/elementwise_float_c_mul_row.gen"
};

// broadcast
int32_t b_row[1 * N] = {
#include "../npu-kernels/testdata_gen/generated/broadcast_b_row.gen"
};
int32_t b_col[M * 1] = {
#include "../npu-kernels/testdata_gen/generated/broadcast_b_col.gen"
};

int32_t scalar[1] = {
#include "../npu-kernels/testdata_gen/generated/broadcast_scalar.gen"
};

int32_t c_add_row_bcast_real[M * N] = {
#include "../npu-kernels/testdata_gen/generated/broadcast_c_add_row_bcast.gen"
};

int32_t c_mul_row_bcast_real[M * N] = {
#include "../npu-kernels/testdata_gen/generated/broadcast_c_mul_row_bcast.gen"
};

int32_t c_sub_col_bcast_real[M * N] = {
#include "../npu-kernels/testdata_gen/generated/broadcast_c_sub_col_bcast.gen"
};

int32_t c_or_col_bcast_real[M * N] = {
#include "../npu-kernels/testdata_gen/generated/broadcast_c_or_col_bcast.gen"
};

int32_t c_mul_scalar_bcast_real[M * N] = {
#include "../npu-kernels/testdata_gen/generated/broadcast_c_mul_scalar_bcast.gen"
};

int32_t c_xor_scalar_bcast_real[M * N] = {
#include "../npu-kernels/testdata_gen/generated/broadcast_c_xor_scalar_bcast.gen"
};

int32_t c[M * N] = {0};
float c_fp32[M * N] = {0};

extern struct NPUDeviceKernelSetDesc __general_entry_fpga;

static struct NPURuntimeState runtime;
static struct NPUKernelDesc *kernel_desc;

static uint64_t src_0_addr = 0x81000000;
static uint64_t src_1_addr = 0x82000000;
static uint64_t dst_0_addr = 0x83000000;
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

static void testrt_npu_memcpy_in(void *input0, void *input1, void *output, const long dims[3][4]) {
  printf("[INFO] Copy data to device ... \n");
  size_t ret = 0;
  size_t input0_size = dims[0][0] * dims[0][1] * dims[0][2] * dims[0][3];
  ret = npu_memcpy_to_dev(&runtime.dev_state, src_0_addr, (void *)input0,
                          input0_size * sizeof(int32_t));
  if (ret < 0)
    printf("[ERROR] Failed to write src0 to device at 0x%lx: err=%ld\n",
           src_0_addr, ret);
  printf("[INFO] Copied src0(%ld Bytes) to device at 0x%lx\n",
        input0_size * sizeof(int32_t), src_0_addr);

  size_t input1_size = dims[1][0] * dims[1][1] * dims[1][2] * dims[1][3];
  ret = npu_memcpy_to_dev(&runtime.dev_state, src_1_addr, (void *)input1,
                          input1_size * sizeof(int32_t));
  if (ret < 0)
    printf("[ERROR] Failed to write src1 to device at 0x%lx: err=%ld\n",
           src_1_addr, ret);
  printf("[INFO] Copied src0(%ld Bytes) to device at 0x%lx\n",
         input1_size * sizeof(int32_t), src_1_addr);

  size_t output_size = dims[2][0] * dims[2][1] * dims[2][2] * dims[2][3];
  ret = npu_memcpy_to_dev(&runtime.dev_state, dst_0_addr, (void *)output,
                          output_size * sizeof(int32_t));
  if (ret < 0)
    printf("[ERROR] Failed to write dst0 to device: at 0x%lx err=%ld\n",
           dst_0_addr, ret);
  printf("[INFO] Copied dst0(%ld Bytes) to device at 0x%lx\n",
         output_size * sizeof(int32_t), dst_0_addr);
}

static void testrt_npu_memcpy_out(void *output) {
  printf("[INFO] Copy data from device ... \n");
  size_t ret = 0;
  ret = npu_memcpy_from_dev(&runtime.dev_state, (void *)output, dst_0_addr,
                            M * N * sizeof(int32_t));
  if (ret < 0)
    printf("[ERROR] Failed to read dst0 from device: at 0x%lx: err=%ld\n",
           dst_0_addr, ret);
  printf("[INFO] Copied dst0(%ld Bytes) from device at 0x%lx\n",
         M * N * sizeof(int32_t), dst_0_addr);
}

void check_result(int32_t *output_real) {
  printf("[INFO] Check result ...\n");
  for (int n = 0; n < N; n++) {
    for (int m = 0; m < M; m++) {
      check(c[n * M + m], output_real[n * M + m]);
    }
  }
  printf("[INFO] Result check successfully!\n");
}

void check_fp32_result(float *output_real) {
  printf("[INFO] Check result ...\n");
  for (int n = 0; n < N; n++) {
    for (int m = 0; m < M; m++) {
      check(c_fp32[n * M + m], output_real[n * M + m]);
    }
  }
  printf("[INFO] Result check successfully!\n");
}

int testrt_npu_config_tcsrs(const long dims[3][4], const long strides[3][4]) {
  ssize_t err = 0;

  init_tcfg(dims, strides);
  err = npu_memcpy_to_dev(&runtime.dev_state, tcsr_addr, (void *)tcsr,
                          sizeof(tcsr));
  printf("[INFO] Copy tcsr to device with size:%dB and ret:%d\n", sizeof(tcsr),
         err);
}

int run_kernel(const char *kernel_name, void *input0, void *input1,
               void *output,  const long dims[3][4], const long strides[3][4]) {
  // Get kernel
  testrt_npu_get_kernel_desc(kernel_name);
  testrt_npu_config_tcsrs(dims, strides);

  printf("[INFO] M=%d, N=%d\n", M, N);

  // Copy in
  testrt_npu_memcpy_in(input0, input1, output, dims);

  int kernel_ret;
  ssize_t err = 0;
  printf("[INFO] Start kernel %s ... \n", kernel_name);
  err = npu_runtime_kernel_start(&runtime, kernel_desc, &kernel_ret, 5,
                                 src_0_addr, src_1_addr, dst_0_addr, M, N);
  printf("[INFO] Kernel ret=%d\n", kernel_ret);
  if (err != NPU_SUCCESS) {
    printf("[ERROR] Failed to start kernel : err=%ld\n", err);
    npu_runtime_deinit(&runtime);
    return -1;
  }
  printf("[INFO] Kernel %s run successfully.\n", kernel_name);

  // Copy out
  testrt_npu_memcpy_out(output);
}

int run_kernel_with_broadcast(const char *kernel_name, void *input0,
                              void *input1, void *output, const long dims[3][4],
                              const long strides[3][4]) {
  // Get kernel
  testrt_npu_get_kernel_desc(kernel_name);

  printf("[INFO] M=%d, N=%d\n", M, N);

  // Copy in
  testrt_npu_memcpy_in(input0, input1, output, dims);
  testrt_npu_config_tcsrs(dims, strides);

  int kernel_ret;

  printf("[INFO] Start kernel %s ... \n", kernel_name);
  int err = npu_runtime_kernel_start(&runtime, kernel_desc, &kernel_ret, 4,
                                 src_0_addr, src_1_addr, dst_0_addr, tcsr_addr);
  printf("[INFO] Kernel ret=%d\n", kernel_ret);
  if (err != NPU_SUCCESS) {
    printf("[ERROR] Failed to start kernel : err=%ld\n", err);
    npu_runtime_deinit(&runtime);
    return -1;
  }
  printf("[INFO] Kernel %s run successfully.\n", kernel_name);

  // Copy out
  testrt_npu_memcpy_out(output);
}

int run_lutop_kenrel(const char *kernel_name, void *input0, void *input1,
                     void *output, uint8_t lutop,  const long dims[3][4], const long strides[3][4]) {
  // Get kernel
  testrt_npu_get_kernel_desc(kernel_name);

  printf("[INFO] M=%d, N=%d\n", M, N);

  // Copy in
  testrt_npu_memcpy_in(input0, input1, output, dims);

  testrt_npu_config_tcsrs(dims, strides);


  int kernel_ret;
  ssize_t err = 0;
  printf("[INFO] Start kernel %s ... \n", kernel_name);
  err =
      npu_runtime_kernel_start(&runtime, kernel_desc, &kernel_ret, 5,
                               src_0_addr, src_1_addr, dst_0_addr, lutop, tcsr_addr);
  printf("[INFO] Kernel ret=%d\n", kernel_ret);
  if (err != NPU_SUCCESS) {
    printf("[ERROR] Failed to start kernel : err=%ld\n", err);
    npu_runtime_deinit(&runtime);
    return -1;
  }
  printf("[INFO] Kernel %s run successfully.\n", kernel_name);

  // Copy out
  testrt_npu_memcpy_out(output);
}

int main(int argc, char **argv) {
  testrt_npu_init();
  testrt_npu_load_kernel_set();
  testrt_run_tx_init_kernel();

  const long dims_0[3][4] = {{N, M, 1, 1}, {N, M, 1, 1}, {N, M, 1, 1}};
  const long strides_0[3][4] = {{1, N, M * N, M * N}, {1, N, N, N}, {1, N, M * N, M * N}};

  run_kernel_with_broadcast("tx_int_add_s32", a_in, b_in, c, dims_0, strides_0);
  check_result(c_add_real);

  run_kernel_with_broadcast("tx_int_sub_s32", a_in, b_in, c, dims_0, strides_0);
  check_result(c_sub_real);

  run_kernel_with_broadcast("tx_int_mul_s32", a_in, b_in, c, dims_0, strides_0);
  check_result(c_mul_real);

  run_kernel_with_broadcast("tx_int_srl_s32", a_in, shift_in, c, dims_0, strides_0);
  check_result(c_srl_real);

  run_kernel_with_broadcast("tx_int_sra_s32", a_in, shift_in, c, dims_0, strides_0);
  check_result(c_sra_real);

  run_kernel_with_broadcast("tx_int_sll_s32", a_in, shift_in, c, dims_0, strides_0);
  check_result(c_sll_real);

  run_lutop_kenrel("tx_int_lutop_s32", a_in, b_in, c, 8, dims_0, strides_0);
  check_result(c_and_real);

  run_lutop_kenrel("tx_int_lutop_s32", a_in, b_in, c, 14, dims_0, strides_0);
  check_result(c_or_real);

  run_lutop_kenrel("tx_int_lutop_s32", a_in, b_in, c, 6, dims_0, strides_0);
  check_result(c_xor_real);

  run_kernel_with_broadcast("tx_fp_add_fp32", a_fp32_in, b_fp32_in, c_fp32, dims_0, strides_0);
  check_fp32_result(c_fp32_add_real);

  run_kernel_with_broadcast("tx_fp_sub_fp32", a_fp32_in, b_fp32_in, c_fp32, dims_0, strides_0);
  check_fp32_result(c_fp32_sub_real);

  run_kernel_with_broadcast("tx_fp_mul_fp32", a_fp32_in, b_fp32_in, c_fp32, dims_0, strides_0);
  check_fp32_result(c_fp32_mul_real);

  const long dims_1[3][4] = {{N, M, 1, 1}, {N, 1, 1, 1}, {N, M, 1, 1}};
  const long strides_1[3][4] = {{1, N, M * N, M * N}, {1, N, N, N}, {1, N, M * N, M * N}};

  run_kernel_with_broadcast("tx_int_add_s32", a_in, b_row, c, dims_1, strides_1);
  check_result(c_add_row_bcast_real);

  run_kernel_with_broadcast("tx_int_mul_s32", a_in, b_row, c, dims_1, strides_1);
  check_result(c_mul_row_bcast_real);

  const long dims_2[3][4] = {{N, M, 1, 1}, {1, M, 1, 1}, {N, M, 1, 1}}; 
  const long strides_2[3][4] = {{1, N, M * N, M * N}, {1, 1, M, M}, {1, N, M * N, M * N}};

  run_kernel_with_broadcast("tx_int_sub_s32", a_in, b_col, c, dims_2, strides_2);
  check_result(c_sub_col_bcast_real);

  run_lutop_kenrel("tx_int_lutop_s32", a_in, b_in, c, 14, dims_0, strides_0);
  check_result(c_or_real);
  
  run_lutop_kenrel("tx_int_lutop_s32", a_in, b_col, c, 14, dims_2, strides_2);
  check_result(c_or_col_bcast_real);


  const long dims_3[3][4] = {{N, M, 1, 1}, {1, 1, 1, 1}, {N, M, 1, 1}}; 
  const long strides_3[3][4] = {{1, N, M * N, M * N}, {1, 1, 1, 1}, {1, N, M * N, M * N}};

  run_kernel_with_broadcast("tx_int_mul_s32", a_in, scalar, c, dims_3, strides_3);
  check_result(c_mul_scalar_bcast_real);

  run_lutop_kenrel("tx_int_lutop_s32", a_in, scalar, c, 6, dims_3, strides_3);
  check_result(c_xor_scalar_bcast_real);

  npu_runtime_deinit(&runtime);
  return 0;
}
