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

#include "../npu-kernels/testdata_gen/generated/convert_size.gen"

float a_fp32_in[M * N] = {
#include "../npu-kernels/testdata_gen/generated/convert_float_a_row.gen"
};

int32_t a_floor_real[M * N] = {
#include "../npu-kernels/testdata_gen/generated/convert_a_floor_row.gen"
};

int32_t a_ceil_real[M * N] = {
#include "../npu-kernels/testdata_gen/generated/convert_a_ceil_row.gen"
};

int32_t a_round_real[M * N] = {
#include "../npu-kernels/testdata_gen/generated/convert_a_round_row.gen"
};

uint8_t a_e4m3_round_real[M * N] = {
#include "../npu-kernels/testdata_gen/generated/convert_a_fp8_e4m3_round_row.gen"
};

uint16_t a_fp16_round_real[M * N] = {
#include "../npu-kernels/testdata_gen/generated/convert_a_fp16_round_row.gen"
};

uint16_t a_bf16_round_real[M * N] = {
#include "../npu-kernels/testdata_gen/generated/convert_a_bf16_round_row.gen"
};

int32_t c[M * N] = {0};
int8_t c_e4m3[M * N] = {0};
int16_t c_fp16[M * N] = {0};
int16_t c_bf16[M * N] = {0};

extern struct NPUDeviceKernelSetDesc __general_entry_fpga;

static struct NPURuntimeState runtime;
static struct NPUKernelDesc *kernel_desc;

static uint64_t src_addr = 0x81000000;
static uint64_t dst_addr = 0x82000000;
static uint64_t tcsr_addr = 0x84000000;

static TCFG tcsr[2];

void init_tcfg(const long dims[2][4], const long strides[2][4]) {
  for (int i = 0; i < 2; ++i) {
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

static void testrt_npu_memcpy_in(void *input, void *output,
                                 const long dims[2][4],
                                 uint32_t input_type_size,
                                 uint32_t output_type_size) {
  printf("[INFO] Copy data to device ... \n");
  size_t ret = 0;
  size_t input_size = dims[0][0] * dims[0][1] * dims[0][2] * dims[0][3];
  ret = npu_memcpy_to_dev(&runtime.dev_state, src_addr, (void *)input,
                          input_size * sizeof(input_type_size));
  if (ret < 0)
    printf("[ERROR] Failed to write src to device at 0x%lx: err=%ld\n",
           src_addr, ret);
  printf("[INFO] Copied src(%ld Bytes) to device at 0x%lx\n",
         input_size * sizeof(input_type_size), src_addr);

  size_t output_size = dims[1][0] * dims[1][1] * dims[1][2] * dims[1][3];
  ret = npu_memcpy_to_dev(&runtime.dev_state, dst_addr, (void *)output,
                          output_size * sizeof(output_type_size));
  if (ret < 0)
    printf("[ERROR] Failed to write dst0 to device: at 0x%lx err=%ld\n",
           dst_addr, ret);
  printf("[INFO] Copied dst0(%ld Bytes) to device at 0x%lx\n",
         output_size * sizeof(output_type_size), dst_addr);
}

static void testrt_npu_memcpy_out(void *output, uint32_t output_type_size) {
  printf("[INFO] Copy data from device ... \n");
  size_t ret = 0;
  ret = npu_memcpy_from_dev(&runtime.dev_state, (void *)output, dst_addr,
                            M * N * output_type_size);
  if (ret < 0)
    printf("[ERROR] Failed to read dst0 from device: at 0x%lx: err=%ld\n",
           dst_addr, ret);
  printf("[INFO] Copied dst0(%ld Bytes) from device at 0x%lx\n",
         M * N * output_type_size, dst_addr);
}

typedef enum {
  CONVERT_U8,
  CONVERT_U16,
  CONVERT_U32,
} ConvertType;

void check_result(void *output, void *output_real, ConvertType dtype) {
  printf("[INFO] Check result ...\n");

  for (int n = 0; n < N; n++) {
    for (int m = 0; m < M; m++) {
      size_t offset = n * M + m;
      switch (dtype) {
      case CONVERT_U8: {
        uint8_t val_output = ((uint8_t *)output)[offset];
        uint8_t val_real = ((uint8_t *)output_real)[offset];
        if (val_output < val_real - 1 || val_output > val_real + 1) {
          printf(
              "[ERROR] Mismatch at [%d][%d]: output=0x%02X, expected=0x%02X\n",
              n, m, val_output, val_real);
          return;
        }
        break;
      }
      case CONVERT_U16: {
        uint16_t val_output = ((uint16_t *)output)[offset];
        uint16_t val_real = ((uint16_t *)output_real)[offset];
        if (val_output != val_real) {
          printf(
              "[ERROR] Mismatch at [%d][%d]: output=0x%04X, expected=0x%04X\n",
              n, m, val_output, val_real);
          return;
        }
        break;
      }
      case CONVERT_U32: {
        uint32_t val_output = ((uint32_t *)output)[offset];
        uint32_t val_real = ((uint32_t *)output_real)[offset];
        if (val_output != val_real) {
          printf(
              "[ERROR] Mismatch at [%d][%d]: output=0x%08X, expected=0x%08X\n",
              n, m, val_output, val_real);
          return;
        }
        break;
      }
      default:
        printf("[ERROR] Unsupported data type for checking\n");
        return;
      }
    }
  }
  printf("[INFO] Result check successfully!\n");
}

void check_fp32_result(float *output_real) {
  printf("[INFO] Check result ...\n");
  for (int n = 0; n < N; n++) {
    for (int m = 0; m < M; m++) {
      fp_check(c[n * M + m], output_real[n * M + m], 0.0f);
    }
  }
  printf("[INFO] Result check successfully!\n");
}

void testrt_npu_config_tcsrs(const long dims[2][4], const long strides[2][4]) {
  int err = 0;

  init_tcfg(dims, strides);
  err = npu_memcpy_to_dev(&runtime.dev_state, tcsr_addr, (void *)tcsr,
                          sizeof(tcsr));
  printf("[INFO] Copy tcsr to device with size:%ld and ret:%d\n", sizeof(tcsr),
         err);
}

int run_kernel(const char *kernel_name, void *input, void *output,
               const long dims[2][4], const long strides[2][4],
               uint32_t input_type_size, uint32_t output_type_size) {
  // Get kernel
  testrt_npu_get_kernel_desc(kernel_name);

  // Copy in
  testrt_npu_memcpy_in(input, output, dims, input_type_size, output_type_size);
  testrt_npu_config_tcsrs(dims, strides);

  printf("[INFO] Start kernel %s ... \n", kernel_name);

  int kernel_ret = 0;
  int err = 0;
  npu_runtime_kernel_start(&runtime, kernel_desc, &kernel_ret, 3, src_addr,
                           dst_addr, tcsr_addr);
  if (err != NPU_SUCCESS) {
    printf("[ERROR] Failed to start kernel : err=%d\n", err);
    npu_runtime_deinit(&runtime);
    return -1;
  }
  printf("[INFO] Kernel %s run successfully.\n", kernel_name);

  // Copy out
  testrt_npu_memcpy_out(output, output_type_size);
  return 0;
}

int main(int argc, char **argv) {
  testrt_npu_init();
  testrt_npu_load_kernel_set();
  testrt_run_tx_init_kernel();

  const long dims[2][4] = {{N, M, 1, 1}, {N, M, 1, 1}};
  const long strides[2][4] = {{1, N, M * N, M * N}, {1, N, M * N, M * N}};

  run_kernel("tx_convert_fp32_s32_round", a_fp32_in, c, dims, strides, 4, 4);
  check_result(c, a_round_real, CONVERT_U32);

  run_kernel("tx_convert_fp32_s32_floor", a_fp32_in, c, dims, strides, 4, 4);
  check_result(c, a_floor_real, CONVERT_U32);

  run_kernel("tx_convert_fp32_s32_ceil", a_fp32_in, c, dims, strides, 4, 4);
  check_result(c, a_ceil_real, CONVERT_U32);

  run_kernel("tx_convert_fp32_e4m3_round", a_fp32_in, c_e4m3, dims, strides, 4,
             1);
  check_result(c_e4m3, a_e4m3_round_real, CONVERT_U8);

  run_kernel("tx_convert_fp32_fp16_round", a_fp32_in, c_fp16, dims, strides, 4,
             2);
  check_result(c_fp16, a_fp16_round_real, CONVERT_U16);

  run_kernel("tx_convert_fp32_bf16_round", a_fp32_in, c_bf16, dims, strides, 4,
             2);
  check_result(c_bf16, a_bf16_round_real, CONVERT_U16);

  npu_runtime_deinit(&runtime);
  return 0;
}
