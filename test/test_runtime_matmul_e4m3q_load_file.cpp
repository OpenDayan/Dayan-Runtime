#include <cstddef>
#include <fcntl.h>
#include <iostream>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
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

#define fp_checkr(m, n, x, expect, eps)                                        \
  if ((x) < (expect) - (eps) || (x) > (expect) + (eps)) {                      \
    printf("Error: %s:%d: (%d,%d) %s = %x, expected %x\n", __FILE__, __LINE__, \
           m, n, #x, *(int *)&(x), *(int *)&(expect));                         \
  }

typedef union {
  float f32;
  int32_t i32;
} fp_bits_t;

#define fp_check2(m, n, x, expect, eps)                                              \
  do {                                                                         \
    fp_bits_t x_bits, expect_bits;                                             \
    x_bits.f32 = x;                                                            \
    expect_bits.f32 = expect;                                                  \
    if (x_bits.i32 != expect_bits.i32) {                                       \
      printf("Warning: %s:%d: (%d,%d) %s = %08x, expected %08x\n", __FILE__,   \
             __LINE__, m, n, #x, (x_bits.i32), (expect_bits.i32));             \
    }                                                                          \
  } while (0)

#define fp_check2f(m, n, x, expect, eps)                                             \
  do {                                                                         \
    fp_bits_t x_bits, expect_bits;                                             \
    x_bits.f32 = x;                                                            \
    expect_bits.f32 = expect;                                                  \
    if (x_bits.i32 != expect_bits.i32) {                                       \
      printf("Warning: %s:%d: (%d,%d) %s = %f, expected %f, diff %f\n",        \
             __FILE__, __LINE__, m, n, #x, (x), (expect), (x - expect));       \
    }                                                                          \
  } while (0)

void dump_mem(uint8_t *buf, size_t size) {
  for (ssize_t i = 0; i < size; i++) {
    if (i % 16 == 0) {
      printf("\n");
      printf("%04lx | ", (uintptr_t)buf + i);
    }
    printf("%02x ", buf[i]);
  }
  printf("\n\n");
}

// #include "generated/e4m3q_mat_size.gen"

#define M 1536
#define N 3
#define K 1536
#define QK 256

// // A^T mem               DIM0 DIM1
// float a_in[M * K] = {
// #include "generated/e4m3q_a_row.gen"
// }; // A[M][K]
// // B mem
// float b_in[N * K] = {
// #include "generated/e4m3q_b_col.gen"
// }; // B[N][K]
// // C mem

// // A^T mem               DIM0 DIM1
// float as_in[M * K / QK] = {
// #include "generated/e4m3q_as_row.gen"
// }; // A[M][K]
// // B mem
// float bs_in[N * K / QK] = {
// #include "generated/e4m3q_bs_col.gen"
// }; // B[N][K]
// // C mem

// struct blk_e4m3q {
//   float scale;
//   uint8_t v[QK];
// };

// struct blk_e4m3q a[M * K / QK];
// struct blk_e4m3q b[N * K / QK];

uint8_t *a;
uint8_t *b;
float *c_real;

float c[N * M] = {0};

// float c_real[N * M] = {
// #include "generated/e4m3q_c_col.gen"
// }; // C[N][M]

uint8_t __f32_to_e4m3(float f) { return RVT_EMU_FP8::from_float<4>(f); }

float __e4m3_to_f32(uint8_t e4m3) {
  RVT_EMU_FP8::FP8<4> data;
  data.bits = e4m3;
  return RVT_EMU_FP8::to_float(data);
}

extern struct NPUDeviceKernelSetDesc __general_entry_fpga;

int check_file_size(const std::string &filename, size_t expect_size) {
  int ret = 0;

  struct stat file_stat;

  ret = stat(filename.c_str(), &file_stat);
  if (ret < 0) {
    std::cout << filename << ": stat failed" << std::endl;
    return ret;
  }

  if (file_stat.st_size != expect_size) {
    std::cout << filename << ": invalid size, expect " << expect_size
              << ", file " << file_stat.st_size << std::endl;
    return -1;
  }

  return 0;
}

int load_file_malloc(const std::string &filename, void **ptr, size_t size) {
  int ret = 0;
  size_t left = size;
  uint8_t *read_ptr = NULL;

  int fd = open(filename.c_str(), O_RDONLY);
  if (fd < 0) {
    std::cout << filename << ": read failed" << std::endl;
    ret = -1;
    goto open_err;
  }

  *ptr = aligned_alloc(4096, size);
  if (*ptr == NULL) {
    std::cout << filename << ": malloc failed" << std::endl;
    ret = -1;
    goto mmap_err;
  }

  read_ptr = (uint8_t *)*ptr;
  while (left > 0) {
    ssize_t read_size = read(fd, read_ptr, left);
    if (read_size < 0) {
      std::cout << filename << ": read failed" << std::endl;
      ret = -1;
      goto read_err;
    }
    if (read_size == 0) {
      std::cout << filename << ": unexpected EOF" << std::endl;
      ret = -1;
      goto read_err;
    }
    left -= read_size;
    read_ptr += read_size; // Move pointer forward
  }

  return fd;

read_err:

mmap_err:
  close(fd);
open_err:
  return ret;
}

int fd_a = 0;
int fd_b = 0;
int fd_c = 0;
const ssize_t size_a = M * (K + K * 4 / QK) * sizeof(int8_t);
const ssize_t size_b = N * (K + K * 4 / QK) * sizeof(int8_t);
const ssize_t size_c = N * M * sizeof(float);

void free_close_file(void *addr, size_t size, int fd) {
  free(addr);
  close(fd);
}

void munmap_close_file_all() {
  free_close_file(a, size_a, fd_a);
  free_close_file(b, size_b, fd_b);
  free_close_file(c_real, size_c, fd_c);
}

int load_data(const std::string &filename_prefix) {

  const std::string filename_a = filename_prefix + "a.bin";
  const std::string filename_b = filename_prefix + "b.bin";
  const std::string filename_c = filename_prefix + "c.bin";

  int ret = 0;

  ret = check_file_size(filename_a, size_a);
  if (ret < 0) {
    goto invalid_size_err;
  }

  ret = check_file_size(filename_b, size_b);
  if (ret < 0) {
    goto invalid_size_err;
  }

  ret = check_file_size(filename_c, size_c);
  if (ret < 0) {
    goto invalid_size_err;
  }

  fd_a = load_file_malloc(filename_a, (void **)&a, size_a);
  if (ret < 0) {
    goto load_a_err;
  }

  fd_b = load_file_malloc(filename_b, (void **)&b, size_b);
  if (ret < 0) {
    goto load_b_err;
  }

  fd_c = load_file_malloc(filename_c, (void **)&c_real, size_c);
  if (ret < 0) {
    goto load_c_err;
  }

  return 0;

load_c_err:

  free_close_file(b, size_b, fd_b);
load_b_err:

  free_close_file(a, size_a, fd_a);
load_a_err:

invalid_size_err:
  return ret;
}

// void trans_f32_to_e4m3() {
//   for (int m = 0; m < M; m++) {
//     for (int n = 0; n < N; n++) {
//       c[m * N + n] = 0.0;
//     }
//   }
//   for (int m = 0; m < M; m++) {
//     for (int k = 0; k < (K / QK); k++) {
//       a[m * K / QK + k].scale = as_in[m * K / QK + k];
//     }
//   }
//   for (int n = 0; n < N; n++) {
//     for (int k = 0; k < (K / QK); k++) {
//       b[n * K / QK + k].scale = bs_in[n * K / QK + k];
//     }
//   }
//   for (int n = 0; n < N; n++) {
//     for (int k = 0; k < K; k++) {
//       b[(n * K + k) / QK].v[k % QK] = __f32_to_e4m3(b_in[n * K + k]);
//       fp_check(__e4m3_to_f32(b[(n * K + k) / QK].v[k % QK]), b_in[n * K + k],
//                0.0f);
//     }
//   }
//   for (int m = 0; m < M; m++) {
//     for (int k = 0; k < K; k++) {
//       a[(m * K + k) / QK].v[k % QK] = __f32_to_e4m3(a_in[m * K + k]);
//       fp_check(__e4m3_to_f32(a[(m * K + k) / QK].v[k % QK]), a_in[m * K + k],
//                0.0f);
//     }
//   }
//   fp_check(a[1].scale, ((float *)a)[1 * (QK / sizeof(float) + 1)], 0.0f);
// }

struct NPURuntimeState runtime;
struct NPUKernelDesc *kernel_desc;

uint64_t src_0_addr = 0x81000000;
uint64_t src_1_addr = 0x91000000;
uint64_t dst_0_addr = 0xa1000000;

void deinit() {
  munmap_close_file_all();
  npu_runtime_deinit(&runtime);
}

void testrt_npu_init() {
  ssize_t err = 0;
  printf("[INFO] Initializing Runtime..\n");
  err = npu_runtime_init(&runtime);
  if (err != NPU_SUCCESS) {
    printf("[ERROR] Failed to initialize Runtime: err=%zd\n", err);
    exit(-1);
  }
  printf("[INFO] Runtime initialized successfully.\n");
}

void testrt_npu_load_kernel_set() {
  ssize_t err = 0;
  err = npu_runtime_load_kernelset(&runtime, nullptr);
  if (err != NPU_SUCCESS) {
    printf("Failed to load kernel set: err=%zd\n", err);
    deinit();
    exit(-1);
  }
  printf("[INFO] Kernel set loaded successfully.\n");
}

void testrt_run_tx_init_kernel() {
  ssize_t err = 0;
  struct NPUKernelDesc *tx_init_kernel_desc;
  err =
      npu_runtime_get_kernel_desc(&runtime, "__tx_init", &tx_init_kernel_desc);
  if (err != NPU_SUCCESS) {
    printf("Failed to get kernel description for __tx_init: err=%zd\n", err);
    deinit();
    exit(-1);
  }
  printf("[INFO] Loaded __tx_init kernel desc successfully.\n");
  int kernel_ret;
  printf("[INFO] Start kernel __tx_init ... \n");
  err = npu_runtime_kernel_start(&runtime, tx_init_kernel_desc, &kernel_ret, 0);
  printf("[INFO] Kernel ret=%d\n", kernel_ret);
  if (err != NPU_SUCCESS) {
    printf("[ERROR] Failed to start kernel : err=%zu\n", err);
    deinit();
    exit(-1);
  }
  printf("[INFO] Kernel __tx_init run successfully.\n");
}

void testrt_npu_get_kernel_desc() {
  ssize_t err = 0;
  err = npu_runtime_get_kernel_desc(&runtime, "gmma_e4m3q_fp32_BT_CT_QK256",
                                    &kernel_desc);
  if (err != NPU_SUCCESS) {
    printf("Failed to get kernel description: err=%zu\n", err);
    deinit();
    exit(-1);
  }
  printf("[INFO] Loaded gmma_e4m3q_fp32_BT_CT_QK256 kernel desc successfully.\n");
}

void testrt_npu_memcpy_in(uint64_t lda, uint64_t ldb, uint64_t ldc) {
  printf("[INFO] Copy data to device ... \n");
  ssize_t ret = 0;
  ret = npu_memcpy_to_dev(&runtime.dev_state, src_0_addr, (void *)a,
                          M * lda * sizeof(uint8_t));
  if (ret < 0)
    printf("[ERROR] Failed to write src0 to device at 0x%lx: err=%zu\n",
           src_0_addr, ret);
  printf("[INFO] Copied src0(%ld Bytes) to device at 0x%lx\n", M * lda,
         src_0_addr);

  ret = npu_memcpy_to_dev(&runtime.dev_state, src_1_addr, (void *)b,
                          N * ldb * sizeof(uint8_t));
  if (ret < 0)
    printf("[ERROR] Failed to write src1 to device at 0x%lx: err=%zu\n",
           src_1_addr, ret);
  printf("[INFO] Copied src1(%ld Bytes) to device at 0x%lx\n", N * ldb,
         src_1_addr);

  ret = npu_memcpy_to_dev(&runtime.dev_state, dst_0_addr, (void *)c,
                          N * ldc * sizeof(float));
  if (ret < 0)
    printf("[ERROR] Failed to write dst0 to device: at 0x%lx err=%zu\n",
           dst_0_addr, ret);
  printf("[INFO] Copied dst0(%ld Bytes) to device at 0x%lx\n",
         N * ldc * sizeof(float), dst_0_addr);
}

void testrt_npu_memcpy_out(uint64_t ldc) {
  printf("[INFO] Copy data from device ... \n");
  ssize_t ret = 0;
  ret = npu_memcpy_from_dev(&runtime.dev_state, (void *)c, dst_0_addr,
                            N * ldc * sizeof(float));
  if (ret < 0)
    printf("[ERROR] Failed to read dst0 from device: at 0x%lx: err=%zu\n",
           dst_0_addr, ret);
  printf("[INFO] Copied dst0(%ld Bytes) from device at 0x%lx\n", ret,
         dst_0_addr);
}

void check_result() {
  printf("[INFO] Check result ...\n");
  for (int n = 0; n < N; n++) {
    for (int m = 0; m < M; m++) {
      fp_checkr(m, n, c[n * M + m], c_real[n * M + m], 1.0f);
    }
  }
  printf("[INFO] Result check successfully!\n");
}

int main(int argc, char **argv) {
  // trans_f32_to_e4m3();
  int load_ret = load_data("data/export_");
  if (load_ret < 0) {
    return load_ret;
  }
  // printf("a:\n");
  // test_dump_host_mem((uint8_t *)a, 128);
  // printf("b:\n");
  // test_dump_host_mem((uint8_t *)b, 128);

  testrt_npu_init();
  testrt_npu_load_kernel_set();
  testrt_run_tx_init_kernel();

  testrt_npu_get_kernel_desc();

  uint64_t lda = K + K * 4 / QK;
  uint64_t ldb = K + K * 4 / QK;
  uint64_t ldc = M;
  printf("[INFO] M=%d, N=%d, K=%d\n", M, N, K);
  printf("[INFO] lda=%ld, ldb=%ld, ldc=%ld\n", lda, ldb, ldc);

  // Copy in
  testrt_npu_memcpy_in(lda, ldb, ldc);

  // test_dump_mem(&runtime.dev_state, src_0_addr, 128);
  // test_dump_mem(&runtime.dev_state, src_1_addr, 128);
  // test_dump_mem(&runtime.dev_state, dst_0_addr, 128);

  int kernel_ret;
  ssize_t err = 0;
  printf("[INFO] Start kernel gmma_e4m3q_fp32_BT_CT_QK256 ... \n");
  err = npu_runtime_kernel_start(
      &runtime, kernel_desc, &kernel_ret, 9, src_0_addr, src_1_addr, dst_0_addr,
      (uint64_t)M, (uint64_t)N, (uint64_t)K, lda, ldb, ldc);
  printf("[INFO] Kernel ret=%d\n", kernel_ret);
  if (err != NPU_SUCCESS) {
    printf("[ERROR] Failed to start kernel : err=%zu\n", err);
    deinit();
    return -1;
  }
  printf("[INFO] Kernel gmma_e4m3q_fp32_BT_CT_QK256 run successfully.\n");

  // Copy out
  testrt_npu_memcpy_out(ldc);

  // Check result
  check_result();

  // End
  deinit();

  return 0;
}
