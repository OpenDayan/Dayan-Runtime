#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <ostream>
#include <stdio.h>
#include <sys/types.h>
#include <unistd.h>

#include "NPUDriver.h"
#include "NPURuntime.h"

extern char __bin_helloworld_fpga[];
extern size_t __bin_size_helloworld_fpga;

struct test_runtime {
  /* data */
};

struct NPUKernelDesc kernels[] = {{.entry = 0x80002000, .name = "main"}};

struct NPUDeviceKernelSetDesc kernel_set = {
    .magic = 0,
    .flags = 0,
    .version = 0,
    .reserve1 = 0,
    .load_start = 0x80000000,
    .load_size = __bin_size_helloworld_fpga,
    .kernel_count = 1,
    .general_entry = 0x80000000,
    .args_ptr = 0x81000000,
    .data = (uint8_t *)&__bin_helloworld_fpga[0],
    .kernels = &kernels[0]};

#define CCACHE_CTRL_ADDR 0x02010000
#define CCACHE_CTRL_OFFSET_CONFIG 0x0000
#define CCACHE_CTRL_OFFSET_CONFIG_BANKS 0x0000
#define CCACHE_CTRL_OFFSET_CONFIG_WAYS 0x0001
#define CCACHE_CTRL_OFFSET_CONFIG_LG_SETS 0x0002
#define CCACHE_CTRL_OFFSET_CONFIG_LG_BLOCK_BYTES 0x0003

typedef struct {
  uint8_t banks;
  uint8_t ways;
  uint8_t lg_sets;
  uint8_t lg_block_bytes;
} ccache_reg_t;
static_assert(sizeof(ccache_reg_t) == 4);

void test_get_l2_info(const struct NPUDeviceState *device) {
  ccache_reg_t ccache_config;
  npu_memcpy_from_dev(device, &ccache_config,
                      CCACHE_CTRL_ADDR + CCACHE_CTRL_OFFSET_CONFIG,
                      sizeof(ccache_reg_t));
  printf("L2 Cache Config:\n");
  printf("  Banks: %u\n", ccache_config.banks);
  printf("  Ways: %u\n", ccache_config.ways);
  printf("  Sets: %u\n", 1 << ccache_config.lg_sets);
  printf("  Block Size: %u bytes\n", 1 << ccache_config.lg_block_bytes);
}

void test_dump_mem(const struct NPUDeviceState *device, uint64_t addr,
                   size_t size) {
  uint8_t *buf = (uint8_t *)malloc(size);
  if (!buf) {
    std::cerr << "Failed to allocate memory for buffer" << std::endl;
    return;
  }

  ssize_t ret = npu_memcpy_from_dev(device, buf, addr, size);
  if (ret < 0) {
    std::cerr << "Failed to read memory from device: " << ret << std::endl;
    free(buf);
    return;
  }

  for (ssize_t i = 0; i < size; i++) {
    if (i % 16 == 0) {
      printf("\n");
      printf("%04lx | ", (uintptr_t)addr + i);
    }
    printf("%02x ", buf[i]);
  }
  printf("\n\n");

  free(buf);
}

int main() {
  std::cout << "Initializing Runtime..." << std::endl;
  struct NPURuntimeState runtime = {0};
  NPURuntimeError_t err;
  err = npu_runtime_init(&runtime);
  if (err != NPU_SUCCESS) {
    std::cerr << "Failed to initialize Runtime: " << err << std::endl;
    return -1;
  }

  std::cout << "Runtime initialized successfully." << std::endl;

  // Get L2 cache info
  // test_get_l2_info(&runtime.dev_state);

  // Perform operations with the device
  err = npu_runtime_load_kernelset(&runtime, &kernel_set);
  if (err != NPU_SUCCESS) {
    std::cerr << "Failed to load kernel set: " << err << std::endl;
    npu_runtime_deinit(&runtime);
    return -1;
  }

  std::cout << "Kernel set loaded successfully." << std::endl;

  test_dump_mem(&runtime.dev_state, 0x80002000, 0x40);

  struct NPUKernelDesc *kernel_desc;
  err = npu_runtime_get_kernel_desc(&runtime, "main", &kernel_desc);
  if (err != NPU_SUCCESS) {
    std::cerr << "Failed to get kernel description: " << err << std::endl;
    npu_runtime_deinit(&runtime);
    return -1;
  }

  std::cout << "Kernel Entry: " << std::hex << kernel_desc->entry
            << std::dec << std::endl;

  test_dump_mem(&runtime.dev_state, 0x81000000, 8);

  err = npu_runtime_kernel_start(&runtime, kernel_desc, NULL, 0);
  if (err != NPU_SUCCESS) {
    std::cerr << "Failed to start kernel: " << err << std::endl;
    npu_runtime_deinit(&runtime);
    return -1;
  }

  std::cout << "Kernel started successfully." << std::endl;

  // test_dump_mem(&runtime.dev_state, 0x82000000, 32);

  test_dump_mem(&runtime.dev_state, 0x81000000, 128);

  test_dump_mem(&runtime.dev_state, 0x82000000, 128);

  for (int i = 0; i < 20; i++) {
    // Check if the kernel is running
    size_t ret;
    long local_var;
    ret = npu_memcpy_from_dev(&runtime.dev_state, &local_var, 0x82000000,
                              sizeof(long));
    if (ret < 0) {
      std::cerr << "Failed to read local_var from device: " << ret << std::endl;
    }
    std::cout << "Kernel local_var: " << local_var << std::endl;

    // test_dump_mem(&runtime.dev_state, 0x81000000, 32);

    sleep(1); // Sleep for 1 second
  }

  npu_runtime_deinit(&runtime);

  return 0;
}
