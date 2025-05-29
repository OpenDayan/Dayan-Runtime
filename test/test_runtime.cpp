#include <chrono>
#include <iostream>
#include <ostream>
#include <stdio.h>
#include <unistd.h>

#include "NPURuntime.h"

extern char __bin_helloworld_fpga[];
extern size_t __bin_size_helloworld_fpga;

struct test_runtime
{
  /* data */
};

struct NPUKernelDesc kernels[] = {
  {
    .entry = 0x80002000,
    .name = "main"
  }
};

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
  .data = (uint8_t*)&__bin_helloworld_fpga[0],
  .kernels = &kernels[0]
};


int main() {
  std::cout << "Initializing Runtime..." << std::endl;
  struct NPURuntimeState runtime;
  NPURuntimeError_t err;
  err = npu_runtime_init(&runtime);
  if (err != NPU_SUCCESS) {
    std::cerr << "Failed to initialize Runtime: " << err << std::endl;
    return -1;
  }

  std::cout << "Runtime initialized successfully." << std::endl;

  // Perform operations with the device
  err = npu_runtime_load_kernelset(&runtime, &kernel_set);
  if (err != NPU_SUCCESS) {
    std::cerr << "Failed to load kernel set: " << err << std::endl;
    npu_runtime_deinit(&runtime);
    return -1;
  }

  struct NPUKernelDesc* kernel_desc;
  err = npu_runtime_get_kernel_desc(&runtime, "main", &kernel_desc);
  if (err != NPU_SUCCESS) {
    std::cerr << "Failed to get kernel description: " << err << std::endl;
    npu_runtime_deinit(&runtime);
    return -1;
  }

  err = npu_runtime_kernel_start(&runtime, kernel_desc, NULL, 0);
  if (err != NPU_SUCCESS) {
    std::cerr << "Failed to start kernel: " << err << std::endl;
    npu_runtime_deinit(&runtime);
    return -1;
  }

  for(int i = 0 ; i < 20 ; i++) {
    // Check if the kernel is running
    long local_var;
    npu_memcpy_from_dev(&runtime.dev_state, &local_var, 0x81000000, sizeof(long));
    std::cout << "Kernel local_var: " << local_var << std::endl;
    sleep(1); // Sleep for 1 second
  }

  npu_runtime_deinit(&runtime);

  return 0;
}
