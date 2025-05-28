#include <chrono>
#include <iostream>
#include <ostream>

#include "NPURuntime.h"

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

  npu_runtime_deinit(&runtime);

  return 0;
}
