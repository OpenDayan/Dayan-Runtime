#include <chrono>
#include <cstddef>
#include <iostream>
#include <ostream>

#include "NPUDriver.h"

#define BUF_SIZE (1024 * 1024 * 1024)

ssize_t test_memcpy(const struct NPUDeviceState *device) {
  ssize_t ret;

  const uint64_t device_addr = 0x81000000;

  const size_t test_num = 1;

  uint8_t *h2c_buf = (uint8_t *)aligned_alloc(4096, BUF_SIZE);
  if (!h2c_buf) {
    std::cerr << "Failed to allocate h2c_buf" << std::endl;
    return -1;
  }
  std::cout << "h2c_buf address: " << static_cast<void *>(h2c_buf) << std::endl;

  for (size_t i = 0; i < BUF_SIZE; ++i) {
    h2c_buf[i] = 0xFF;
  }

  auto h2c_start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < test_num; i++) {
    ret = npu_memcpy_to_dev(device, device_addr, h2c_buf, BUF_SIZE);
  }
  auto h2c_end = std::chrono::high_resolution_clock::now();

  if (ret < 0) {
    std::cerr << "npu_memcpy_to_dev fail: " << ret << std::endl;
    return -2;
  }

  std::chrono::duration<double> h2c_duration = h2c_end - h2c_start;
  double h2c_speed = ret * test_num / (1024.0 * 1024.0) / h2c_duration.count();

  std::cout << "H2C: Wrote " << ret << "*" << test_num << " bytes to device. ";
  std::cout << "Speed: " << h2c_speed << " MiB/s" << std::endl;

  char *c2h_buf = (char *)aligned_alloc(4096, BUF_SIZE);
  if (!c2h_buf) {
    std::cerr << "Failed to allocate c2h_buf" << std::endl;
    return -3;
  }
  std::cout << "c2h_buf address: " << static_cast<void *>(c2h_buf) << std::endl;

  auto c2h_start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < test_num; i++) {
    ret = npu_memcpy_from_dev(device, c2h_buf, device_addr, BUF_SIZE);
  }
  auto c2h_end = std::chrono::high_resolution_clock::now();

  if (ret < 0) {
    std::cerr << "npu_memcpy_from_dev fail: " << ret << std::endl;
    return -4;
  }

  std::chrono::duration<double> c2h_duration = c2h_end - c2h_start;
  double c2h_speed = ret * test_num/ (1024.0 * 1024.0) / c2h_duration.count();

  std::cout << "C2H: Read " << ret << "*" << test_num << " bytes from device. ";
  std::cout << "Speed: " << c2h_speed << " MiB/s" << std::endl;

  bool match = true;
  for (ssize_t i = 0; i < ret; ++i) {
    if (c2h_buf[i] != h2c_buf[i]) {
      match = false;
      std::cerr << "Data mismatch at byte " << i << std::endl;
      break;
    }
  }

  if (match) {
    std::cout << "memcpy test passed." << std::endl;
  }

  return 0;
}

int main() {
  std::cout << "Initializing NPU Device..." << std::endl;
  struct NPUDeviceState device = {0};
  device.xdma_id = 0;
  ssize_t ret;

  ret = npu_device_init(&device);
  if (ret < 0) {
    // Handle error
    std::cerr << "Failed to initialize NPU Device: " << ret << std::endl;
    return ret;
  }

  std::cout << "NPU Device initialized successfully." << std::endl;
  // Perform operations with the device
  test_memcpy(&device);

  npu_device_deinit(&device);
  return 0;
}
