#include <chrono>
#include <cstddef>
#include <iostream>
#include <ostream>

#include "NPUDriver.h"

ssize_t test_memset(const struct NPUDeviceState *device, uint64_t begin,
                    uint8_t value, size_t size) {
  ssize_t ret;

  uint8_t *buf = (uint8_t *)aligned_alloc(4096, size);
  if (!buf) {
    std::cerr << "Failed to allocate buf" << std::endl;
    return -1;
  }
  std::cout << "h2c_buf address: " << static_cast<void *>(buf) << std::endl;

  for (size_t i = 0; i < size; ++i) {
    buf[i] = value;
  }

  auto h2c_start = std::chrono::high_resolution_clock::now();
  ret = npu_memcpy_to_dev(device, begin, buf, size);
  auto h2c_end = std::chrono::high_resolution_clock::now();

  if (ret < 0) {
    std::cerr << "npu_memcpy_to_dev fail: " << ret << std::endl;
    return -2;
  }

  std::chrono::duration<double> h2c_duration = h2c_end - h2c_start;
  double h2c_speed = ret / (1024.0 * 1024.0) / h2c_duration.count();

  std::cout << "H2C: Wrote " << ret << " bytes to device. ";
  std::cout << "Speed: " << h2c_speed << " MiB/s" << std::endl;

  auto c2h_start = std::chrono::high_resolution_clock::now();
  ret = npu_memcpy_from_dev(device, buf, begin, size);
  auto c2h_end = std::chrono::high_resolution_clock::now();
  if (ret < 0) {
    std::cerr << "npu_memcpy_from_dev fail: " << ret << std::endl;
    return -3;
  }

  std::chrono::duration<double> c2h_duration = c2h_end - c2h_start;
  double c2h_speed = ret / (1024.0 * 1024.0) / c2h_duration.count();

  std::cout << "C2H: Read " << ret << " bytes from device. ";
  std::cout << "Speed: " << c2h_speed << " MiB/s" << std::endl;

  for (size_t i = 0; i < size; ++i) {
    if (buf[i] != value) {
      std::cerr << "Data mismatch at index " << i << ": expected "
                << static_cast<int>(value) << ", got "
                << static_cast<int>(buf[i]) << std::endl;
      free(buf);
      return -4;
    }
  }
  std::cout << "Data verification successful." << std::endl;
  free(buf);

  return 0;
}

int main(int argc, char *argv[]) {

  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " <begin_address> <value> <size>"
              << std::endl;
    return -1;
  }

  uint64_t begin = std::stoull(argv[1], nullptr, 0);
  uint8_t value = static_cast<uint8_t>(std::stoul(argv[2], nullptr, 0));
  size_t size = static_cast<size_t>(std::stoull(argv[3], nullptr, 0));

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
  test_memset(&device, begin, value, size);

  npu_device_deinit(&device);
  return 0;
}
