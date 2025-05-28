#include "NPUDriver.h"

int main() {
  struct NPUDeviceState device = {0};
  device.xdma_id = 0;
  ssize_t ret = npu_device_init(&device);
  if (ret < 0) {
    // Handle error
    return ret;
  }
  // Perform operations with the device
  npu_device_deinit(&device);
  return 0;
}
