#ifndef _NPU_DRIVER_H_
#define _NPU_DRIVER_H_

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

// Device State Structure
struct NPUDeviceState {
  // TODO: Define needed stateful data for the NPU Device here.
  size_t xdma_id;
  int xdma_c2h_fd;
  int xdma_h2c_fd;
  uint8_t valid;
};

// All Driver APIs are synchronous

// Define Basic Driver API
ssize_t npu_device_init(struct NPUDeviceState *device);
void npu_device_deinit(struct NPUDeviceState *device);

ssize_t npu_mmio_read64(const struct NPUDeviceState *device, uint64_t address,
                        uint64_t *value);
ssize_t npu_mmio_write64(const struct NPUDeviceState *device, uint64_t address,
                         uint64_t value);
ssize_t npu_mmio_read32(const struct NPUDeviceState *device, uint64_t address,
                        uint32_t *value);
ssize_t npu_mmio_write32(const struct NPUDeviceState *device, uint64_t address,
                         uint32_t value);
ssize_t npu_mmio_read16(const struct NPUDeviceState *device, uint64_t address,
                        uint16_t *value);
ssize_t npu_mmio_write16(const struct NPUDeviceState *device, uint64_t address,
                         uint16_t value);
ssize_t npu_mmio_read8(const struct NPUDeviceState *device, uint64_t address,
                       uint8_t *value);
ssize_t npu_mmio_write8(const struct NPUDeviceState *device, uint64_t address,
                        uint8_t value);

ssize_t npu_memcpy_to_dev(const struct NPUDeviceState *device, uint64_t dst,
                          void *src, size_t size);
ssize_t npu_memcpy_from_dev(const struct NPUDeviceState *device, void *dst,
                            uint64_t src, size_t size);

ssize_t npu_reset(const struct NPUDeviceState *device);
ssize_t npu_start(const struct NPUDeviceState *device, uint64_t boot_address);

// function return negative values on error
typedef enum {
  NPU_DEVICE_SUCCESS,
  NPU_DEVICE_ERRNO_UNINITIALIZED,
  NPU_DEVICE_ERRNO_UNSUPPORTED,
  NPU_DEVICE_ERRNO_IO,
  NPU_DEVICE_ERRNO_INVALID_ADDRESS,
  NPU_DEVICE_ERRNO_INVALID_SIZE,
  NPU_DEVICE_ERRNO_INVALID_ARGS, // null pointers, etc.
} NPUDeviceErrno_t;

#ifdef __cplusplus
}
#endif

#endif
