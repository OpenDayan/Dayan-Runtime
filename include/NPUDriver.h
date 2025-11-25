#ifndef _NPU_DRIVER_H_
#define _NPU_DRIVER_H_

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#ifdef __cplusplus
extern "C"
{
#endif

  // Device State Structure
  struct NPUDeviceState
  {
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
                              const void *src, size_t size);
  ssize_t npu_memcpy_from_dev(const struct NPUDeviceState *device, void *dst,
                              uint64_t src, size_t size);

  ssize_t npu_check(const struct NPUDeviceState *device);                        // Check device state: WaitLaunch / Busy / Empty
  ssize_t npu_abort(const struct NPUDeviceState *device);                        // Stop Kernel Excution
  ssize_t npu_start(const struct NPUDeviceState *device, uint64_t boot_address); // Start Kernel Excution

  // function return negative values on error
  typedef enum
  {
    NPU_DEVICE_SUCCESS,
    NPU_DEVICE_ERRNO_UNINITIALIZED,
    NPU_DEVICE_ERRNO_UNSUPPORTED,
    NPU_DEVICE_ERRNO_IO,
    NPU_DEVICE_ERRNO_INVALID_ADDRESS,
    NPU_DEVICE_ERRNO_INVALID_SIZE,
    NPU_DEVICE_ERRNO_INVALID_ARGS, // null pointers, etc.
    NPU_DEVICE_ERRNO_BUSY,
    NPU_DEVICE_ERRNO_WAITLAUNCH,
    NPU_DEVICE_ERRNO_EMPTY,
  } NPUDeviceErrno_t;

#ifdef __cplusplus
}
#endif

// For debug
static inline void test_dump_mem(const struct NPUDeviceState *device, uint64_t addr,
                                 size_t size)
{
  uint8_t *buf = (uint8_t *)malloc(size);
  if (!buf)
  {
    printf("Failed to allocate memory for buffer\n");
    return;
  }

  size_t ret = npu_memcpy_from_dev(device, buf, addr, size);
  if (ret < 0)
  {
    printf("Failed to read memory from device: %ld\n", ret);
    free(buf);
    return;
  }

  for (size_t i = 0; i < size; i++)
  {
    if (i % 16 == 0)
    {
      printf("\n");
      printf("%04lx | ", (uintptr_t)addr + i);
    }
    printf("%02x ", buf[i]);
  }
  printf("\n\n");

  free(buf);
}

static inline void test_dump_host_mem(uint8_t *buf,
                                 size_t size)
{
  for (size_t i = 0; i < size; i++)
  {
    if (i % 16 == 0)
    {
      printf("\n");
      printf("%04lx | ", (uintptr_t)buf + i);
    }
    printf("%02x ", buf[i]);
  }
  printf("\n\n");

}

#endif
