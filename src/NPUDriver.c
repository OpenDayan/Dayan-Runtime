#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>
#include <string.h>

#include "NPUDriver.h"

#if 1
#define H2C_DEV "/tmp/virtual_npu/xdma0_h2c_0"
#define C2H_DEV "/tmp/virtual_npu/xdma0_c2h_0"
#else
#define H2C_DEV "/dev/xdma0_h2c_0"
#define C2H_DEV "/dev/xdma0_c2h_0"
#endif

#define MMIO_ADDR_BOOT_ADDR_REG 0x1000
#define MMIO_ADDR_TILE_RESET_CTRL 0x110000
#define MMIO_ADDR_CLINT 0x2000000

#ifdef __cplusplus
extern "C" {
#endif

ssize_t npu_device_init(struct NPUDeviceState *device) {
  ssize_t ret;
  if (device == NULL) {
    ret = -NPU_DEVICE_ERRNO_INVALID_ARGS;
    goto npu_device_error;
  }

  if (device->xdma_id != 0) {
    // currently only support one device
    ret = -NPU_DEVICE_ERRNO_UNSUPPORTED;
    goto xdma_id_error;
  }

  device->xdma_h2c_fd = open(H2C_DEV, O_WRONLY);
  if (device->xdma_h2c_fd < 0) {
    ret = -NPU_DEVICE_ERRNO_IO;
    goto xdma_h2c_error;
  }

  device->xdma_c2h_fd = open(C2H_DEV, O_RDONLY);
  if (device->xdma_c2h_fd < 0) {
    ret = -NPU_DEVICE_ERRNO_IO;
    goto xdma_c2h_error;
  }

  device->valid = 1; // Mark the device as valid

  return NPU_DEVICE_SUCCESS;

  // close(device->xdma_c2h_fd);
  // device->xdma_c2h_fd = -1;

xdma_c2h_error:
  close(device->xdma_h2c_fd);
  device->xdma_h2c_fd = -1;

xdma_h2c_error:

xdma_id_error:

npu_device_error:
  return ret;
}

void npu_device_deinit(struct NPUDeviceState *device) {
  if (device == NULL) {
    return;
  }

  if (device->xdma_c2h_fd >= 0) {
    close(device->xdma_c2h_fd);
    device->xdma_c2h_fd = -1;
  }

  if (device->xdma_h2c_fd >= 0) {
    close(device->xdma_h2c_fd);
    device->xdma_h2c_fd = -1;
  }

  device->valid = 0; // Mark the device as invalid
}

#define FUNC_NPU_MMIO_READ(type, name)                                         \
  ssize_t npu_mmio_read##name(const struct NPUDeviceState *device,             \
                              uint64_t address, type *value) {                 \
    if (device == NULL || value == NULL) {                                     \
      return -NPU_DEVICE_ERRNO_INVALID_ARGS;                                   \
    }                                                                          \
    if (!device->valid) {                                                      \
      return -NPU_DEVICE_ERRNO_UNINITIALIZED;                                  \
    }                                                                          \
    if (address % sizeof(type) != 0) {                                         \
      return -NPU_DEVICE_ERRNO_INVALID_ADDRESS;                                \
    }                                                                          \
    ssize_t ret;                                                               \
    ret = lseek(device->xdma_c2h_fd, address, SEEK_SET);                       \
    if (ret < 0) {                                                             \
      return -NPU_DEVICE_ERRNO_IO;                                             \
    }                                                                          \
    uint8_t buf[sizeof(type)];                                                 \
    ret = read(device->xdma_c2h_fd, buf, sizeof(type));                        \
    if (ret < 0) {                                                             \
      return -NPU_DEVICE_ERRNO_IO;                                             \
    }                                                                          \
    if (ret != sizeof(type)) {                                                 \
      return -NPU_DEVICE_ERRNO_IO;                                             \
    }                                                                          \
    *value = *(type *)buf;                                                     \
    return ret;                                                                \
  }

#define FUNC_NPU_MMIO_WRITE(type, name)                                        \
  ssize_t npu_mmio_write##name(const struct NPUDeviceState *device,            \
                               uint64_t address, type value) {                 \
                                                                               \
    if (device == NULL) {                                                      \
      return -NPU_DEVICE_ERRNO_INVALID_ARGS;                                   \
    }                                                                          \
    if (!device->valid) {                                                      \
      return -NPU_DEVICE_ERRNO_UNINITIALIZED;                                  \
    }                                                                          \
    if (address % sizeof(type) != 0) {                                         \
      return -NPU_DEVICE_ERRNO_INVALID_ADDRESS;                                \
    }                                                                          \
    ssize_t ret;                                                               \
    ret = lseek(device->xdma_h2c_fd, address, SEEK_SET);                       \
    if (ret < 0) {                                                             \
      return -NPU_DEVICE_ERRNO_IO;                                             \
    }                                                                          \
    uint8_t buf[sizeof(type)];                                                 \
    *(type *)buf = value;                                                      \
    ret = write(device->xdma_h2c_fd, buf, sizeof(type));                       \
    if (ret < 0) {                                                             \
      return -NPU_DEVICE_ERRNO_IO;                                             \
    }                                                                          \
    if (ret != sizeof(type)) {                                                 \
      return -NPU_DEVICE_ERRNO_IO;                                             \
    }                                                                          \
    return ret;                                                                \
  }

FUNC_NPU_MMIO_READ(uint64_t, 64)
FUNC_NPU_MMIO_READ(uint32_t, 32)
FUNC_NPU_MMIO_READ(uint16_t, 16)
FUNC_NPU_MMIO_READ(uint8_t, 8)
FUNC_NPU_MMIO_WRITE(uint64_t, 64)
FUNC_NPU_MMIO_WRITE(uint32_t, 32)
FUNC_NPU_MMIO_WRITE(uint16_t, 16)
FUNC_NPU_MMIO_WRITE(uint8_t, 8)

static char __attribute__ ((aligned (65536))) __npu_memcpy_buffer[65536];
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

ssize_t npu_memcpy_to_dev(const struct NPUDeviceState *device, uint64_t dst,
                          const void *src, size_t size) {
  if (device == NULL) {
    return -NPU_DEVICE_ERRNO_INVALID_ARGS;
  }

  if (src == NULL) {
    return -NPU_DEVICE_ERRNO_INVALID_ADDRESS;
  }

  if (!device->valid) {
    return -NPU_DEVICE_ERRNO_UNINITIALIZED;
  }

  if (size == 0) {
    return -NPU_DEVICE_ERRNO_INVALID_SIZE;
  }

  ssize_t ret;

  uint64_t wsize = 0;
  while(wsize != size) {
    ret = lseek(device->xdma_h2c_fd, dst + wsize, SEEK_SET);
    if (ret < 0) {
      return -NPU_DEVICE_ERRNO_IO;
    }
    memcpy(__npu_memcpy_buffer, src + wsize, MIN(65536, size - wsize));
    ret = write(device->xdma_h2c_fd, __npu_memcpy_buffer, MIN(65536, size - wsize));
    if (ret < 0) {
      return -NPU_DEVICE_ERRNO_IO;
    }
    wsize += ret;
  }

  return ret;
}

ssize_t npu_memcpy_from_dev(const struct NPUDeviceState *device, void *dst,
                            uint64_t src, size_t size) {
  if (device == NULL) {
    return -NPU_DEVICE_ERRNO_INVALID_ARGS;
  }

  if (dst == NULL) {
    return -NPU_DEVICE_ERRNO_INVALID_ADDRESS;
  }

  if (!device->valid) {
    return -NPU_DEVICE_ERRNO_UNINITIALIZED;
  }

  if (size == 0) {
    return -NPU_DEVICE_ERRNO_INVALID_SIZE;
  }

  ssize_t ret;

  uint64_t rsize = 0;
  while(rsize != size) {
    ret = lseek(device->xdma_c2h_fd, src + rsize, SEEK_SET);
    if (ret < 0) {
      return -NPU_DEVICE_ERRNO_IO;
    }
    ret = read(device->xdma_c2h_fd, __npu_memcpy_buffer, MIN(65536, size - rsize));
    memcpy(dst + rsize, __npu_memcpy_buffer, MIN(65536, size - rsize));
    if (ret < 0) {
      return -NPU_DEVICE_ERRNO_IO;
    }
    rsize += ret;
  }

  return ret;
}

// Check device state: WaitLaunch / Busy / Empty
ssize_t npu_check(const struct NPUDeviceState *device) {

  if (device == NULL) {
    return -NPU_DEVICE_ERRNO_INVALID_ARGS;
  }

  if (!device->valid) {
    return -NPU_DEVICE_ERRNO_UNINITIALIZED;
  }

  ssize_t ret;
  uint64_t boot_address;
  ret = npu_mmio_read64(device, MMIO_ADDR_BOOT_ADDR_REG, &boot_address);
  if (ret < 0) {
    return ret;
  }
  if(boot_address == 0x0) {
    // Busy
    return -NPU_DEVICE_ERRNO_BUSY;
  } else if(boot_address == 0x10000) {
    // Empty
    return -NPU_DEVICE_ERRNO_EMPTY;
  } else {
    // Kernel not launch yet
    return -NPU_DEVICE_ERRNO_WAITLAUNCH;
  }
}

// NPU_DEVICE_ERRNO_EMPTY -> WaitLaunch
ssize_t npu_start(const struct NPUDeviceState *device, uint64_t boot_address) {
  if (device == NULL) {
    return -NPU_DEVICE_ERRNO_INVALID_ARGS;
  }

  if (!device->valid) {
    return -NPU_DEVICE_ERRNO_UNINITIALIZED;
  }

  ssize_t ret;
  ret = npu_check(device);
  if(ret != -NPU_DEVICE_ERRNO_EMPTY) {
    return ret;
  }

  ret = npu_mmio_write64(device, MMIO_ADDR_BOOT_ADDR_REG, boot_address);
  if (ret < 0) {
    return ret;
  }

  // write clint msip to start
  ret = npu_mmio_write32(device, MMIO_ADDR_CLINT, 1);
  return ret;
}

// Any state -> NPU_DEVICE_ERRNO_EMPTY
ssize_t npu_abort(const struct NPUDeviceState *device) {
  // write the boot address to the start register

  if (device == NULL) {
    return -NPU_DEVICE_ERRNO_INVALID_ARGS;
  }

  if (!device->valid) {
    return -NPU_DEVICE_ERRNO_UNINITIALIZED;
  }

  ssize_t ret;

  ret = npu_mmio_write64(device, MMIO_ADDR_BOOT_ADDR_REG, 0x10000);
  if (ret < 0) {
    return ret;
  }

  // write clint msip to start
  ret = npu_mmio_write32(device, MMIO_ADDR_CLINT, 1);
  return ret;
}

#ifdef __cplusplus
}
#endif
