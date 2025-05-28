#ifndef _NPU_DRIVER_H_
#define _NPU_DRIVER_H_

#include <stdint.h>
#include <stdlib.h>

// Device State Structure
struct NPUDeviceState {
    // TODO: Define needed stateful data for the NPU Device here.
};

// Define Basic Driver API
void npu_device_init(struct NPUDeviceState* device);
void npu_device_deinit(struct NPUDeviceState* device);

void npu_mmio_read64(struct NPUDeviceState* device, uint64_t address, uint64_t *value);
void npu_mmio_write64(struct NPUDeviceState* device, uint64_t address, uint64_t value);
void npu_mmio_read32(struct NPUDeviceState* device, uint64_t address, uint32_t *value);
void npu_mmio_write32(struct NPUDeviceState* device, uint64_t address, uint32_t value);
void npu_mmio_read16(struct NPUDeviceState* device, uint64_t address, uint16_t *value);
void npu_mmio_write16(struct NPUDeviceState* device, uint64_t address, uint16_t value);
void npu_mmio_read8(struct NPUDeviceState* device, uint64_t address, uint8_t *value);
void npu_mmio_write8(struct NPUDeviceState* device, uint64_t address, uint8_t value);

void npu_memcpy_to_dev  (struct NPUDeviceState* device, uint64_t dst, void *src, size_t size);
void npu_memcpy_from_dev(struct NPUDeviceState* device, void *dst, uint64_t src, size_t size);

void npu_reset(struct NPUDeviceState* device);
void npu_start(struct NPUDeviceState* device, uint64_t boot_address);

#endif
