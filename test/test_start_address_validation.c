#define _POSIX_C_SOURCE 200809L

#include "NPUDriver.h"

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>

#define BOOT_MAILBOX_ADDR 0x1000
#define CLINT_MSIP_ADDR 0x20000

static int write_exact_at(int fd, const void *buf, size_t size, off_t offset) {
  ssize_t written = pwrite(fd, buf, size, offset);
  if (written != (ssize_t)size) {
    fprintf(stderr, "pwrite at 0x%llx failed: ret=%zd errno=%d\n",
            (unsigned long long)offset, written, errno);
    return -1;
  }
  return 0;
}

static int read_exact_at(int fd, void *buf, size_t size, off_t offset) {
  ssize_t received = pread(fd, buf, size, offset);
  if (received != (ssize_t)size) {
    fprintf(stderr, "pread at 0x%llx failed: ret=%zd errno=%d\n",
            (unsigned long long)offset, received, errno);
    return -1;
  }
  return 0;
}

static int run_case(struct NPUDeviceState *device, uint64_t reserved_address) {
  uint64_t idle = 0x10000;
  uint64_t mailbox = 0;
  uint32_t clear = 0;
  uint32_t msip = 1;

  if (write_exact_at(device->xdma_c2h_fd, &idle, sizeof(idle),
                     BOOT_MAILBOX_ADDR) < 0 ||
      write_exact_at(device->xdma_c2h_fd, &clear, sizeof(clear),
                     CLINT_MSIP_ADDR) < 0) {
    return 1;
  }

  ssize_t ret = npu_start(device, reserved_address);
  if (ret != -NPU_DEVICE_ERRNO_INVALID_ADDRESS) {
    fprintf(stderr, "start(0x%llx): expected invalid address, got %zd\n",
            (unsigned long long)reserved_address, ret);
    return 1;
  }
  if (read_exact_at(device->xdma_c2h_fd, &mailbox, sizeof(mailbox),
                    BOOT_MAILBOX_ADDR) < 0 ||
      read_exact_at(device->xdma_c2h_fd, &msip, sizeof(msip),
                    CLINT_MSIP_ADDR) < 0) {
    return 1;
  }
  if (mailbox != idle || msip != 0) {
    fprintf(stderr,
            "start(0x%llx) changed MMIO: mailbox=0x%llx msip=0x%x\n",
            (unsigned long long)reserved_address,
            (unsigned long long)mailbox, msip);
    return 1;
  }
  return 0;
}

int main(void) {
  char device_path[] = "/tmp/rvnpu-start-address-XXXXXX";
  int device_fd = mkstemp(device_path);
  int result = 1;
  uint32_t clear = 0;

  if (device_fd < 0 ||
      ftruncate(device_fd, CLINT_MSIP_ADDR + sizeof(clear)) < 0) {
    fprintf(stderr, "failed to initialize fake device: errno=%d\n", errno);
    goto cleanup;
  }

  struct NPUDeviceState device = {
      .xdma_id = 0,
      .core_id = 0,
      .xdma_c2h_fd = device_fd,
      .xdma_h2c_fd = device_fd,
      .xdma_user_fd = -1,
      .xdma_event_fd = -1,
      .valid = 1,
  };
  if (run_case(&device, 0) != 0 || run_case(&device, 0x10000) != 0) {
    goto cleanup;
  }

  result = 0;

cleanup:
  if (device_fd >= 0) {
    close(device_fd);
  }
  unlink(device_path);
  if (result == 0) {
    puts("PASS: npu_start rejects reserved mailbox state values");
  }
  return result;
}
