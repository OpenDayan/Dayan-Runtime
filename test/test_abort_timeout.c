#define _POSIX_C_SOURCE 200809L

#include "NPUDriver.h"

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

#define BOOT_MAILBOX_ADDR 0x1000
#define CLINT_MSIP_ADDR 0x20000

static int read_exact_at(int fd, void *buf, size_t size, off_t offset) {
  ssize_t received = pread(fd, buf, size, offset);
  if (received != (ssize_t)size) {
    fprintf(stderr, "pread at 0x%llx failed: ret=%zd errno=%d\n",
            (unsigned long long)offset, received, errno);
    return -1;
  }
  return 0;
}

int main(void) {
  char device_path[] = "/tmp/rvnpu-device-timeout-XXXXXX";
  int device_fd = -1;
  int result = 1;
  uint32_t launch = 0;
  uint64_t mailbox = 0;
  struct timespec start;
  struct timespec finish;

  device_fd = mkstemp(device_path);
  if (device_fd < 0) {
    fprintf(stderr, "mkstemp failed: errno=%d\n", errno);
    goto cleanup;
  }
  if (ftruncate(device_fd, CLINT_MSIP_ADDR + sizeof(launch)) < 0) {
    fprintf(stderr, "ftruncate failed: errno=%d\n", errno);
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

  clock_gettime(CLOCK_MONOTONIC, &start);
  ssize_t ret = npu_abort(&device);
  clock_gettime(CLOCK_MONOTONIC, &finish);
  if (ret != -NPU_DEVICE_ERRNO_TIMEOUT) {
    fprintf(stderr, "expected abort timeout %d, got %zd\n",
            -NPU_DEVICE_ERRNO_TIMEOUT, ret);
    goto cleanup;
  }
  int64_t elapsed_ms = (finish.tv_sec - start.tv_sec) * 1000 +
                       (finish.tv_nsec - start.tv_nsec) / 1000000;
  if (elapsed_ms < 5 || elapsed_ms >= 1000) {
    fprintf(stderr, "abort timeout was not bounded as configured: %lld ms\n",
            (long long)elapsed_ms);
    goto cleanup;
  }
  if (read_exact_at(device_fd, &mailbox, sizeof(mailbox), BOOT_MAILBOX_ADDR) < 0 ||
      mailbox != 0x10000) {
    fprintf(stderr, "unexpected abort mailbox: 0x%llx\n",
            (unsigned long long)mailbox);
    goto cleanup;
  }
  if (read_exact_at(device_fd, &launch, sizeof(launch), CLINT_MSIP_ADDR) < 0 ||
      launch != 1) {
    fprintf(stderr, "unexpected launch MSIP: 0x%x\n", launch);
    goto cleanup;
  }

  result = 0;

cleanup:
  if (device_fd >= 0) {
    close(device_fd);
  }
  unlink(device_path);

  if (result == 0) {
    puts("PASS: npu_abort reports a bounded acknowledgement timeout");
  }
  return result;
}
