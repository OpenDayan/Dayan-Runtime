#define _POSIX_C_SOURCE 200809L

#include "NPUDriver.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define BOOT_MAILBOX_ADDR 0x1000
#define CLINT_MSIP_ADDR 0x20000
#define ACK_DELAY_NS (50L * 1000L * 1000L)
#define ACK_POLL_NS (100L * 1000L)
#define ACK_POLL_LIMIT 10000

struct ack_context {
  int fd;
  int saw_launch;
};

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

static void *acknowledge_abort(void *opaque) {
  struct ack_context *context = opaque;
  const struct timespec delay = {.tv_sec = 0, .tv_nsec = ACK_DELAY_NS};
  const struct timespec poll_delay = {.tv_sec = 0, .tv_nsec = ACK_POLL_NS};
  uint32_t launch = 0;
  uint32_t cleared = 0;

  for (int attempt = 0; attempt < ACK_POLL_LIMIT; ++attempt) {
    if (read_exact_at(context->fd, &launch, sizeof(launch),
                      CLINT_MSIP_ADDR) < 0) {
      return (void *)1;
    }
    if (launch == 1) {
      context->saw_launch = 1;
      break;
    }
    if (attempt + 1 == ACK_POLL_LIMIT) {
      fprintf(stderr, "abort MSIP was never written\n");
      return (void *)1;
    }
    while (nanosleep(&poll_delay, NULL) < 0 && errno == EINTR) {
    }
  }

  while (nanosleep(&delay, NULL) < 0 && errno == EINTR) {
  }
  if (write_exact_at(context->fd, &cleared, sizeof(cleared),
                     CLINT_MSIP_ADDR) < 0) {
    return (void *)1;
  }
  return NULL;
}

int main(void) {
  char device_path[] = "/tmp/rvnpu-device-XXXXXX";
  int device_fd = -1;
  pthread_t ack_thread;
  int thread_started = 0;
  int result = 1;
  uint32_t acknowledged = 1;
  uint64_t mailbox = 0;
  void *thread_result = NULL;

  device_fd = mkstemp(device_path);
  if (device_fd < 0) {
    fprintf(stderr, "mkstemp failed: errno=%d\n", errno);
    goto cleanup;
  }
  if (ftruncate(device_fd, CLINT_MSIP_ADDR + sizeof(uint32_t)) < 0) {
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
  struct ack_context context = {.fd = device_fd, .saw_launch = 0};

  if (pthread_create(&ack_thread, NULL, acknowledge_abort, &context) != 0) {
    fprintf(stderr, "pthread_create failed\n");
    goto cleanup;
  }
  thread_started = 1;

  ssize_t ret = npu_abort(&device);

  if (ret < 0) {
    fprintf(stderr, "npu_abort failed: %zd\n", ret);
    goto cleanup;
  }
  if (read_exact_at(device_fd, &acknowledged, sizeof(acknowledged),
                    CLINT_MSIP_ADDR) < 0 ||
      acknowledged != 0) {
    fprintf(stderr, "npu_abort returned before MSIP acknowledgement\n");
    goto cleanup;
  }
  pthread_join(ack_thread, &thread_result);
  thread_started = 0;
  if (thread_result != NULL || !context.saw_launch) {
    fprintf(stderr, "acknowledger did not observe the abort MSIP\n");
    goto cleanup;
  }
  if (read_exact_at(device_fd, &mailbox, sizeof(mailbox), BOOT_MAILBOX_ADDR) < 0 ||
      mailbox != 0x10000) {
    fprintf(stderr, "unexpected abort mailbox: 0x%llx\n",
            (unsigned long long)mailbox);
    goto cleanup;
  }
  result = 0;

cleanup:
  if (thread_started) {
    pthread_join(ack_thread, &thread_result);
    if (thread_result != NULL) {
      result = 1;
    }
  }
  if (device_fd >= 0) {
    close(device_fd);
  }
  unlink(device_path);

  if (result == 0) {
    puts("PASS: npu_abort waits for the ROM's MSIP acknowledgement");
  }
  return result;
}
