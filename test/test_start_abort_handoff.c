#define _POSIX_C_SOURCE 200809L

#include "NPUDriver.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

#define BOOT_MAILBOX_ADDR 0x1000
#define CLINT_MSIP_ADDR 0x20000
#define BOOT_MAILBOX_STRIDE 8
#define CLINT_MSIP_STRIDE 4
#define GENERAL_ENTRY UINT64_C(0x80000864)
#define CRT_DELAY_NS (50L * 1000L * 1000L)
#define POLL_DELAY_NS (100L * 1000L)
#define POLL_LIMIT 10000

struct device_context {
  int fd;
  off_t mailbox_addr;
  off_t clint_addr;
  int abort_published_before_crt_handoff;
  int abort_mailbox_overwritten;
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

static int wait_for_msip(int fd, off_t clint_addr, uint32_t expected) {
  const struct timespec delay = {.tv_sec = 0, .tv_nsec = POLL_DELAY_NS};
  for (int attempt = 0; attempt < POLL_LIMIT; ++attempt) {
    uint32_t msip = 0;
    if (read_exact_at(fd, &msip, sizeof(msip), clint_addr) < 0) {
      return -1;
    }
    if ((msip & 1u) == expected) {
      return 0;
    }
    while (nanosleep(&delay, NULL) < 0 && errno == EINTR) {
    }
  }
  fprintf(stderr, "timed out waiting for MSIP=%u\n", expected);
  return -1;
}

static void *run_device(void *opaque) {
  struct device_context *context = opaque;
  uint64_t mailbox = 0;
  uint32_t clear = 0;
  const struct timespec crt_delay = {.tv_sec = 0, .tv_nsec = CRT_DELAY_NS};

  if (wait_for_msip(context->fd, context->clint_addr, 1) < 0 ||
      read_exact_at(context->fd, &mailbox, sizeof(mailbox),
                    context->mailbox_addr) < 0 ||
      mailbox != GENERAL_ENTRY) {
    fprintf(stderr, "device did not receive the launch mailbox\n");
    return (void *)1;
  }

  /* ROM has latched the entry and now acknowledges the launch. */
  if (write_exact_at(context->fd, &clear, sizeof(clear),
                     context->clint_addr) < 0) {
    return (void *)1;
  }
  while (nanosleep(&crt_delay, NULL) < 0 && errno == EINTR) {
  }

  /* CRT has not written its busy state yet. A serialized abort must not have
   * reused the mailbox during this post-ack handoff window. */
  if (read_exact_at(context->fd, &mailbox, sizeof(mailbox),
                    context->mailbox_addr) < 0) {
    return (void *)1;
  }
  if (mailbox != GENERAL_ENTRY) {
    context->abort_published_before_crt_handoff = 1;
  }

  mailbox = 0;
  if (write_exact_at(context->fd, &mailbox, sizeof(mailbox),
                     context->mailbox_addr) < 0) {
    return (void *)1;
  }

  if (wait_for_msip(context->fd, context->clint_addr, 1) < 0 ||
      read_exact_at(context->fd, &mailbox, sizeof(mailbox),
                    context->mailbox_addr) < 0) {
    return (void *)1;
  }
  if (mailbox != 0x10000) {
    context->abort_mailbox_overwritten = 1;
  }
  if (write_exact_at(context->fd, &clear, sizeof(clear),
                     context->clint_addr) < 0) {
    return (void *)1;
  }
  return NULL;
}

static int run_case(int core_id) {
  char device_path[] = "/tmp/rvnpu-start-abort-XXXXXX";
  int device_fd = -1;
  int result = 1;
  pthread_t device_thread;
  int thread_started = 0;
  void *thread_result = NULL;
  uint64_t idle = 0x10000;
  uint32_t clear = 0;
  off_t mailbox_addr = BOOT_MAILBOX_ADDR + core_id * BOOT_MAILBOX_STRIDE;
  off_t clint_addr = CLINT_MSIP_ADDR + core_id * CLINT_MSIP_STRIDE;

  device_fd = mkstemp(device_path);
  if (device_fd < 0 ||
      ftruncate(device_fd, clint_addr + sizeof(clear)) < 0 ||
      write_exact_at(device_fd, &idle, sizeof(idle), mailbox_addr) < 0 ||
      write_exact_at(device_fd, &clear, sizeof(clear), clint_addr) < 0) {
    fprintf(stderr, "failed to initialize fake device: errno=%d\n", errno);
    goto cleanup;
  }

  struct NPUDeviceState device = {
      .xdma_id = 0,
      .core_id = core_id,
      .xdma_c2h_fd = device_fd,
      .xdma_h2c_fd = device_fd,
      .xdma_user_fd = -1,
      .xdma_event_fd = -1,
      .valid = 1,
  };
  struct device_context context = {
      .fd = device_fd,
      .mailbox_addr = mailbox_addr,
      .clint_addr = clint_addr,
      .abort_published_before_crt_handoff = 0,
      .abort_mailbox_overwritten = 0,
  };

  if (pthread_create(&device_thread, NULL, run_device, &context) != 0) {
    fprintf(stderr, "pthread_create failed\n");
    goto cleanup;
  }
  thread_started = 1;

  ssize_t ret = npu_start(&device, GENERAL_ENTRY);
  if (ret < 0) {
    fprintf(stderr, "npu_start failed: %zd\n", ret);
    goto cleanup;
  }
  ret = npu_abort(&device);
  if (ret < 0) {
    fprintf(stderr, "npu_abort failed: %zd\n", ret);
    goto cleanup;
  }

  pthread_join(device_thread, &thread_result);
  thread_started = 0;
  if (thread_result != NULL || context.abort_published_before_crt_handoff ||
      context.abort_mailbox_overwritten) {
    fprintf(stderr,
            "start/abort mailbox handoff was unsafe: early=%d overwritten=%d\n",
            context.abort_published_before_crt_handoff,
            context.abort_mailbox_overwritten);
    goto cleanup;
  }

  result = 0;

cleanup:
  if (thread_started) {
    pthread_join(device_thread, &thread_result);
  }
  if (device_fd >= 0) {
    close(device_fd);
  }
  unlink(device_path);
  return result;
}

int main(void) {
  if (run_case(0) != 0 || run_case(1) != 0) {
    return 1;
  }
  puts("PASS: every core waits for CRT handoff before a serialized abort");
  return 0;
}
