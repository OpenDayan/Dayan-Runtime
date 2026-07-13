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
#define OBSERVATION_DELAY_NS (50L * 1000L * 1000L)
#define SENTINEL_MAILBOX UINT64_C(0xfeedfacecafebeef)

struct quiescence_context {
  int c2h_fd;
  int h2c_fd;
  int command_published_early;
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

static void *release_prior_command(void *opaque) {
  struct quiescence_context *context = opaque;
  const struct timespec delay = {.tv_sec = 0, .tv_nsec = OBSERVATION_DELAY_NS};
  uint64_t mailbox = 0;
  uint32_t msip = 0;
  uint32_t cleared = 0;

  while (nanosleep(&delay, NULL) < 0 && errno == EINTR) {
  }
  if (read_exact_at(context->h2c_fd, &mailbox, sizeof(mailbox),
                    BOOT_MAILBOX_ADDR) < 0 ||
      read_exact_at(context->h2c_fd, &msip, sizeof(msip),
                    CLINT_MSIP_ADDR) < 0) {
    return (void *)1;
  }
  if (mailbox != SENTINEL_MAILBOX || msip != 0) {
    context->command_published_early = 1;
  }
  if (write_exact_at(context->c2h_fd, &cleared, sizeof(cleared),
                     CLINT_MSIP_ADDR) < 0) {
    return (void *)1;
  }
  return NULL;
}

static int make_device_files(int *c2h_fd, int *h2c_fd,
                             char c2h_path[], char h2c_path[]) {
  uint32_t prior_msip = 1;
  uint32_t no_command = 0;
  uint64_t sentinel_mailbox = SENTINEL_MAILBOX;

  *c2h_fd = mkstemp(c2h_path);
  *h2c_fd = mkstemp(h2c_path);
  if (*c2h_fd < 0 || *h2c_fd < 0) {
    return -1;
  }
  if (ftruncate(*c2h_fd, CLINT_MSIP_ADDR + sizeof(prior_msip)) < 0 ||
      ftruncate(*h2c_fd, CLINT_MSIP_ADDR + sizeof(no_command)) < 0 ||
      write_exact_at(*c2h_fd, &prior_msip, sizeof(prior_msip),
                     CLINT_MSIP_ADDR) < 0 ||
      write_exact_at(*h2c_fd, &sentinel_mailbox, sizeof(sentinel_mailbox),
                     BOOT_MAILBOX_ADDR) < 0 ||
      write_exact_at(*h2c_fd, &no_command, sizeof(no_command),
                     CLINT_MSIP_ADDR) < 0) {
    return -1;
  }
  return 0;
}

static int run_case(int start_command) {
  char c2h_path[] = "/tmp/rvnpu-c2h-quiescence-XXXXXX";
  char h2c_path[] = "/tmp/rvnpu-h2c-quiescence-XXXXXX";
  int c2h_fd = -1;
  int h2c_fd = -1;
  int result = 1;
  pthread_t release_thread;
  int thread_started = 0;
  void *thread_result = NULL;
  uint64_t empty_mailbox = 0x10000;
  uint64_t expected_mailbox = start_command ? UINT64_C(0x80000864) : 0x10000;
  uint64_t written_mailbox = 0;
  uint32_t written_msip = 0;

  if (make_device_files(&c2h_fd, &h2c_fd, c2h_path, h2c_path) < 0) {
    fprintf(stderr, "failed to create fake device files: errno=%d\n", errno);
    goto cleanup;
  }
  if (write_exact_at(c2h_fd, &empty_mailbox, sizeof(empty_mailbox),
                     BOOT_MAILBOX_ADDR) < 0) {
    goto cleanup;
  }

  struct NPUDeviceState device = {
      .xdma_id = 0,
      .core_id = 0,
      .xdma_c2h_fd = c2h_fd,
      .xdma_h2c_fd = h2c_fd,
      .xdma_user_fd = -1,
      .xdma_event_fd = -1,
      .valid = 1,
  };
  struct quiescence_context context = {
      .c2h_fd = c2h_fd,
      .h2c_fd = h2c_fd,
      .command_published_early = 0,
  };

  if (pthread_create(&release_thread, NULL, release_prior_command, &context) != 0) {
    fprintf(stderr, "pthread_create failed\n");
    goto cleanup;
  }
  thread_started = 1;

  ssize_t ret = start_command ? npu_start(&device, expected_mailbox)
                              : npu_abort(&device);
  if (ret < 0) {
    fprintf(stderr, "%s failed: %zd\n", start_command ? "npu_start" : "npu_abort",
            ret);
    goto cleanup;
  }

  pthread_join(release_thread, &thread_result);
  thread_started = 0;
  if (thread_result != NULL) {
    goto cleanup;
  }
  if (context.command_published_early) {
    fprintf(stderr, "%s published while a prior MSIP was still pending\n",
            start_command ? "npu_start" : "npu_abort");
    goto cleanup;
  }
  if (read_exact_at(h2c_fd, &written_mailbox, sizeof(written_mailbox),
                    BOOT_MAILBOX_ADDR) < 0 ||
      written_mailbox != expected_mailbox) {
    fprintf(stderr, "unexpected command mailbox: 0x%llx\n",
            (unsigned long long)written_mailbox);
    goto cleanup;
  }
  if (read_exact_at(h2c_fd, &written_msip, sizeof(written_msip),
                    CLINT_MSIP_ADDR) < 0 ||
      written_msip != 1) {
    fprintf(stderr, "unexpected command MSIP: 0x%x\n", written_msip);
    goto cleanup;
  }

  result = 0;

cleanup:
  if (thread_started) {
    pthread_join(release_thread, &thread_result);
  }
  if (c2h_fd >= 0) {
    close(c2h_fd);
  }
  if (h2c_fd >= 0) {
    close(h2c_fd);
  }
  unlink(c2h_path);
  unlink(h2c_path);
  return result;
}

int main(void) {
  if (run_case(0) != 0 || run_case(1) != 0) {
    return 1;
  }
  puts("PASS: mailbox commands wait for a quiescent MSIP");
  return 0;
}
