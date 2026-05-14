#include <string.h>
#include <sys/types.h>
#include <stdarg.h>
#include <unistd.h>
#include <dlfcn.h>
#include <elf.h>
#include <limits.h>
#include <dirent.h>

#include "NPUDriver.h"

#include "NPURuntime.h"

#ifdef __cplusplus
extern "C" {
#endif

// Keep link compatibility for binaries that still reference this symbol directly.
__attribute__((weak)) struct NPUDeviceKernelSetDesc __bin_kernels_pack = {0};

static ssize_t npu_runtime_build_kernelset_from_elf_image(
    struct NPURuntimeState *runtime,
    const struct NPUKernelELFImage *image,
    struct NPUDeviceKernelSetDesc *out_desc,
    uint8_t **out_data,
    struct NPUKernelDesc **out_kernels,
    char ***out_names,
    size_t *out_name_count);

static void npu_runtime_release_owned_kernelset(struct NPURuntimeState *runtime) {
  if (runtime == NULL) {
    return;
  }
  if (runtime->owned_kernel_names) {
    for (size_t i = 0; i < runtime->owned_kernel_name_count; i++) {
      free(runtime->owned_kernel_names[i]);
    }
    free(runtime->owned_kernel_names);
    runtime->owned_kernel_names = NULL;
  }
  if (runtime->owned_kernel_descs) {
    free(runtime->owned_kernel_descs);
    runtime->owned_kernel_descs = NULL;
  }
  if (runtime->owned_kernelset_data) {
    free(runtime->owned_kernelset_data);
    runtime->owned_kernelset_data = NULL;
  }
  runtime->owned_kernel_name_count = 0;
}

static void npu_runtime_dump_kernelset_brief(const char *tag,
                                             const struct NPUDeviceKernelSetDesc *desc) {
  const char *debug = getenv("NPU_RUNTIME_DEBUG_DESC");
  if (debug == NULL || desc == NULL || desc->data == NULL) {
    return;
  }
  printf("[npu-runtime][%s] load_start=0x%lx load_size=%lu kernel_count=%lu general_entry=0x%lx args_ptr=0x%lx\n",
         tag,
         (unsigned long)desc->load_start,
         (unsigned long)desc->load_size,
         (unsigned long)desc->kernel_count,
         (unsigned long)desc->general_entry,
         (unsigned long)desc->args_ptr);
  printf("[npu-runtime][%s] data[0..7]=%u %u %u %u %u %u %u %u\n",
         tag,
         (unsigned)desc->data[0], (unsigned)desc->data[1], (unsigned)desc->data[2], (unsigned)desc->data[3],
         (unsigned)desc->data[4], (unsigned)desc->data[5], (unsigned)desc->data[6], (unsigned)desc->data[7]);
  if (desc->kernel_count > 0 && desc->kernels != NULL && desc->kernels[0].name != NULL) {
    printf("[npu-runtime][%s] first_kernel=%s entry=0x%lx\n",
           tag, desc->kernels[0].name, (unsigned long)desc->kernels[0].entry);
  }
}

static int npu_runtime_probe_so_for_kernel(struct NPURuntimeState *runtime,
                                           const char *so_path,
                                           const char *kernel_name) {
  typedef const struct NPUKernelELFImage *(*get_elf_fn_t)(void);

  void *handle = dlopen(so_path, RTLD_NOW | RTLD_LOCAL);
  if (handle == NULL) {
    return 0;
  }

  dlerror();
  get_elf_fn_t get_elf_fn = (get_elf_fn_t)dlsym(handle, "npu_get_kernel_elf_image");
  const char *err = dlerror();
  if (err || get_elf_fn == NULL) {
    dlclose(handle);
    return 0;
  }

  const struct NPUKernelELFImage *elf_image = get_elf_fn();
  if (elf_image == NULL || elf_image->data == NULL || elf_image->size == 0) {
    dlclose(handle);
    return 0;
  }

  struct NPUDeviceKernelSetDesc tmp_desc = {0};
  uint8_t *tmp_data = NULL;
  struct NPUKernelDesc *tmp_kernels = NULL;
  char **tmp_names = NULL;
  size_t tmp_name_count = 0;
  ssize_t ret = npu_runtime_build_kernelset_from_elf_image(
      runtime, elf_image, &tmp_desc, &tmp_data, &tmp_kernels, &tmp_names, &tmp_name_count);
  if (ret != NPU_SUCCESS) {
    dlclose(handle);
    return 0;
  }

  int found = 0;
  for (size_t i = 0; i < tmp_desc.kernel_count; i++) {
    const char *name = tmp_desc.kernels[i].name;
    if (name != NULL && strcmp(name, kernel_name) == 0) {
      found = 1;
      break;
    }
  }

  for (size_t i = 0; i < tmp_name_count; i++) {
    free(tmp_names[i]);
  }
  free(tmp_names);
  free(tmp_kernels);
  free(tmp_data);
  dlclose(handle);
  return found;
}

static ssize_t npu_runtime_try_load_kernel_by_name(
    struct NPURuntimeState *runtime,
    const char *kernel_name) {
  if (runtime == NULL || kernel_name == NULL || kernel_name[0] == '\0') {
    return NPU_ERROR_INVALID_ARGUMENT;
  }

  // Prevent path traversal; runtime resolves by logical kernel name only.
  if (strchr(kernel_name, '/') != NULL || strchr(kernel_name, '\\') != NULL) {
    return NPU_ERROR_INVALID_ARGUMENT;
  }

  const char *kernel_so_dir = getenv("NPU_KERNEL_SO_DIR");
  if (kernel_so_dir == NULL || kernel_so_dir[0] == '\0') {
    kernel_so_dir = "/workspace/npu-kernels/bin";
  }

  char so_path[PATH_MAX];
  int written = snprintf(so_path, sizeof(so_path), "%s/%s.so", kernel_so_dir, kernel_name);
  if (written <= 0 || (size_t)written >= sizeof(so_path)) {
    return NPU_ERROR_INVALID_ARGUMENT;
  }

  if (npu_runtime_probe_so_for_kernel(runtime, so_path, kernel_name)) {
    return npu_runtime_load_kernelset(runtime, so_path);
  }

  DIR *dir = opendir(kernel_so_dir);
  if (dir == NULL) {
    return NPU_ERROR_KERNEL_NOT_FOUND;
  }

  struct dirent *ent = NULL;
  while ((ent = readdir(dir)) != NULL) {
    const char *name = ent->d_name;
    size_t len = strlen(name);
    if (len < 4 || strcmp(name + len - 3, ".so") != 0) {
      continue;
    }

    int so_written = snprintf(so_path, sizeof(so_path), "%s/%s", kernel_so_dir, name);
    if (so_written <= 0 || (size_t)so_written >= sizeof(so_path)) {
      continue;
    }

    if (!npu_runtime_probe_so_for_kernel(runtime, so_path, kernel_name)) {
      continue;
    }

    closedir(dir);
    return npu_runtime_load_kernelset(runtime, so_path);
  }

  closedir(dir);
  return NPU_ERROR_KERNEL_NOT_FOUND;
}

static ssize_t npu_runtime_build_kernelset_from_elf_image(
    struct NPURuntimeState *runtime,
    const struct NPUKernelELFImage *image,
    struct NPUDeviceKernelSetDesc *out_desc,
    uint8_t **out_data,
    struct NPUKernelDesc **out_kernels,
    char ***out_names,
    size_t *out_name_count) {
  if (runtime == NULL || image == NULL || out_desc == NULL || out_data == NULL ||
      out_kernels == NULL || out_names == NULL || out_name_count == NULL) {
    return NPU_ERROR_INVALID_ARGUMENT;
  }
  if (image->data == NULL || image->size < sizeof(Elf64_Ehdr)) {
    return NPU_ERROR_INVALID_ARGUMENT;
  }

  const uint8_t *bytes = image->data;
  const Elf64_Ehdr *eh = (const Elf64_Ehdr *)bytes;
  if (!(eh->e_ident[EI_MAG0] == ELFMAG0 &&
        eh->e_ident[EI_MAG1] == ELFMAG1 &&
        eh->e_ident[EI_MAG2] == ELFMAG2 &&
        eh->e_ident[EI_MAG3] == ELFMAG3 &&
        eh->e_ident[EI_CLASS] == ELFCLASS64)) {
    return NPU_ERROR_INVALID_ARGUMENT;
  }
  if (eh->e_phoff == 0 || eh->e_phnum == 0 ||
      eh->e_phoff + (size_t)eh->e_phnum * sizeof(Elf64_Phdr) > image->size) {
    return NPU_ERROR_INVALID_ARGUMENT;
  }

  const Elf64_Phdr *ph = (const Elf64_Phdr *)(bytes + eh->e_phoff);
  uint64_t load_start = UINT64_MAX;
  uint64_t load_end = 0;
  for (uint16_t i = 0; i < eh->e_phnum; i++) {
    if (ph[i].p_type != PT_LOAD) {
      continue;
    }
    if (ph[i].p_paddr < load_start) {
      load_start = ph[i].p_paddr;
    }
    uint64_t seg_end = ph[i].p_paddr + ph[i].p_memsz;
    if (seg_end > load_end) {
      load_end = seg_end;
    }
  }
  if (load_start == UINT64_MAX || load_end <= load_start) {
    return NPU_ERROR_INVALID_ARGUMENT;
  }

  size_t image_size = (size_t)(load_end - load_start);
  uint8_t *flat_image = (uint8_t *)calloc(image_size, 1);
  if (!flat_image) {
    return NPU_ERROR_OUT_OF_MEMORY;
  }
  for (uint16_t i = 0; i < eh->e_phnum; i++) {
    if (ph[i].p_type != PT_LOAD || ph[i].p_filesz == 0) {
      continue;
    }
    if (ph[i].p_offset + ph[i].p_filesz > image->size) {
      free(flat_image);
      return NPU_ERROR_INVALID_ARGUMENT;
    }
    size_t dst_off = (size_t)(ph[i].p_paddr - load_start);
    if (dst_off + ph[i].p_filesz > image_size) {
      free(flat_image);
      return NPU_ERROR_INVALID_ARGUMENT;
    }
    memcpy(flat_image + dst_off, bytes + ph[i].p_offset, ph[i].p_filesz);
  }

  if (eh->e_shoff == 0 || eh->e_shnum == 0 ||
      eh->e_shoff + (size_t)eh->e_shnum * sizeof(Elf64_Shdr) > image->size) {
    free(flat_image);
    return NPU_ERROR_INVALID_ARGUMENT;
  }

  const Elf64_Shdr *sh = (const Elf64_Shdr *)(bytes + eh->e_shoff);
  if (eh->e_shstrndx >= eh->e_shnum) {
    free(flat_image);
    return NPU_ERROR_INVALID_ARGUMENT;
  }
  const Elf64_Shdr *shstr = &sh[eh->e_shstrndx];
  if (shstr->sh_offset + shstr->sh_size > image->size) {
    free(flat_image);
    return NPU_ERROR_INVALID_ARGUMENT;
  }
  const char *shstrtab = (const char *)(bytes + shstr->sh_offset);

  int symtab_idx = -1;
  int text_kernel_idx = -1;
  for (uint16_t i = 0; i < eh->e_shnum; i++) {
    const char *sec_name = (sh[i].sh_name < shstr->sh_size) ? (shstrtab + sh[i].sh_name) : "";
    if (strcmp(sec_name, ".symtab") == 0) {
      symtab_idx = (int)i;
    }
    if (strcmp(sec_name, ".text.npu_kernel") == 0) {
      text_kernel_idx = (int)i;
    }
  }
  if (symtab_idx < 0) {
    free(flat_image);
    return NPU_ERROR_INVALID_ARGUMENT;
  }

  const Elf64_Shdr *symtab_sh = &sh[symtab_idx];
  if (symtab_sh->sh_link >= eh->e_shnum ||
      symtab_sh->sh_offset + symtab_sh->sh_size > image->size ||
      symtab_sh->sh_entsize != sizeof(Elf64_Sym)) {
    free(flat_image);
    return NPU_ERROR_INVALID_ARGUMENT;
  }
  const Elf64_Shdr *strtab_sh = &sh[symtab_sh->sh_link];
  if (strtab_sh->sh_offset + strtab_sh->sh_size > image->size) {
    free(flat_image);
    return NPU_ERROR_INVALID_ARGUMENT;
  }

  const Elf64_Sym *syms = (const Elf64_Sym *)(bytes + symtab_sh->sh_offset);
  size_t sym_count = symtab_sh->sh_size / sizeof(Elf64_Sym);
  const char *strtab = (const char *)(bytes + strtab_sh->sh_offset);

  struct NPUKernelDesc *kernels = (struct NPUKernelDesc *)calloc(sym_count + 1, sizeof(struct NPUKernelDesc));
  char **names = (char **)calloc(sym_count + 1, sizeof(char *));
  if (!kernels || !names) {
    free(kernels);
    free(names);
    free(flat_image);
    return NPU_ERROR_OUT_OF_MEMORY;
  }

  uint64_t args_ptr = 0;
  uint64_t heap = 0;
  uint64_t perf = 0;
  size_t kernel_count = 0;
  for (size_t i = 0; i < sym_count; i++) {
    const Elf64_Sym *s = &syms[i];
    if (s->st_name >= strtab_sh->sh_size) {
      continue;
    }
    const char *name = strtab + s->st_name;
    if (name[0] == '\0') {
      continue;
    }
    if (strcmp(name, "entry_args") == 0) {
      args_ptr = s->st_value;
      continue;
    }
    if (strcmp(name, "__perf_counter") == 0) {
      perf = s->st_value;
      continue;
    }
    if (strcmp(name, "_heap") == 0) {
      heap = s->st_value;
      continue;
    }

    if (ELF64_ST_TYPE(s->st_info) != STT_FUNC || s->st_shndx == SHN_UNDEF) {
      continue;
    }
    if (text_kernel_idx >= 0) {
      if ((int)s->st_shndx != text_kernel_idx) {
        const char *sec_name = (s->st_shndx < eh->e_shnum && sh[s->st_shndx].sh_name < shstr->sh_size)
                                   ? (shstrtab + sh[s->st_shndx].sh_name)
                                   : "";
        if (strncmp(sec_name, ".text.npu_kernel", 16) != 0) {
          continue;
        }
      }
    }
    if (name[0] == '$' || name[0] == '.') {
      continue;
    }

    names[kernel_count] = strdup(name);
    if (!names[kernel_count]) {
      for (size_t j = 0; j < kernel_count; j++) {
        free(names[j]);
      }
      free(names);
      free(kernels);
      free(flat_image);
      return NPU_ERROR_OUT_OF_MEMORY;
    }
    kernels[kernel_count].entry = s->st_value;
    kernels[kernel_count].name = names[kernel_count];
    kernel_count++;
  }
  kernels[kernel_count].entry = 0;
  kernels[kernel_count].name = NULL;

  memset(out_desc, 0, sizeof(*out_desc));
  out_desc->load_start = load_start;
  out_desc->load_size = image_size;
  out_desc->kernel_count = kernel_count;
  out_desc->general_entry = eh->e_entry;
  out_desc->args_ptr = args_ptr;
  out_desc->data = flat_image;
  out_desc->kernels = kernels;
  out_desc->heap = heap;
  out_desc->perf = perf;

  *out_data = flat_image;
  *out_kernels = kernels;
  *out_names = names;
  *out_name_count = kernel_count;
  return NPU_SUCCESS;
}

static ssize_t npu_runtime_load_kernelset_desc(
    struct NPURuntimeState *runtime,
    const struct NPUDeviceKernelSetDesc *kernel_set_desc) {
  if (runtime == NULL || kernel_set_desc == NULL) {
    return NPU_ERROR_INVALID_ARGUMENT;
  }

  npu_runtime_dump_kernelset_brief("load", kernel_set_desc);
  npu_runtime_release_owned_kernelset(runtime);
  runtime->kernel_set_desc = *kernel_set_desc;

  ssize_t ret;

  ret = npu_abort(&runtime->dev_state);
  if (ret < 0) {
    printf("Failed to abort NPU device: %zd\n", ret);
    return NPU_ERROR_DEVICE_FAILURE;
  }

  // load bin
  ret = npu_memcpy_to_dev(&runtime->dev_state,
                          runtime->kernel_set_desc.load_start,
                          kernel_set_desc->data, kernel_set_desc->load_size);
  if (ret < 0) {
    return NPU_ERROR_DEVICE_FAILURE;
  }

  ret = npu_mm_init(&runtime->mm, kernel_set_desc->heap, 0x180000000ul);

  runtime->printchar_size = NPU_PRINTCHAR_BUF_SIZE;
  npu_runtime_malloc(runtime, &runtime->printchar_base, runtime->printchar_size);

  return NPU_SUCCESS;
}

ssize_t npu_runtime_init(struct NPURuntimeState *runtime) {
  ssize_t ret;
  if (runtime == NULL) {
    return NPU_ERROR_INVALID_ARGUMENT;
  }

  ret = npu_device_init(&runtime->dev_state);
  if (ret < 0) {
    printf("Failed to initialize NPU device: %zd\n", ret);
    return NPU_ERROR_DEVICE_FAILURE;
  }

  runtime->mm = NULL;
  memset(&runtime->kernel_set_desc, 0, sizeof(runtime->kernel_set_desc));
  runtime->kernelset_so_handle = NULL;
  runtime->owned_kernelset_data = NULL;
  runtime->owned_kernel_descs = NULL;
  runtime->owned_kernel_names = NULL;
  runtime->owned_kernel_name_count = 0;

  return NPU_SUCCESS;
}

void npu_runtime_deinit(struct NPURuntimeState *runtime) {
  npu_device_deinit(&runtime->dev_state);
  if(runtime->mm) {
    npu_mm_deinit(&runtime->mm);
  }
  npu_runtime_release_owned_kernelset(runtime);
  if (runtime->kernelset_so_handle) {
    dlclose(runtime->kernelset_so_handle);
    runtime->kernelset_so_handle = NULL;
  }
}

ssize_t npu_runtime_load_kernelset(
    struct NPURuntimeState *runtime,
    const char *so_path) {
  if (runtime == NULL) {
    return NPU_ERROR_INVALID_ARGUMENT;
  }

  const char *kernelset_so = so_path;
  if (kernelset_so == NULL || kernelset_so[0] == '\0') {
    // Pure lazy per-kernel mode: no eager pack loading.
    return NPU_SUCCESS;
  }

  void *new_handle = dlopen(kernelset_so, RTLD_NOW | RTLD_LOCAL);
  if (!new_handle) {
    printf("Failed to load kernelset shared object '%s': %s\n",
           kernelset_so, dlerror());
    return NPU_ERROR_DEVICE_FAILURE;
  }

  const struct NPUDeviceKernelSetDesc *kernel_set_desc = NULL;
  const char *resolved_symbol_name = "__bin_kernels_pack";

  dlerror();
  kernel_set_desc = (const struct NPUDeviceKernelSetDesc *)dlsym(new_handle, resolved_symbol_name);
  const char *sym_err = dlerror();

  // Backward-compatible fallback: function-export style (descriptor).
  if (sym_err) {
    typedef const struct NPUDeviceKernelSetDesc *(*get_desc_fn_t)(void);
    dlerror();
    get_desc_fn_t get_desc_fn = (get_desc_fn_t)dlsym(new_handle, "npu_get_kernelset_desc");
    const char *fn_err = dlerror();
    if (!fn_err && get_desc_fn) {
      kernel_set_desc = get_desc_fn();
      sym_err = NULL;
    }
  }

  if (sym_err || kernel_set_desc == NULL) {
    // New path: ELF image-export style.
    typedef const struct NPUKernelELFImage *(*get_elf_fn_t)(void);
    dlerror();
    get_elf_fn_t get_elf_fn = (get_elf_fn_t)dlsym(new_handle, "npu_get_kernel_elf_image");
    const char *elf_err = dlerror();
    if (elf_err || get_elf_fn == NULL) {
      printf("Failed to resolve kernelset symbol in '%s': %s\n",
             kernelset_so, sym_err ? sym_err : "NULL descriptor");
      dlclose(new_handle);
      return NPU_ERROR_DEVICE_FAILURE;
    }

    const struct NPUKernelELFImage *elf_image = get_elf_fn();
    if (elf_image == NULL || elf_image->data == NULL || elf_image->size == 0) {
      printf("Invalid kernel ELF image from '%s'\n", kernelset_so);
      dlclose(new_handle);
      return NPU_ERROR_DEVICE_FAILURE;
    }

    struct NPUDeviceKernelSetDesc elf_desc = {0};
    uint8_t *data_copy = NULL;
    struct NPUKernelDesc *kernels_copy = NULL;
    char **names_copy = NULL;
    size_t names_count = 0;
    ssize_t build_ret = npu_runtime_build_kernelset_from_elf_image(
        runtime, elf_image, &elf_desc, &data_copy, &kernels_copy, &names_copy, &names_count);
    if (build_ret != NPU_SUCCESS) {
      dlclose(new_handle);
      return build_ret;
    }

    npu_runtime_release_owned_kernelset(runtime);
    ssize_t ret = npu_runtime_load_kernelset_desc(runtime, &elf_desc);
    if (ret != NPU_SUCCESS) {
      for (size_t i = 0; i < names_count; i++) {
        free(names_copy[i]);
      }
      free(names_copy);
      free(kernels_copy);
      free(data_copy);
      dlclose(new_handle);
      return ret;
    }

    runtime->owned_kernelset_data = data_copy;
    runtime->owned_kernel_descs = kernels_copy;
    runtime->owned_kernel_names = names_copy;
    runtime->owned_kernel_name_count = names_count;
    if (runtime->kernelset_so_handle) {
      dlclose(runtime->kernelset_so_handle);
    }
    runtime->kernelset_so_handle = new_handle;
    return NPU_SUCCESS;
  }

  if (kernel_set_desc->data == NULL || kernel_set_desc->kernels == NULL ||
      kernel_set_desc->load_size == 0 || kernel_set_desc->kernel_count == 0) {
    printf("Invalid kernelset descriptor from '%s'\n", kernelset_so);
    dlclose(new_handle);
    return NPU_ERROR_INVALID_ARGUMENT;
  }

  uint8_t *data_copy = (uint8_t *)malloc(kernel_set_desc->load_size);
  struct NPUKernelDesc *kernels_copy =
      (struct NPUKernelDesc *)calloc(kernel_set_desc->kernel_count + 1,
                                     sizeof(struct NPUKernelDesc));
  char **names_copy = (char **)calloc(kernel_set_desc->kernel_count, sizeof(char *));
  if (data_copy == NULL || kernels_copy == NULL || names_copy == NULL) {
    free(data_copy);
    free(kernels_copy);
    free(names_copy);
    dlclose(new_handle);
    return NPU_ERROR_OUT_OF_MEMORY;
  }

  memcpy(data_copy, kernel_set_desc->data, kernel_set_desc->load_size);
  for (size_t i = 0; i < kernel_set_desc->kernel_count; i++) {
    kernels_copy[i].entry = kernel_set_desc->kernels[i].entry;
    if (kernel_set_desc->kernels[i].name) {
      names_copy[i] = strdup(kernel_set_desc->kernels[i].name);
      kernels_copy[i].name = names_copy[i];
    } else {
      kernels_copy[i].name = NULL;
    }
  }
  kernels_copy[kernel_set_desc->kernel_count].entry = 0;
  kernels_copy[kernel_set_desc->kernel_count].name = NULL;

  struct NPUDeviceKernelSetDesc copied_desc = *kernel_set_desc;
  copied_desc.data = data_copy;
  copied_desc.kernels = kernels_copy;
  npu_runtime_dump_kernelset_brief("load-so", &copied_desc);

  npu_runtime_release_owned_kernelset(runtime);

  ssize_t ret = npu_runtime_load_kernelset_desc(runtime, &copied_desc);
  if (ret != NPU_SUCCESS) {
    for (size_t i = 0; i < kernel_set_desc->kernel_count; i++) {
      free(names_copy[i]);
    }
    free(names_copy);
    free(kernels_copy);
    free(data_copy);
    dlclose(new_handle);
    return ret;
  }

  runtime->owned_kernelset_data = data_copy;
  runtime->owned_kernel_descs = kernels_copy;
  runtime->owned_kernel_names = names_copy;
  runtime->owned_kernel_name_count = kernel_set_desc->kernel_count;

  if (runtime->kernelset_so_handle) {
    dlclose(runtime->kernelset_so_handle);
  }
  runtime->kernelset_so_handle = new_handle;
  return NPU_SUCCESS;
}

// Blocking until kernel exit
ssize_t npu_runtime_kernel_start(const struct NPURuntimeState *runtime,
                         const struct NPUKernelDesc *kernel_desc,
                         int* kernel_ret, size_t arg_count, ...) {
  if (runtime == NULL || arg_count > NPU_MAX_KERNEL_ARGS) {
    return NPU_ERROR_INVALID_ARGUMENT;
  }

  if (kernel_desc == NULL || kernel_desc->entry == 0) {
    return NPU_ERROR_KERNEL_NOT_FOUND;
  }

  ssize_t ret;

  ret = npu_check(&runtime->dev_state);
  if(ret != -NPU_DEVICE_ERRNO_EMPTY) {
    return ret;
  }

  struct GeneralEntryArgs general_entry_args;

  general_entry_args.kernel_entry = kernel_desc->entry;
  general_entry_args.arg_count = arg_count;
  general_entry_args.ret = 0xdeadbeaf;
  // general_entry_args.putchar_buf = 0x80040000; // 0x80040000 - 0x81000000 is putchar buf
  // general_entry_args.putchar_buflength = 0x81000000 - 0x80040000;
  general_entry_args.putchar_buf = runtime->printchar_base;
  general_entry_args.putchar_buflength = runtime->printchar_size;
  general_entry_args.putchar_wptr = 0;

  va_list args;
  va_start(args, arg_count);
  for(int i = 0 ; i < arg_count; i++) {
    general_entry_args.args[i] = va_arg(args, uint64_t);
  }
  va_end(args);

  ret = npu_memcpy_to_dev(&runtime->dev_state, runtime->kernel_set_desc.args_ptr,
                          &general_entry_args, sizeof(general_entry_args) + (arg_count - 30) * sizeof(uint64_t));

  if (ret < 0) {
    printf("Failed to copy arguments to device: %zd\n", ret);
    return NPU_ERROR_DEVICE_FAILURE;
  }

  ret = npu_start(&runtime->dev_state, runtime->kernel_set_desc.general_entry);
  if (ret < 0) {
    printf("Failed to start NPU device: %zd\n", ret);
    return NPU_ERROR_DEVICE_FAILURE;
  }

  long t0 = 0;
  while(1) {
    ret = npu_check(&runtime->dev_state);
    if (ret == -NPU_DEVICE_ERRNO_IO) {
      ret = NPU_ERROR_DEVICE_FAILURE;
      break;
    }
    if(ret == -NPU_DEVICE_ERRNO_EMPTY) {
      ret = npu_memcpy_from_dev(&runtime->dev_state, &general_entry_args,
        runtime->kernel_set_desc.args_ptr, sizeof(general_entry_args) - 30 * sizeof(uint64_t));
      if(kernel_ret) {
        *kernel_ret = general_entry_args.ret;
      }
      static char printbuf[NPU_PRINTCHAR_BUF_SIZE];
      npu_memcpy_from_dev(&runtime->dev_state, printbuf, general_entry_args.putchar_buf, general_entry_args.putchar_wptr);
      for(long i = 0 ; i < general_entry_args.putchar_wptr ; i ++) {
        putchar(printbuf[i]);
      }
      // printf("Exit using %ldus printbuf %d\n",t0 * 100, general_entry_args.putchar_wptr);
      return NPU_SUCCESS;
    }
    t0++;
    usleep(100); // 0.1ms
  }
  return ret;
}

ssize_t npu_runtime_get_kernel_desc(const struct NPURuntimeState *runtime,
                            const char *kernel_name,
                            struct NPUKernelDesc **kernel_desc_out) {
  if (runtime == NULL || kernel_name == NULL || kernel_desc_out == NULL) {
    return NPU_ERROR_INVALID_ARGUMENT;
  }

  for (size_t i = 0; i < runtime->kernel_set_desc.kernel_count; i++) {
    if (strcmp(runtime->kernel_set_desc.kernels[i].name, kernel_name) == 0) {
      *kernel_desc_out = &runtime->kernel_set_desc.kernels[i];
      return NPU_SUCCESS;
    }
  }

  // Redirect-and-load path for per-kernel .so layout.
  struct NPURuntimeState *runtime_mut = (struct NPURuntimeState *)runtime;
  ssize_t load_ret = npu_runtime_try_load_kernel_by_name(runtime_mut, kernel_name);
  if (load_ret != NPU_SUCCESS) {
    return NPU_ERROR_KERNEL_NOT_FOUND;
  }

  for (size_t i = 0; i < runtime->kernel_set_desc.kernel_count; i++) {
    if (strcmp(runtime->kernel_set_desc.kernels[i].name, kernel_name) == 0) {
      *kernel_desc_out = &runtime->kernel_set_desc.kernels[i];
      return NPU_SUCCESS;
    }
  }

  return NPU_ERROR_KERNEL_NOT_FOUND;
}

ssize_t npu_runtime_memset(const struct NPURuntimeState *runtime, uint64_t  dev_ptr, int c, size_t n) {
  struct NPUKernelDesc *memset_desc = NULL;
  ssize_t ret;
  ret = npu_runtime_get_kernel_desc(runtime, "kernel_memset", &memset_desc);
  if(ret < 0) return ret;
  npu_runtime_kernel_start(runtime, memset_desc, (int*)&ret, 3, dev_ptr, (uint64_t) c, n);
  return ret;
}

#ifdef __cplusplus
}
#endif
