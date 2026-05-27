#include "avl_tree.hpp"
#include "NPURuntime.h"
#include <iostream>
#include <dirent.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

class NPUMemoryBlock {
public:
    uint64_t start;
    uint64_t size;
    NPUMemoryBlock() : start(0), size(0) {};
    NPUMemoryBlock(uint64_t start, uint64_t size) : start(start), size(size) {};
    bool operator <(const NPUMemoryBlock &other) const {
        return this->start < other.start;
    }
    bool operator ==(const NPUMemoryBlock &other) const {
        return this->start == other.start;
    }
};

class NPUMemoryManagement {
private:
    AVL::tree<NPUMemoryBlock> fm;
    AVL::tree<NPUMemoryBlock> um;
    size_t total_mem;
    size_t free_mem;
public:
    NPUMemoryManagement(uint64_t heap_start, uint64_t heap_end) {
        fm.insert(NPUMemoryBlock(heap_start, heap_end - heap_start));
        total_mem = heap_end - heap_start;
        free_mem  = total_mem;
    }
    void dump(std::ostream& os) {
        for(auto it : fm) {
            os << "FM Entry: 0x" << std::hex << it.start << " 0x" << it.size << std::endl;
        }
        for(auto it : um) {
            os << "UM Entry: 0x" << std::hex << it.start << " 0x" << it.size << std::endl;
        }
    }
    bool alloc(uint64_t *addr, size_t size) {
        for(auto iter : fm) {
            if(iter.size >= size) {
                fm.remove(iter);
                *addr = iter.start;
                if(iter.size > size) {
                    fm.insert(NPUMemoryBlock(iter.start + size, iter.size - size));
                }
                um.insert(NPUMemoryBlock(iter.start, size));
                free_mem -= size;
                return true;
            }
        }
        return false;
    }
    bool dealloc(uint64_t addr) {
        auto it = um.find(NPUMemoryBlock(addr, 0));
        if(it == um.end()) {
            return false;
        }
        NPUMemoryBlock b = *it;
        free_mem += b.size;
        um.remove(b);
        fm.insert(b);
        it = fm.find(b);
        auto left_it = it; left_it--;
        auto right_it = it; right_it++;
        if(&(*left_it) != NULL && (left_it->start + left_it->size) == it->start) {
            b = *left_it;
            fm.remove(b);
            it->start = b.start;
            it->size += b.size;
        }
        if(&(*right_it) != NULL && (it->start + it->size) == right_it->start) {
            it->size += right_it->size;
            fm.remove(*right_it);
        }
        return true;
    }
    ssize_t get_freemem() {
        return (ssize_t) free_mem;
    }
    ssize_t get_totalmem() {
        return (ssize_t) total_mem;
    }
};
extern "C" {
    static bool npu_runtime_mm_isa_matches(const char *entry_isa, const char *device_isa) {
        if (device_isa == NULL || device_isa[0] == '\0') {
            device_isa = "v0.1";
        }
        if (entry_isa == NULL || entry_isa[0] == '\0') {
            entry_isa = "v0.1";
        }
        return strcmp(entry_isa, device_isa) == 0;
    }

    static ssize_t npu_runtime_lazy_bootstrap_mm(struct NPURuntimeState *runtime) {
        if (runtime == NULL) {
            return NPU_ERROR_INVALID_ARGUMENT;
        }
        if (runtime->mm != NULL) {
            return NPU_SUCCESS;
        }

        // Prefer embedded fatbin-native images first so mm bootstrap no longer
        // depends on per-kernel .so artifacts.
        if (runtime->fatbin_entries != NULL && runtime->fatbin_entry_count > 0) {
            for (size_t i = 0; i < runtime->fatbin_entry_count; i++) {
                const struct NPUFatBinaryEntry *entry = &runtime->fatbin_entries[i];
                if (entry == NULL || entry->image == NULL) {
                    continue;
                }
                if (!npu_runtime_mm_isa_matches(entry->isa_version, runtime->device_isa_version)) {
                    continue;
                }
                ssize_t ret = npu_runtime_load_kernelset_image(runtime, entry->image);
                if (ret == NPU_SUCCESS && runtime->mm != NULL) {
                    return NPU_SUCCESS;
                }
            }
            for (size_t i = 0; i < runtime->fatbin_entry_count; i++) {
                const struct NPUFatBinaryEntry *entry = &runtime->fatbin_entries[i];
                if (entry == NULL || entry->image == NULL) {
                    continue;
                }
                ssize_t ret = npu_runtime_load_kernelset_image(runtime, entry->image);
                if (ret == NPU_SUCCESS && runtime->mm != NULL) {
                    return NPU_SUCCESS;
                }
            }
        }

        const char *bootstrap_so = getenv("NPU_KERNEL_BOOTSTRAP_SO");
        if (bootstrap_so != NULL && bootstrap_so[0] != '\0') {
            ssize_t ret = npu_runtime_load_kernelset(runtime, bootstrap_so);
            if (ret == NPU_SUCCESS && runtime->mm != NULL) {
                return NPU_SUCCESS;
            }
        }

        const char *kernel_so_dir = getenv("NPU_KERNEL_SO_DIR");
        if (kernel_so_dir == NULL || kernel_so_dir[0] == '\0') {
            kernel_so_dir = "/workspace/npu-kernels/bin";
        }

        DIR *dir = opendir(kernel_so_dir);
        if (dir == NULL) {
            return NPU_ERROR_NOT_INITIALIZED;
        }

        struct dirent *ent = NULL;
        while ((ent = readdir(dir)) != NULL) {
            const char *name = ent->d_name;
            size_t len = strlen(name);
            if (len < 4 || strcmp(name + len - 3, ".so") != 0) {
                continue;
            }

            char so_path[PATH_MAX];
            int written = snprintf(so_path, sizeof(so_path), "%s/%s", kernel_so_dir, name);
            if (written <= 0 || (size_t)written >= sizeof(so_path)) {
                continue;
            }

            ssize_t ret = npu_runtime_load_kernelset(runtime, so_path);
            if (ret == NPU_SUCCESS && runtime->mm != NULL) {
                closedir(dir);
                return NPU_SUCCESS;
            }
        }

        closedir(dir);
        return NPU_ERROR_NOT_INITIALIZED;
    }

    ssize_t npu_mm_init(void** mm, uint64_t heap_start, uint64_t heap_end) {
        *mm = new NPUMemoryManagement(heap_start, heap_end);
        return NPU_SUCCESS;
    }

    ssize_t npu_mm_deinit(void** mm) {
        delete (NPUMemoryManagement*)*mm;
        *mm = NULL;
        return NPU_SUCCESS;
    }

    ssize_t npu_runtime_malloc(const struct NPURuntimeState *runtime, uint64_t *dev_ptr, size_t size) {
        if (runtime == NULL || dev_ptr == NULL) {
            return NPU_ERROR_INVALID_ARGUMENT;
        }
        if (runtime->mm == NULL) {
            ssize_t ret = npu_runtime_lazy_bootstrap_mm((struct NPURuntimeState *)runtime);
            if (ret != NPU_SUCCESS) {
                return ret;
            }
        }
        auto mm = (NPUMemoryManagement*)runtime->mm;
        bool success = mm->alloc(dev_ptr, size);
        return success ? NPU_SUCCESS : -NPU_DEVICE_ERRNO_INVALID_SIZE;
    }

    ssize_t npu_runtime_free(const struct NPURuntimeState *runtime, uint64_t  dev_ptr) {
        if (runtime == NULL) {
            return NPU_ERROR_INVALID_ARGUMENT;
        }
        if (runtime->mm == NULL) {
            ssize_t ret = npu_runtime_lazy_bootstrap_mm((struct NPURuntimeState *)runtime);
            if (ret != NPU_SUCCESS) {
                return ret;
            }
        }
        auto mm = (NPUMemoryManagement*)runtime->mm;
        bool success = mm->dealloc(dev_ptr);
        return success ? NPU_SUCCESS : -NPU_DEVICE_ERRNO_INVALID_ARGS;
    }

    ssize_t npu_runtime_get_meminfo(const struct NPURuntimeState *runtime,
        size_t * free, size_t * total) {
        if (runtime == NULL || free == NULL || total == NULL) {
            return NPU_ERROR_INVALID_ARGUMENT;
        }
        if (runtime->mm == NULL) {
            ssize_t ret = npu_runtime_lazy_bootstrap_mm((struct NPURuntimeState *)runtime);
            if (ret != NPU_SUCCESS) {
                return ret;
            }
        }
        auto mm = (NPUMemoryManagement*)runtime->mm;
        if(!mm) {
            return -NPU_DEVICE_ERRNO_INVALID_ARGS;
        }
        *free = mm->get_freemem();
        *total = mm->get_totalmem();
        return NPU_SUCCESS;
    }

    // Hidden debug function
    void npu_mm_dump(const struct NPURuntimeState *runtime) {
        auto mm = (NPUMemoryManagement*)runtime->mm;
        mm->dump(std::cout);
    }
}
