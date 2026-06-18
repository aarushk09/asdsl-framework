/** Windows 2 MB large-page allocation helper (Phase 6, optional). */
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace asdsl_large_pages {

inline bool large_pages_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char* v = std::getenv("ASDSL_LARGE_PAGES");
        enabled = (v && v[0] != '0') ? 1 : 0;
    }
    return enabled != 0;
}

inline void* alloc_aligned(size_t nbytes, size_t alignment, bool* used_large_pages) {
#ifdef _WIN32
    if (large_pages_enabled()) {
        SIZE_T page = GetLargePageMinimum();
        if (page > 0) {
            const size_t rounded = ((nbytes + page - 1) / page) * page;
            void* p = VirtualAlloc(
                nullptr, rounded, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE);
            if (p) {
                if (used_large_pages) {
                    *used_large_pages = true;
                }
                return p;
            }
        }
    }
    if (used_large_pages) {
        *used_large_pages = false;
    }
    return _aligned_malloc(nbytes, alignment);
#else
    (void)alignment;
    if (used_large_pages) {
        *used_large_pages = false;
    }
    void* p = nullptr;
    if (posix_memalign(&p, 64, nbytes) != 0) {
        return nullptr;
    }
    return p;
#endif
}

inline void free_aligned(void* p, bool used_large_pages) {
    if (!p) {
        return;
    }
#ifdef _WIN32
    if (used_large_pages) {
        VirtualFree(p, 0, MEM_RELEASE);
    } else {
        _aligned_free(p);
    }
#else
    (void)used_large_pages;
    free(p);
#endif
}

/** Contiguous byte buffer; uses 2 MB pages when ASDSL_LARGE_PAGES=1. */
class ByteBuffer {
public:
    ByteBuffer() = default;
    ~ByteBuffer() { clear(); }

    ByteBuffer(const ByteBuffer&) = delete;
    ByteBuffer& operator=(const ByteBuffer&) = delete;

    ByteBuffer(ByteBuffer&& other) noexcept { swap(other); }
    ByteBuffer& operator=(ByteBuffer&& other) noexcept {
        if (this != &other) {
            clear();
            swap(other);
        }
        return *this;
    }

    void clear() {
        if (ptr_) {
            free_aligned(ptr_, large_);
        }
        ptr_ = nullptr;
        size_ = 0;
        cap_ = 0;
        large_ = false;
    }

    void resize(size_t nbytes) {
        if (nbytes <= cap_) {
            size_ = nbytes;
            return;
        }
        clear();
        bool lp = false;
        ptr_ = static_cast<uint8_t*>(alloc_aligned(nbytes, 64, &lp));
        if (!ptr_) {
            throw std::bad_alloc();
        }
        large_ = lp;
        cap_ = nbytes;
        size_ = nbytes;
    }

    void assign(const uint8_t* src, size_t nbytes) {
        resize(nbytes);
        if (nbytes > 0) {
            std::memcpy(ptr_, src, nbytes);
        }
    }

    void assign(const uint8_t* begin, const uint8_t* end) {
        assign(begin, static_cast<size_t>(end - begin));
    }

    uint8_t* data() { return ptr_; }
    const uint8_t* data() const { return ptr_; }
    size_t size() const { return size_; }
    bool large_page_backed() const { return large_; }

private:
    void swap(ByteBuffer& other) noexcept {
        std::swap(ptr_, other.ptr_);
        std::swap(size_, other.size_);
        std::swap(cap_, other.cap_);
        std::swap(large_, other.large_);
    }

    uint8_t* ptr_ = nullptr;
    size_t size_ = 0;
    size_t cap_ = 0;
    bool large_ = false;
};

}  // namespace asdsl_large_pages
