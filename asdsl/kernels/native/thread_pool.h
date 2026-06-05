#ifndef NOMINMAX
#define NOMINMAX
#endif
#pragma once
#include <vector>
#include <thread>
#include <atomic>
#include <functional>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <immintrin.h>
#include <cstdint>
#include <cassert>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <pthread.h>
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#endif

namespace asdsl {

class ThreadPool;
extern thread_local ThreadPool* tl_active_pool;

// Get physical core count
inline int get_physical_cores() {
#if defined(_WIN32)
    DWORD buffer_size = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &buffer_size);
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
        return std::thread::hardware_concurrency() / 2;
    }
    std::vector<char> buffer(buffer_size);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buffer.data(), &buffer_size)) {
        return std::thread::hardware_concurrency() / 2;
    }
    
    // First pass to find the maximum EfficiencyClass (which represents P-Cores)
    BYTE max_efficiency_class = 0;
    DWORD offset = 0;
    while (offset < buffer_size) {
        auto info = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(buffer.data() + offset);
        if (info->Relationship == RelationProcessorCore) {
            if (info->Processor.EfficiencyClass > max_efficiency_class) {
                max_efficiency_class = info->Processor.EfficiencyClass;
            }
        }
        offset += info->Size;
    }
    
    int pcore_count = 0;
    offset = 0;
    while (offset < buffer_size) {
        auto info = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(buffer.data() + offset);
        if (info->Relationship == RelationProcessorCore) {
            // Only count if it matches the highest efficiency class (P-cores)
            if (max_efficiency_class == 0 || info->Processor.EfficiencyClass == max_efficiency_class) {
                pcore_count++;
            }
        }
        offset += info->Size;
    }
    return pcore_count > 0 ? pcore_count : std::thread::hardware_concurrency() / 2;
#else
    // Simplified for Linux
    return std::thread::hardware_concurrency() / 2;
#endif
}

inline void pin_thread_to_core(std::thread& t, int core_id) {
#if defined(_WIN32)
    HANDLE handle = t.native_handle();
    DWORD_PTR mask = (1ULL << core_id);
    SetThreadAffinityMask(handle, mask);
#else
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(t.native_handle(), sizeof(cpu_set_t), &cpuset);
#endif
}

class ThreadPool {
    struct Job {
        std::function<void(int)> func;
        int begin;
        int end;
        int grain_size;
        std::atomic<int> current_idx;
        std::atomic<int> tasks_completed;
        int total_tasks;
        std::atomic<int> workers_inside; // <-- used to wait for workers to leave
        uint64_t job_id; // Unique identifier for each job iteration
    };

    std::atomic<bool> stop_flag{false};

    Job persistent_job;
    std::atomic<Job*> current_job{nullptr};

    std::vector<std::thread> workers;

    void worker_loop(int thread_id) {

        Job* local_job = nullptr;
        uint64_t local_job_id = 0;
        // Graduated backoff counter for idle periods (no active job).
        // Phases: <32768 → _mm_pause(), <36864 → yield(), else → sleep(1µs).
        //
        // Threshold tuning: at ~10 ns per _mm_pause(), 32768 iterations ≈ 327 µs.
        // The longest inter-barrier gap during forward_token() is swiglu_quantize
        // on 17920 elements, measured at ~15 µs on this hardware. 327 µs >> 15 µs,
        // so workers will always still be in _mm_pause() when the next parallel_for
        // fires during active generation. yield() is only reached between generate()
        // calls (user think-time), preventing thermal throttling on long sessions.
        int idle_spin = 0;

        while (!stop_flag.load(std::memory_order_relaxed)) {
            Job* expected = current_job.load(std::memory_order_acquire);
            if (expected == nullptr) {
                local_job = nullptr;
                // Graduated backoff: spin → yield → sleep
                if (++idle_spin < 32768) {
                    _mm_pause();
                } else if (idle_spin < 36864) {
                    std::this_thread::yield();
                } else {
                    std::this_thread::sleep_for(std::chrono::microseconds(1));
                    idle_spin = 36864; // cap to avoid integer overflow
                }
            } else if (expected != local_job || local_job_id != expected->job_id) {
                idle_spin = 0; // reset backoff on new job
                local_job = expected;
                local_job_id = expected->job_id;
                local_job->workers_inside.fetch_add(1, std::memory_order_relaxed);

                while (true) {
                    int start = local_job->current_idx.fetch_add(local_job->grain_size, std::memory_order_relaxed);
                    if (start >= local_job->end) {
                        break;
                    }
                    int end = std::min(start + local_job->grain_size, local_job->end);
                    for (int i = start; i < end; ++i) {
                        local_job->func(i);
                    }
                    local_job->tasks_completed.fetch_add(1, std::memory_order_release);
                }
                local_job->workers_inside.fetch_sub(1, std::memory_order_release);
            } else {
                // Brief spin: waiting for master to clear current_job after all tasks
                // are completed. This window is typically <50 cycles — do not back off.
                _mm_pause();
            }
        }
    }

public:

    // Logical processor IDs for P-cores ONLY (highest EfficiencyClass), no HT siblings.
    //
    // Rationale: on Intel Raptor Lake (8P + 4E), including E-cores hurts bandwidth-bound
    // workloads. All 12 cores share the same memory controller; E-core DRAM requests
    // compete for bandwidth without proportional throughput contribution due to lower
    // per-core sustained frequency and memory bus priority. llama.cpp achieves 3.44 tok/s
    // with -t 8 (P-cores only); ASDSL's 12-thread config degraded effective BW from
    // 20.4 GB/s to 17.5 GB/s (measured). Filtering to P-cores targets parity.
    static std::vector<int> get_pcore_logical_ids() {
        std::vector<int> ids;
#if defined(_WIN32)
        DWORD buffer_size = 0;
        GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &buffer_size);
        if (GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
            // API unavailable: conservative fallback — use half of logical cores (P-cores on HT)
            int n = (int)std::thread::hardware_concurrency();
            int half = std::max(1, n / 2);
            for (int i = 0; i < half; ++i) ids.push_back(i);
            return ids;
        }
        std::vector<char> buf(buffer_size);
        GetLogicalProcessorInformationEx(RelationProcessorCore,
            (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buf.data(), &buffer_size);

        struct CoreEntry {
            int id;
            BYTE eff;
        };

        // Pass 1: find the maximum EfficiencyClass (= P-core class)
        BYTE max_eff = 0;
        DWORD off = 0;
        while (off < buffer_size) {
            auto* info = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(buf.data() + off);
            if (info->Relationship == RelationProcessorCore) {
                if (info->Processor.EfficiencyClass > max_eff)
                    max_eff = info->Processor.EfficiencyClass;
            }
            off += info->Size;
        }

        // Pass 2: collect only P-core logical IDs (EfficiencyClass == max_eff)
        std::vector<CoreEntry> entries;
        off = 0;
        while (off < buffer_size) {
            auto* info = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(buf.data() + off);
            if (info->Relationship == RelationProcessorCore &&
                info->Processor.EfficiencyClass == max_eff) {
                for (WORD g = 0; g < info->Processor.GroupCount; ++g) {
                    KAFFINITY mask = info->Processor.GroupMask[g].Mask;
                    USHORT group = info->Processor.GroupMask[g].Group;
                    // First set bit only: one logical CPU per physical core (skip HT sibling)
                    for (int bit = 0; bit < 64; ++bit) {
                        if (mask & (KAFFINITY(1) << bit)) {
                            entries.push_back({(int)(group * 64 + bit), max_eff});
                            break;
                        }
                    }
                }
            }
            off += info->Size;
        }
        std::sort(entries.begin(), entries.end(), [](const CoreEntry& a, const CoreEntry& b) {
            return a.id < b.id;
        });
        for (const auto& e : entries) ids.push_back(e.id);
#else
        // Linux: assume first half of logical cores are P-cores (no SMT on typical setups)
        int n = (int)std::thread::hardware_concurrency();
        int half = std::max(1, n / 2);
        for (int i = 0; i < half; ++i) ids.push_back(i);
#endif
        if (ids.empty()) {
            // Final fallback: at least 1 thread
            int n = (int)std::thread::hardware_concurrency();
            int half = std::max(1, n / 2);
            for (int i = 0; i < half; ++i) ids.push_back(i);
        }
        return ids;
    }

    // Returns ALL logical CPU IDs for P-cores, including HyperThreading siblings.
    // On a 4P+HT system returns [0,1,2,3,4,5,6,7]; on 8P without HT returns [0..7].
    // Use this for OMP_PLACES to fill all P-core execution slots.
    static std::vector<int> get_all_pcore_logical_ids() {
        std::vector<int> ids;
#if defined(_WIN32)
        DWORD buffer_size = 0;
        GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &buffer_size);
        if (GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
            int n = (int)std::thread::hardware_concurrency();
            int half = std::max(1, n / 2);
            for (int i = 0; i < half * 2; ++i) ids.push_back(i);
            return ids;
        }
        std::vector<char> buf(buffer_size);
        GetLogicalProcessorInformationEx(RelationProcessorCore,
            (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buf.data(), &buffer_size);

        // Pass 1: find max EfficiencyClass (= P-core class)
        BYTE max_eff = 0;
        DWORD off = 0;
        while (off < buffer_size) {
            auto* info = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(buf.data() + off);
            if (info->Relationship == RelationProcessorCore) {
                if (info->Processor.EfficiencyClass > max_eff)
                    max_eff = info->Processor.EfficiencyClass;
            }
            off += info->Size;
        }

        // Pass 2: collect ALL logical IDs for P-cores (every set bit, including HT siblings)
        off = 0;
        while (off < buffer_size) {
            auto* info = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(buf.data() + off);
            if (info->Relationship == RelationProcessorCore &&
                info->Processor.EfficiencyClass == max_eff) {
                for (WORD g = 0; g < info->Processor.GroupCount; ++g) {
                    KAFFINITY mask = info->Processor.GroupMask[g].Mask;
                    USHORT group = info->Processor.GroupMask[g].Group;
                    for (int bit = 0; bit < 64; ++bit) {
                        if (mask & (KAFFINITY(1) << bit)) {
                            ids.push_back((int)(group * 64 + bit));
                        }
                    }
                }
            }
            off += info->Size;
        }
        std::sort(ids.begin(), ids.end());
#else
        int n = (int)std::thread::hardware_concurrency();
        int half = std::max(1, n / 2);
        for (int i = 0; i < half; ++i) ids.push_back(i);
#endif
        if (ids.empty()) {
            int n = (int)std::thread::hardware_concurrency();
            for (int i = 0; i < n; ++i) ids.push_back(i);
        }
        return ids;
    }

    // Default constructor: auto-detect P-cores and spawn one worker per core.
    ThreadPool() {
        persistent_job.job_id = 0;
        auto pcore_ids = get_pcore_logical_ids();
        int n_threads = (int)pcore_ids.size();
        for (int i = 0; i < n_threads; ++i) {
            workers.emplace_back(&ThreadPool::worker_loop, this, i);
            pin_thread_to_core(workers.back(), pcore_ids[i]);
        }
    }

    // Zero-worker constructor: no background threads are created.
    // All parallel_for calls fall through to serial execution on the master thread
    // (or are never called when replaced with #pragma omp parallel for).
    // Use this when parallelism is handled externally (e.g. OpenMP).
    explicit ThreadPool(int n_workers) {
        persistent_job.job_id = 0;
        if (n_workers <= 0) return;  // zero-worker mode: no threads created
        auto pcore_ids = get_pcore_logical_ids();
        n_workers = std::min(n_workers, (int)pcore_ids.size());
        for (int i = 0; i < n_workers; ++i) {
            workers.emplace_back(&ThreadPool::worker_loop, this, i);
            pin_thread_to_core(workers.back(), pcore_ids[i]);
        }
    }

    ~ThreadPool() {
        stop_flag.store(true, std::memory_order_relaxed);
        for (auto& t : workers) {
            if (t.joinable()) {
                t.join();
            }
        }
    }

    int thread_count() const { return (int)workers.size(); }

    void parallel_for(int begin, int end, int grain_size, const std::function<void(int)>& func) {
        if (begin >= end) return;
        
        int total_elements = end - begin;
        if (total_elements <= grain_size || workers.empty()) {
            for (int i = begin; i < end; ++i) {
                func(i);
            }
            return;
        }

        int chunks = (total_elements + grain_size - 1) / grain_size;

        persistent_job.func = func;
        persistent_job.begin = begin;
        persistent_job.end = end;
        persistent_job.grain_size = grain_size;
        persistent_job.current_idx.store(begin, std::memory_order_relaxed);
        persistent_job.tasks_completed.store(0, std::memory_order_relaxed);
        persistent_job.workers_inside.store(1, std::memory_order_relaxed); // master is inside
        persistent_job.total_tasks = chunks;
        persistent_job.job_id++; // increment job definition identifier

        current_job.store(&persistent_job, std::memory_order_release);

        Job& job = persistent_job;
        // Master thread helps
        while (true) {
            int start = job.current_idx.fetch_add(job.grain_size, std::memory_order_relaxed);
            if (start >= job.end) {
                break;
            }
            int chunk_end = std::min(start + job.grain_size, job.end);
            for (int i = start; i < chunk_end; ++i) {
                job.func(i);
            }
            job.tasks_completed.fetch_add(1, std::memory_order_release);
        }

        // Wait for all chunks to finish
        while (job.tasks_completed.load(std::memory_order_acquire) < job.total_tasks) {
            _mm_pause();
        }

        // Drain workers BEFORE clearing current_job and before touching persistent_job
        // again. The original order (clear current_job first, then drain workers_inside)
        // was a data race: after current_job becomes nullptr workers exit their chunk loop
        // and decrement workers_inside, but between the nullptr store and the
        // workers_inside drain the *next* parallel_for call (from any engine instance)
        // could overwrite persistent_job.func with a new lambda.  Any worker thread
        // that had not yet executed its final local_job->func(i) would then call the
        // new lambda with a stale index, crashing into the new engine's data.
        //
        // Correct order:
        //   1. Decrement master's workers_inside share.
        //   2. Wait until every worker has also decremented (fully exited chunk loop).
        //   3. Only then clear current_job so the next submission can safely mutate
        //      persistent_job.
        job.workers_inside.fetch_sub(1, std::memory_order_release);
        while (job.workers_inside.load(std::memory_order_acquire) > 0) {
            _mm_pause();
        }

        // All workers have exited the chunk loop and no longer hold a reference to
        // persistent_job.func.  It is now safe to let a new submission overwrite it.
        current_job.store(nullptr, std::memory_order_release);
    }
    
    // Helper for 2D tiling (e.g. nested loops)
    void parallel_for_2d(int rows, int cols, int grain_row, int grain_col, const std::function<void(int, int)>& func) {
        // Flatten the 2D loop over chunks
        int row_chunks = (rows + grain_row - 1) / grain_row;
        int col_chunks = (cols + grain_col - 1) / grain_col;
        int total_chunks = row_chunks * col_chunks;
        
        parallel_for(0, total_chunks, 1, [&](int chunk_idx) {
            int r_c = chunk_idx / col_chunks;
            int c_c = chunk_idx % col_chunks;
            
            int r_start = r_c * grain_row;
            int r_end = std::min(r_start + grain_row, rows);
            
            int c_start = c_c * grain_col;
            int c_end = std::min(c_start + grain_col, cols);
            
            for (int r = r_start; r < r_end; ++r) {
                for (int c = c_start; c < c_end; ++c) {
                    func(r, c);
                }
            }
        });
    }

    // Per-call-stack active pool: set by UnifiedEngine before every kernel
    // dispatch so that free functions (gemv_q4_avx2.cpp etc.) that call
    // get_instance() transparently use the engine's own isolated pool.
    // Falls back to a process-wide default if no engine has set the pointer
    // (e.g. kernels called directly from Python without an engine instance).
    static ThreadPool& get_instance() {
        // tl_active_pool is set/cleared by ScopedActivePool (below).
        // Defined in thread_pool.h as inline thread_local to avoid ODR issues.
        if (tl_active_pool) return *tl_active_pool;
        // Fallback singleton for code paths not driven through UnifiedEngine.
        static ThreadPool fallback_pool;
        return fallback_pool;
    }
};

// Thread-local pointer written by UnifiedEngine before every parallel_for
// dispatch and cleared afterward.  Declared inline so it can live in a header
// included by multiple translation units without ODR violations (C++17).
inline thread_local ThreadPool* tl_active_pool = nullptr;

// RAII guard: sets tl_active_pool for the duration of a scope.
struct ScopedActivePool {
    explicit ScopedActivePool(ThreadPool* p) { tl_active_pool = p; }
    ~ScopedActivePool()                      { tl_active_pool = nullptr; }
};

} // namespace asdsl
