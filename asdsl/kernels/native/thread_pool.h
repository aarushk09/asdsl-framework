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
        while (!stop_flag.load(std::memory_order_relaxed)) {
            Job* expected = current_job.load(std::memory_order_acquire);
            if (expected == nullptr) {
                local_job = nullptr;
            } else if (expected != local_job || local_job_id != expected->job_id) {
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
                _mm_pause(); // Spin wait
            }
        }
    }

public:
    ThreadPool() {
        persistent_job.job_id = 0;
        int n_threads = get_physical_cores();
        for (int i = 0; i < n_threads; ++i) {
            workers.emplace_back(&ThreadPool::worker_loop, this, i);
            pin_thread_to_core(workers.back(), i);
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

        current_job.store(nullptr, std::memory_order_release);

        // Wait for workers to leave the job's loop
        job.workers_inside.fetch_sub(1, std::memory_order_release);
        while (job.workers_inside.load(std::memory_order_acquire) > 0) {
            _mm_pause();
        }
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

    // Global singleton access
    static ThreadPool& get_instance() {
        static ThreadPool pool;
        return pool;
    }
};

} // namespace asdsl
