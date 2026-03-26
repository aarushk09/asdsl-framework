/**
 * P-core OpenMP thread count cap + per-thread affinity (Windows).
 * Matches Phase 1 STREAM / Q4K GEMV behavior in forward_loop.cpp.
 */
#pragma once

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

namespace asdsl_omp_pinning {

inline bool& pin_openmp_pcores_enabled() {
    static bool v = true;
    return v;
}

inline DWORD_PTR lowest_set_bit_mask(DWORD_PTR mask) {
    if (mask == 0) {
        return 0;
    }
    return mask & (~mask + 1);
}

inline std::vector<DWORD_PTR> detect_windows_pcore_masks() {
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len == 0) {
        return {};
    }

    std::vector<uint8_t> buf(len);
    auto* base = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data());
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, base, &len)) {
        return {};
    }

    uint8_t min_eff = std::numeric_limits<uint8_t>::max();
    std::vector<std::pair<uint8_t, DWORD_PTR>> core_masks;

    uint8_t* ptr = buf.data();
    uint8_t* end = buf.data() + len;
    while (ptr < end) {
        auto* info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(ptr);
        if (info->Relationship == RelationProcessorCore && info->Processor.GroupCount > 0) {
            const GROUP_AFFINITY& ga = info->Processor.GroupMask[0];
            if (ga.Group == 0 && ga.Mask != 0) {
                uint8_t eff = info->Processor.EfficiencyClass;
                min_eff = std::min(min_eff, eff);
                core_masks.push_back({eff, lowest_set_bit_mask(ga.Mask)});
            }
        }
        ptr += info->Size;
    }

    std::vector<DWORD_PTR> pcores;
    for (const auto& e : core_masks) {
        if (e.first == min_eff && e.second != 0) {
            pcores.push_back(e.second);
        }
    }
    if (!pcores.empty()) {
        return pcores;
    }

    for (const auto& e : core_masks) {
        if (e.second != 0) {
            pcores.push_back(e.second);
        }
    }
    return pcores;
}

inline const std::vector<DWORD_PTR>& get_pcore_masks() {
    static std::vector<DWORD_PTR> masks = detect_windows_pcore_masks();
    return masks;
}

inline void bind_omp_thread_to_pcore_if_enabled() {
    if (!pin_openmp_pcores_enabled()) {
        return;
    }
    const auto& masks = get_pcore_masks();
    if (masks.empty()) {
        return;
    }
    const int tid = omp_get_thread_num();
    const DWORD_PTR mask = masks[static_cast<size_t>(tid) % masks.size()];
    if (mask != 0) {
        SetThreadAffinityMask(GetCurrentThread(), mask);
    }
}

inline void configure_openmp_for_pcores() {
    if (!pin_openmp_pcores_enabled()) {
        return;
    }
    const auto& masks = get_pcore_masks();
    if (masks.empty()) {
        return;
    }
    omp_set_dynamic(0);
    const int target_threads = static_cast<int>(masks.size());
    if (target_threads > 0) {
        omp_set_num_threads(target_threads);
    }
}

}  // namespace asdsl_omp_pinning

#else

namespace asdsl_omp_pinning {

inline bool& pin_openmp_pcores_enabled() {
    static bool v = true;
    return v;
}

inline void bind_omp_thread_to_pcore_if_enabled() {}

inline void configure_openmp_for_pcores() {}

}  // namespace asdsl_omp_pinning

#endif
