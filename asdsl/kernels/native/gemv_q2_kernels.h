#pragma once

#include <cstdint>

void gemv_q2_unpacked_impl(
    const uint8_t* w,
    const float* x,
    const float* scales,
    const float* biases,
    float* y,
    int M, int K, int group_size);

void gemv_q2_packed_impl(
    const uint8_t* w_packed,
    const float* x,
    const float* scales,
    const float* biases,
    float* y,
    int M, int K, int group_size);
