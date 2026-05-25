// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "convolution_riscv.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>

#include "benchmark.h"
#include "cpu.h"
#include "layer_type.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector
#include "riscv_activation.h"
#include "riscv_usability.h"
#if NCNN_INT8
#include "convolution_1x1_int8_xsmtvdot.h"
#endif

namespace ncnn {

#if NCNN_INT8
struct riscv_int8_coverage_counters_t
{
    std::atomic<unsigned long long> conv_int8_pack1_1x1_fast_count{0};
    std::atomic<unsigned long long> conv_int8_packn_1x1_fast_count{0};
    std::atomic<unsigned long long> conv_int8_packn_3x3s1_fast_count{0};
    std::atomic<unsigned long long> conv_int8_packn_3x3s1_skip_prepare_count{0};
    std::atomic<unsigned long long> conv_int8_packn_3x3s1_skip_padding_count{0};
    std::atomic<unsigned long long> conv_int8_packn_3x3s1_skip_channel_mismatch_count{0};
    std::atomic<unsigned long long> conv_int8_pack1_3x3s1_fast_count{0};
    std::atomic<unsigned long long> conv_int8_pack1_3x3s2_fast_count{0};
    std::atomic<unsigned long long> conv_int8_packout_used_count{0};
    std::atomic<unsigned long long> conv_int8_packout_forced_pack1_count{0};
    std::atomic<unsigned long long> conv_int8_packout_blocked_activation_count{0};
    std::atomic<unsigned long long> conv_int8_packout_q_omp_used_count{0};
    std::atomic<unsigned long long> conv_int8_packout_q_omp_forced_oldmap_count{0};
    std::atomic<unsigned long long> packout_store_vsse32_count{0};
    std::atomic<unsigned long long> packout_store_vsseg_count{0};
    std::atomic<unsigned long long> s2_load_vlse8_count{0};
    std::atomic<unsigned long long> s2_load_vlseg2_count{0};
    std::atomic<unsigned long long> post_vrvv_used_count{0};
    std::atomic<unsigned long long> post_scalar_used_count{0};
    std::atomic<unsigned long long> conv_int8_fallback_count{0};

    std::atomic<unsigned long long> fallback_reason_dims_ne3{0};
    std::atomic<unsigned long long> fallback_reason_dilation_ne1{0};
    std::atomic<unsigned long long> fallback_reason_stride_not_1_or_2{0};
    std::atomic<unsigned long long> fallback_reason_kernel_not_1_or_3{0};
    std::atomic<unsigned long long> fallback_reason_group_or_channel_mismatch{0};
    std::atomic<unsigned long long> fallback_reason_int8_scale_term_0{0};
    std::atomic<unsigned long long> fallback_reason_int8_uniform_false{0};
    std::atomic<unsigned long long> fallback_reason_bottom_elempack_ne1{0};
    std::atomic<unsigned long long> fallback_reason_disable_global{0};
    std::atomic<unsigned long long> fallback_reason_disable_prep{0};
    std::atomic<unsigned long long> fallback_reason_disable_1x1{0};
    std::atomic<unsigned long long> fallback_reason_disable_3x3s1{0};
    std::atomic<unsigned long long> fallback_reason_disable_3x3s2{0};
};

static riscv_int8_coverage_counters_t g_riscv_int8_coverage_counters;

static inline int riscv_int8_coverage_enabled()
{
    static const int g_enabled = []() -> int
    {
        const char* env = getenv("NCNN_RISCV_INT8_COVERAGE");
        return (env && env[0] != '\0' && env[0] != '0') ? 1 : 0;
    }();
    return g_enabled;
}

static void riscv_int8_coverage_dump()
{
    if (!riscv_int8_coverage_enabled())
        return;

    const auto load = [](const std::atomic<unsigned long long>& v) -> unsigned long long
    {
        return v.load(std::memory_order_relaxed);
    };

    NCNN_LOGE("riscv_int8_coverage conv_int8_pack1_1x1_fast_count=%llu conv_int8_packn_1x1_fast_count=%llu conv_int8_packn_3x3s1_fast_count=%llu conv_int8_packn_3x3s1_skip_prepare_count=%llu conv_int8_packn_3x3s1_skip_padding_count=%llu conv_int8_packn_3x3s1_skip_channel_mismatch_count=%llu conv_int8_pack1_3x3s1_fast_count=%llu conv_int8_pack1_3x3s2_fast_count=%llu conv_int8_packout_used_count=%llu conv_int8_packout_forced_pack1_count=%llu conv_int8_packout_blocked_activation_count=%llu conv_int8_packout_q_omp_used_count=%llu conv_int8_packout_q_omp_forced_oldmap_count=%llu packout_store_vsse32_count=%llu packout_store_vsseg_count=%llu s2_load_vlse8_count=%llu s2_load_vlseg2_count=%llu post_vrvv_used_count=%llu post_scalar_used_count=%llu conv_int8_fallback_count=%llu "
              "fallback_reason_dims_ne3=%llu fallback_reason_dilation_ne1=%llu fallback_reason_stride_not_1_or_2=%llu fallback_reason_kernel_not_1_or_3=%llu "
              "fallback_reason_group_or_channel_mismatch=%llu fallback_reason_int8_scale_term_0=%llu fallback_reason_int8_uniform_false=%llu fallback_reason_bottom_elempack_ne1=%llu "
              "fallback_reason_disable_global=%llu fallback_reason_disable_prep=%llu fallback_reason_disable_1x1=%llu fallback_reason_disable_3x3s1=%llu fallback_reason_disable_3x3s2=%llu",
              load(g_riscv_int8_coverage_counters.conv_int8_pack1_1x1_fast_count),
              load(g_riscv_int8_coverage_counters.conv_int8_packn_1x1_fast_count),
              load(g_riscv_int8_coverage_counters.conv_int8_packn_3x3s1_fast_count),
              load(g_riscv_int8_coverage_counters.conv_int8_packn_3x3s1_skip_prepare_count),
              load(g_riscv_int8_coverage_counters.conv_int8_packn_3x3s1_skip_padding_count),
              load(g_riscv_int8_coverage_counters.conv_int8_packn_3x3s1_skip_channel_mismatch_count),
              load(g_riscv_int8_coverage_counters.conv_int8_pack1_3x3s1_fast_count),
              load(g_riscv_int8_coverage_counters.conv_int8_pack1_3x3s2_fast_count),
              load(g_riscv_int8_coverage_counters.conv_int8_packout_used_count),
              load(g_riscv_int8_coverage_counters.conv_int8_packout_forced_pack1_count),
              load(g_riscv_int8_coverage_counters.conv_int8_packout_blocked_activation_count),
              load(g_riscv_int8_coverage_counters.conv_int8_packout_q_omp_used_count),
              load(g_riscv_int8_coverage_counters.conv_int8_packout_q_omp_forced_oldmap_count),
              load(g_riscv_int8_coverage_counters.packout_store_vsse32_count),
              load(g_riscv_int8_coverage_counters.packout_store_vsseg_count),
              load(g_riscv_int8_coverage_counters.s2_load_vlse8_count),
              load(g_riscv_int8_coverage_counters.s2_load_vlseg2_count),
              load(g_riscv_int8_coverage_counters.post_vrvv_used_count),
              load(g_riscv_int8_coverage_counters.post_scalar_used_count),
              load(g_riscv_int8_coverage_counters.conv_int8_fallback_count),
              load(g_riscv_int8_coverage_counters.fallback_reason_dims_ne3),
              load(g_riscv_int8_coverage_counters.fallback_reason_dilation_ne1),
              load(g_riscv_int8_coverage_counters.fallback_reason_stride_not_1_or_2),
              load(g_riscv_int8_coverage_counters.fallback_reason_kernel_not_1_or_3),
              load(g_riscv_int8_coverage_counters.fallback_reason_group_or_channel_mismatch),
              load(g_riscv_int8_coverage_counters.fallback_reason_int8_scale_term_0),
              load(g_riscv_int8_coverage_counters.fallback_reason_int8_uniform_false),
              load(g_riscv_int8_coverage_counters.fallback_reason_bottom_elempack_ne1),
              load(g_riscv_int8_coverage_counters.fallback_reason_disable_global),
              load(g_riscv_int8_coverage_counters.fallback_reason_disable_prep),
              load(g_riscv_int8_coverage_counters.fallback_reason_disable_1x1),
              load(g_riscv_int8_coverage_counters.fallback_reason_disable_3x3s1),
              load(g_riscv_int8_coverage_counters.fallback_reason_disable_3x3s2));
}

static inline void riscv_int8_coverage_register_atexit_once()
{
    static std::once_flag g_coverage_once;
    std::call_once(g_coverage_once, []() { atexit(riscv_int8_coverage_dump); });
}

struct riscv_int8_packing_audit_key_t
{
    int kernel_w;
    int kernel_h;
    int stride_w;
    int stride_h;
    int dilation_w;
    int dilation_h;
    int in_elempack;
    int bottom_scalar_elemsize;
    int requested_out_elempack;
    int activation_type;

    bool operator<(const riscv_int8_packing_audit_key_t& b) const
    {
        return std::tie(kernel_w, kernel_h, stride_w, stride_h, dilation_w, dilation_h, in_elempack, bottom_scalar_elemsize, requested_out_elempack, activation_type)
            < std::tie(b.kernel_w, b.kernel_h, b.stride_w, b.stride_h, b.dilation_w, b.dilation_h, b.in_elempack, b.bottom_scalar_elemsize, b.requested_out_elempack, b.activation_type);
    }
};

static std::mutex g_riscv_int8_packing_audit_mutex;
static std::map<riscv_int8_packing_audit_key_t, unsigned long long> g_riscv_int8_packing_audit_hist;

static inline int riscv_int8_packing_audit_enabled()
{
    static const int g_enabled = []() -> int
    {
        const char* env = getenv("NCNN_RISCV_INT8_PACKING_AUDIT");
        return (env && env[0] != '\0' && env[0] != '0') ? 1 : 0;
    }();
    return g_enabled;
}

static inline void riscv_int8_packing_audit_record(int kernel_w, int kernel_h, int stride_w, int stride_h, int dilation_w, int dilation_h, int in_elempack, int bottom_scalar_elemsize, int requested_out_elempack, int activation_type)
{
    if (!riscv_int8_packing_audit_enabled())
        return;

    riscv_int8_packing_audit_key_t k;
    k.kernel_w = kernel_w;
    k.kernel_h = kernel_h;
    k.stride_w = stride_w;
    k.stride_h = stride_h;
    k.dilation_w = dilation_w;
    k.dilation_h = dilation_h;
    k.in_elempack = in_elempack;
    k.bottom_scalar_elemsize = bottom_scalar_elemsize;
    k.requested_out_elempack = requested_out_elempack;
    k.activation_type = activation_type;

    std::lock_guard<std::mutex> lock(g_riscv_int8_packing_audit_mutex);
    g_riscv_int8_packing_audit_hist[k] += 1;
}

static void riscv_int8_packing_audit_dump()
{
    if (!riscv_int8_packing_audit_enabled())
        return;

    typedef std::pair<riscv_int8_packing_audit_key_t, unsigned long long> packing_audit_item_t;
    std::vector<packing_audit_item_t> items;
    {
        std::lock_guard<std::mutex> lock(g_riscv_int8_packing_audit_mutex);
        items.reserve(g_riscv_int8_packing_audit_hist.size());
        for (std::map<riscv_int8_packing_audit_key_t, unsigned long long>::const_iterator it = g_riscv_int8_packing_audit_hist.begin(); it != g_riscv_int8_packing_audit_hist.end(); ++it)
        {
            items.push_back(*it);
        }
    }

    std::sort(items.begin(), items.end(), [](const packing_audit_item_t& a, const packing_audit_item_t& b)
    {
        if (a.second != b.second)
            return a.second > b.second;
        return a.first < b.first;
    });

    NCNN_LOGE("riscv_int8_packing_audit bins=%d", (int)items.size());
    for (size_t i = 0; i < items.size(); i++)
    {
        const riscv_int8_packing_audit_key_t& k = items[i].first;
        const unsigned long long count = items[i].second;
        NCNN_LOGE("riscv_int8_packing_audit count=%llu k=%dx%d s=%dx%d d=%dx%d in_pack=%d in_scalar_elemsize=%d req_out_pack=%d activation_type=%d",
                  count,
                  k.kernel_w, k.kernel_h,
                  k.stride_w, k.stride_h,
                  k.dilation_w, k.dilation_h,
                  k.in_elempack, k.bottom_scalar_elemsize,
                  k.requested_out_elempack, k.activation_type);
    }
}

static inline void riscv_int8_packing_audit_register_atexit_once()
{
    static std::once_flag g_packing_audit_once;
    std::call_once(g_packing_audit_once, []() { atexit(riscv_int8_packing_audit_dump); });
}

static inline signed char float2int8_rvv(float v)
{
    int int32 = (int)(v >= 0.f ? v + 0.5f : v - 0.5f);
    if (int32 > 127) return 127;
    if (int32 < -127) return -127;
    return (signed char)int32;
}

static inline int riscv_int8_conv_disabled()
{
    static const int g_disable = []() -> int
    {
        const char* env = getenv("NCNN_RISCV_INT8_CONV_DISABLE");
        return (env && env[0] != '\0' && env[0] != '0') ? 1 : 0;
    }();
    return g_disable;
}

static inline int riscv_int8_conv1x1_disabled()
{
    static const int g_disable = []() -> int
    {
        const char* env = getenv("NCNN_RISCV_INT8_CONV_DISABLE_1X1");
        return (env && env[0] != '\0' && env[0] != '0') ? 1 : 0;
    }();
    return g_disable;
}

static inline int riscv_int8_conv_packn_disabled()
{
    static const int g_disable = []() -> int
    {
        const char* env = getenv("NCNN_RISCV_INT8_CONV_PACKN_DISABLE");
        return (env && env[0] != '\0' && env[0] != '0') ? 1 : 0;
    }();
    return g_disable;
}

static inline int riscv_int8_conv1x1_packn_disabled()
{
    static const int g_disable = []() -> int
    {
        const char* env = getenv("NCNN_RISCV_INT8_CONV_1X1_PACKN_DISABLE");
        return (env && env[0] != '\0' && env[0] != '0') ? 1 : 0;
    }();
    return g_disable;
}

static inline int riscv_int8_conv3x3_disabled()
{
    static const int g_disable = []() -> int
    {
        const char* env = getenv("NCNN_RISCV_INT8_CONV_DISABLE_3X3");
        return (env && env[0] != '\0' && env[0] != '0') ? 1 : 0;
    }();
    return g_disable;
}

static inline int riscv_int8_conv3x3s1_disabled()
{
    static const int g_disable = []() -> int
    {
        const char* env = getenv("NCNN_RISCV_INT8_CONV_DISABLE_3X3S1");
        return (env && env[0] != '\0' && env[0] != '0') ? 1 : 0;
    }();
    return g_disable;
}

static inline int riscv_int8_conv3x3s1_packn_disabled()
{
    static const int g_disable = []() -> int
    {
        const char* env = getenv("NCNN_RISCV_INT8_CONV_3X3S1_PACKN_DISABLE");
        return (env && env[0] != '\0' && env[0] != '0') ? 1 : 0;
    }();
    return g_disable;
}

static inline int riscv_int8_conv3x3s1_packn_enabled()
{
    static const int g_enable = []() -> int
    {
        const char* env = getenv("NCNN_RISCV_INT8_CONV_3X3S1_PACKN_ENABLE");
        return (env && env[0] != '\0' && env[0] != '0') ? 1 : 0;
    }();
    return g_enable;
}

static inline int riscv_int8_conv3x3s2_disabled()
{
    static const int g_disable = []() -> int
    {
        const char* env = getenv("NCNN_RISCV_INT8_CONV_DISABLE_3X3S2");
        return (env && env[0] != '\0' && env[0] != '0') ? 1 : 0;
    }();
    return g_disable;
}

static inline int riscv_int8_conv_prep_disabled()
{
    static const int g_disable = []() -> int
    {
        const char* env = getenv("NCNN_RISCV_INT8_CONV_DISABLE_PREP");
        return (env && env[0] != '\0' && env[0] != '0') ? 1 : 0;
    }();
    return g_disable;
}

static inline int riscv_int8_conv_post_vrvv_disabled()
{
    static const int g_disable = []() -> int
    {
        const char* env = getenv("NCNN_RISCV_INT8_CONV_POST_VRVV_DISABLE");
        return (env && env[0] != '\0' && env[0] != '0') ? 1 : 0;
    }();
    return g_disable;
}

static inline int riscv_int8_conv_packout_disabled()
{
    static const int g_disable = []() -> int
    {
        const char* env = getenv("NCNN_RISCV_INT8_CONV_PACKOUT_DISABLE");
        return (env && env[0] != '\0' && env[0] != '0') ? 1 : 0;
    }();
    return g_disable;
}

static inline int riscv_int8_conv_packout_omp_q_disabled()
{
    static const int g_disable = []() -> int
    {
        const char* env = getenv("NCNN_RISCV_INT8_CONV_PACKOUT_OMP_Q_DISABLE");
        return (env && env[0] != '\0' && env[0] != '0') ? 1 : 0;
    }();
    return g_disable;
}

static inline int riscv_int8_conv_packout_segstore_disabled()
{
    static const int g_disable = []() -> int
    {
        const char* env = getenv("NCNN_RISCV_INT8_CONV_PACKOUT_SEGSTORE_DISABLE");
        return (env && env[0] != '\0' && env[0] != '0') ? 1 : 0;
    }();
    return g_disable;
}

static inline int riscv_int8_conv_s2_vlseg2_disabled()
{
    static const int g_disable = []() -> int
    {
        const char* env = getenv("NCNN_RISCV_INT8_CONV_S2_VLSEG2_DISABLE");
        return (env && env[0] != '\0' && env[0] != '0') ? 1 : 0;
    }();
    return g_disable;
}

static inline int riscv_int8_conv_force_fma()
{
    static const int g_force = []() -> int
    {
        const char* env = getenv("NCNN_RISCV_INT8_CONV_FORCE_FMA");
        return (env && env[0] != '\0' && env[0] != '0') ? 1 : 0;
    }();
    return g_force;
}

static inline int riscv_int8_conv_no_fma()
{
    static const int g_no = []() -> int
    {
        const char* env = getenv("NCNN_RISCV_INT8_CONV_NO_FMA");
        return (env && env[0] != '\0' && env[0] != '0') ? 1 : 0;
    }();
    return g_no;
}

static inline float riscv_int8_conv_accum_fp32(int sum, float scale_in, float bias, int bias_term, int force_fma, int no_fma)
{
    if (force_fma)
        return std::fmaf((float)sum, scale_in, bias_term ? bias : 0.f);

    float sumfp32;
    if (no_fma)
    {
        volatile float prod = (float)sum * scale_in;
        sumfp32 = prod;
    }
    else
    {
        sumfp32 = (float)sum * scale_in;
    }

    if (bias_term)
        sumfp32 += bias;

    return sumfp32;
}

static inline int* riscv_get_sums_buffer(size_t n)
{
    thread_local std::vector<int> buf;
    if (buf.size() < n)
        buf.resize(n);
    return buf.data();
}

static inline int* riscv_get_sums_buffer_slot(size_t n, int slot)
{
    thread_local std::vector<int> bufs[16];
    if (slot < 0 || slot >= 16)
        slot = 0;
    std::vector<int>& buf = bufs[slot];
    if (buf.size() < n)
        buf.resize(n);
    return buf.data();
}

static inline vint8m1_t riscv_int8_load_stride2_i8m1(const signed char* ptr, size_t vl, int use_s2_vlseg2, int coverage_enabled)
{
#if __riscv_vector
    if (use_s2_vlseg2)
    {
        vuint8m1x2_t _p = __riscv_vlseg2e8_v_u8m1x2((const unsigned char*)ptr, vl);
        if (coverage_enabled)
            g_riscv_int8_coverage_counters.s2_load_vlseg2_count.fetch_add((unsigned long long)vl, std::memory_order_relaxed);
        return __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vget_v_u8m1x2_u8m1(_p, 0));
    }
#endif
    if (coverage_enabled)
        g_riscv_int8_coverage_counters.s2_load_vlse8_count.fetch_add((unsigned long long)vl, std::memory_order_relaxed);
    return __riscv_vlse8_v_i8m1(ptr, 2, vl);
}

static inline void riscv_int8_store_fp32_from_sums(float* outptr, const int* sums, int count, float scale_in, float bias, int bias_term, int force_fma, int no_fma, int activation_type, const Mat& activation_params, int out_stride_elems, int disable_post_vrvv, int coverage_enabled)
{
#if __riscv_vector
    if (!disable_post_vrvv && activation_type == 0)
    {
        int j = 0;
        while (j < count)
        {
            size_t vl = __riscv_vsetvl_e32m4((size_t)(count - j));
            vint32m4_t _sum_i32 = __riscv_vle32_v_i32m4(sums + j, vl);
            vfloat32m4_t _sum = __riscv_vfcvt_f_x_v_f32m4(_sum_i32, vl);

            if (force_fma)
            {
                if (bias_term)
                {
                    vfloat32m4_t _acc = __riscv_vfmv_v_f_f32m4(bias, vl);
                    _sum = __riscv_vfmacc_vf_f32m4(_acc, scale_in, _sum, vl);
                }
                else
                {
                    _sum = __riscv_vfmul_vf_f32m4(_sum, scale_in, vl);
                }
            }
            else
            {
                _sum = __riscv_vfmul_vf_f32m4(_sum, scale_in, vl);
                if (bias_term)
                    _sum = __riscv_vfadd_vf_f32m4(_sum, bias, vl);
            }

            if (out_stride_elems == 1)
            {
                __riscv_vse32_v_f32m4(outptr + j, _sum, vl);
            }
            else
            {
                ptrdiff_t out_stride_bytes = (ptrdiff_t)out_stride_elems * 4;
                __riscv_vsse32_v_f32m4(outptr + (size_t)j * out_stride_elems, out_stride_bytes, _sum, vl);
                if (coverage_enabled)
                    g_riscv_int8_coverage_counters.packout_store_vsse32_count.fetch_add(1, std::memory_order_relaxed);
            }
            j += (int)vl;
        }

        if (coverage_enabled)
            g_riscv_int8_coverage_counters.post_vrvv_used_count.fetch_add(1, std::memory_order_relaxed);
        return;
    }
#endif

    for (int j = 0; j < count; j++)
    {
        float sumfp32 = riscv_int8_conv_accum_fp32(sums[j], scale_in, bias, bias_term, force_fma, no_fma);
        sumfp32 = activation_ss(sumfp32, activation_type, activation_params);
        outptr[(size_t)j * out_stride_elems] = sumfp32;
    }

    if (coverage_enabled)
        g_riscv_int8_coverage_counters.post_scalar_used_count.fetch_add(1, std::memory_order_relaxed);
}

static inline vfloat32m1_t riscv_int8_convert_sums_to_f32m1(const int* sums, size_t vl, float scale_in, float bias, int bias_term, int force_fma)
{
    vint32m1_t _sum_i32 = __riscv_vle32_v_i32m1(sums, vl);
    vfloat32m1_t _sum = __riscv_vfcvt_f_x_v_f32m1(_sum_i32, vl);

    if (force_fma)
    {
        if (bias_term)
        {
            vfloat32m1_t _acc = __riscv_vfmv_v_f_f32m1(bias, vl);
            _sum = __riscv_vfmacc_vf_f32m1(_acc, scale_in, _sum, vl);
        }
        else
        {
            _sum = __riscv_vfmul_vf_f32m1(_sum, scale_in, vl);
        }
    }
    else
    {
        _sum = __riscv_vfmul_vf_f32m1(_sum, scale_in, vl);
        if (bias_term)
            _sum = __riscv_vfadd_vf_f32m1(_sum, bias, vl);
    }

    return _sum;
}

static inline void riscv_int8_store_fp32_packx_from_sums(float* outptr_base, int* sums_lanes[16], int lane_count, int count, const float* scale_in_lanes, const float* bias_lanes, int bias_term, int force_fma, int no_fma, int disable_packout_segstore, int disable_post_vrvv, int coverage_enabled)
{
#if __riscv_vector
    if (!disable_post_vrvv && !disable_packout_segstore && (lane_count == 8 || lane_count == 4))
    {
        int j = 0;
        while (j < count)
        {
            size_t vl = __riscv_vsetvl_e32m1((size_t)(count - j));

            if (lane_count == 8)
            {
                vfloat32m1_t _l0 = riscv_int8_convert_sums_to_f32m1(sums_lanes[0] + j, vl, scale_in_lanes[0], bias_lanes[0], bias_term, force_fma);
                vfloat32m1_t _l1 = riscv_int8_convert_sums_to_f32m1(sums_lanes[1] + j, vl, scale_in_lanes[1], bias_lanes[1], bias_term, force_fma);
                vfloat32m1_t _l2 = riscv_int8_convert_sums_to_f32m1(sums_lanes[2] + j, vl, scale_in_lanes[2], bias_lanes[2], bias_term, force_fma);
                vfloat32m1_t _l3 = riscv_int8_convert_sums_to_f32m1(sums_lanes[3] + j, vl, scale_in_lanes[3], bias_lanes[3], bias_term, force_fma);
                vfloat32m1_t _l4 = riscv_int8_convert_sums_to_f32m1(sums_lanes[4] + j, vl, scale_in_lanes[4], bias_lanes[4], bias_term, force_fma);
                vfloat32m1_t _l5 = riscv_int8_convert_sums_to_f32m1(sums_lanes[5] + j, vl, scale_in_lanes[5], bias_lanes[5], bias_term, force_fma);
                vfloat32m1_t _l6 = riscv_int8_convert_sums_to_f32m1(sums_lanes[6] + j, vl, scale_in_lanes[6], bias_lanes[6], bias_term, force_fma);
                vfloat32m1_t _l7 = riscv_int8_convert_sums_to_f32m1(sums_lanes[7] + j, vl, scale_in_lanes[7], bias_lanes[7], bias_term, force_fma);
                vfloat32m1x8_t _pack = __riscv_vcreate_v_f32m1x8(_l0, _l1, _l2, _l3, _l4, _l5, _l6, _l7);
                __riscv_vsseg8e32_v_f32m1x8(outptr_base + (size_t)j * lane_count, _pack, vl);
            }
            else
            {
                vfloat32m1_t _l0 = riscv_int8_convert_sums_to_f32m1(sums_lanes[0] + j, vl, scale_in_lanes[0], bias_lanes[0], bias_term, force_fma);
                vfloat32m1_t _l1 = riscv_int8_convert_sums_to_f32m1(sums_lanes[1] + j, vl, scale_in_lanes[1], bias_lanes[1], bias_term, force_fma);
                vfloat32m1_t _l2 = riscv_int8_convert_sums_to_f32m1(sums_lanes[2] + j, vl, scale_in_lanes[2], bias_lanes[2], bias_term, force_fma);
                vfloat32m1_t _l3 = riscv_int8_convert_sums_to_f32m1(sums_lanes[3] + j, vl, scale_in_lanes[3], bias_lanes[3], bias_term, force_fma);
                vfloat32m1x4_t _pack = __riscv_vcreate_v_f32m1x4(_l0, _l1, _l2, _l3);
                __riscv_vsseg4e32_v_f32m1x4(outptr_base + (size_t)j * lane_count, _pack, vl);
            }

            if (coverage_enabled)
                g_riscv_int8_coverage_counters.packout_store_vsseg_count.fetch_add(1, std::memory_order_relaxed);

            j += (int)vl;
        }

        if (coverage_enabled)
            g_riscv_int8_coverage_counters.post_vrvv_used_count.fetch_add(1, std::memory_order_relaxed);
        return;
    }
#endif

    for (int lane = 0; lane < lane_count; lane++)
    {
        riscv_int8_store_fp32_from_sums(outptr_base + lane, sums_lanes[lane], count, scale_in_lanes[lane], bias_lanes[lane], bias_term, force_fma, no_fma, 0, Mat(), lane_count, disable_post_vrvv, coverage_enabled);
    }
}

static int quantize_to_int8_pack1(const Mat& src, Mat& dst, float scale, const Option& opt)
{
    const int w = src.w;
    const int h = src.h;
    const int channels = src.c;
    const int elempack = src.elempack;
    const int outc = channels * elempack;

    dst.create(w, h, outc, (size_t)1u, opt.blob_allocator);
    if (dst.empty())
        return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* ptr = src.channel(q);
        std::vector<signed char*> outptrs(elempack);
        for (int k = 0; k < elempack; k++)
            outptrs[k] = dst.channel(q * elempack + k);

        for (int i = 0; i < w * h; i++)
        {
            const float* v = ptr + i * elempack;
            for (int k = 0; k < elempack; k++)
                outptrs[k][i] = float2int8_rvv(v[k] * scale);
        }
    }

    return 0;
}

static int quantize_to_int8_packed(const Mat& src, Mat& dst, float scale, const Option& opt)
{
    const int w = src.w;
    const int h = src.h;
    const int channels = src.c;
    const int elempack = src.elempack;

    dst.create(w, h, channels, (size_t)1u * elempack, elempack, opt.blob_allocator);
    if (dst.empty())
        return -100;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        const float* ptr = src.channel(q);
        signed char* outptr = dst.channel(q);

        const int size = w * h * elempack;
        for (int i = 0; i < size; i++)
        {
            outptr[i] = float2int8_rvv(ptr[i] * scale);
        }
    }

    return 0;
}

#if NCNN_RISCV_INT8_CONV_STATS
static std::atomic<int> g_riscv_int8_elemsize_1(0);
static std::atomic<int> g_riscv_int8_elemsize_2(0);
static std::atomic<int> g_riscv_int8_elemsize_4(0);
static std::atomic<int> g_riscv_int8_elempack_ne1(0);
#endif
#endif // NCNN_INT8

#include "convolution_sgemm.h"
#include "convolution_winograd_transform.h"
#include "convolution_winograd_dot.h"
#include "convolution_1x1.h"
#include "convolution_3x3.h"

#if __riscv_vector
#include "convolution_packn.h"
#include "convolution_pack1ton.h"
#include "convolution_packnto1.h"

#include "convolution_sgemm_packn.h"
#include "convolution_sgemm_pack1ton.h"
#include "convolution_sgemm_packnto1.h"
#include "convolution_winograd_transform_packn.h"
#include "convolution_winograd_dot_packn.h"
#include "convolution_1x1_packn.h"
#include "convolution_1x1_pack1ton.h"
#include "convolution_1x1_packnto1.h"
#include "convolution_3x3_packn.h"
#include "convolution_3x3_pack1ton.h"
#include "convolution_7x7_pack1ton.h"
#endif // __riscv_vector

Convolution_riscv::Convolution_riscv()
{
#if __riscv_vector
    support_packing = true;
#endif // __riscv_vector
#if NCNN_ZFH
#if __riscv_vector
    support_fp16_storage = cpu_support_riscv_zvfh();
#else
    support_fp16_storage = cpu_support_riscv_zfh();
#endif
#endif

    activation = 0;
#if NCNN_INT8
    weight_data_int8_1x1_xsmtvdot_mode = CONVOLUTION_1X1_INT8_XSMTVDOT_PATH_NONE;
#endif
}

static void convolution_transform_kernel_packed_rvv(const Mat& weight_data, Mat& weight_data_tm, int num_input, int num_output, int kernel_w, int kernel_h, int elempack, int out_elempack)
{
    const int maxk = kernel_w * kernel_h;

    // src = kw-kh-inch-outch
    // dst = pb-pa-kw-kh-inch/pa-outch/pb
    {
        Mat weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

        weight_data_tm.create(maxk, num_input / elempack, num_output / out_elempack, (size_t)4u * elempack * out_elempack, elempack * out_elempack);

        for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
        {
            float* g00 = weight_data_tm.channel(q / out_elempack);

            for (int p = 0; p + (elempack - 1) < num_input; p += elempack)
            {
                for (int k = 0; k < maxk; k++)
                {
                    for (int i = 0; i < elempack; i++)
                    {
                        for (int j = 0; j < out_elempack; j++)
                        {
                            const float* k00 = weight_data_r2.channel(q + j).row(p + i);

                            g00[0] = k00[k];

                            g00++;
                        }
                    }
                }
            }
        }
    }
}

int Convolution_riscv::create_pipeline(const Option& opt)
{
    if (dynamic_weight)
        return 0;

    activation = create_activation_layer(activation_type, activation_params, opt);

    const int maxk = kernel_w * kernel_h;
    const int num_input = weight_data_size / maxk / num_output;

#if NCNN_INT8
    if (opt.use_int8_inference && weight_data.elemsize == (size_t)1u)
    {
#if __riscv_vector
        weight_data_int8_1x1_packn_tm.release();
        weight_data_int8_3x3s1_packn_tm.release();
        weight_data_int8_1x1_xsmtvdot_tm.release();
        weight_data_int8_1x1_xsmtvdot_mode = CONVOLUTION_1X1_INT8_XSMTVDOT_PATH_NONE;

        const int packn_fp32 = std::max(1, csrr_vlenb() / 4);
        if (kernel_w == 1 && kernel_h == 1 && stride_w == 1 && stride_h == 1 && dilation_w == 1 && dilation_h == 1
            && num_input % packn_fp32 == 0 && num_output % packn_fp32 == 0)
        {
            weight_data_int8_1x1_packn_tm.create(num_input * packn_fp32, num_output / packn_fp32, (size_t)1u);
            if (weight_data_int8_1x1_packn_tm.empty())
                return -100;

            const signed char* kptr = (const signed char*)weight_data;
            for (int g = 0; g < num_output / packn_fp32; g++)
            {
                signed char* gptr = weight_data_int8_1x1_packn_tm.row<signed char>(g);
                for (int q = 0; q < num_input; q++)
                {
                    for (int lane = 0; lane < packn_fp32; lane++)
                    {
                        const int p = g * packn_fp32 + lane;
                        gptr[q * packn_fp32 + lane] = kptr[(size_t)p * num_input + q];
                    }
                }
            }
        }

        if (kernel_w == 3 && kernel_h == 3 && stride_w == 1 && stride_h == 1 && dilation_w == 1 && dilation_h == 1
            && num_output % packn_fp32 == 0)
        {
            weight_data_int8_3x3s1_packn_tm.create(num_input * 9 * packn_fp32, num_output / packn_fp32, (size_t)1u);
            if (weight_data_int8_3x3s1_packn_tm.empty())
                return -100;

            const signed char* kptr = (const signed char*)weight_data;
            for (int g = 0; g < num_output / packn_fp32; g++)
            {
                signed char* gptr = weight_data_int8_3x3s1_packn_tm.row<signed char>(g);
                for (int q = 0; q < num_input; q++)
                {
                    for (int k = 0; k < 9; k++)
                    {
                        signed char* wk = gptr + ((size_t)q * 9 + k) * packn_fp32;
                        for (int lane = 0; lane < packn_fp32; lane++)
                        {
                            const int p = g * packn_fp32 + lane;
                            wk[lane] = kptr[(size_t)p * num_input * 9 + (size_t)q * 9 + k];
                        }
                    }
                }
            }
        }

        const int xsmtvdot_pipeline_mode = convolution_1x1_int8_xsmtvdot_pipeline_mode(opt, activation_type);
        if (xsmtvdot_pipeline_mode != CONVOLUTION_1X1_INT8_XSMTVDOT_PATH_NONE
            && kernel_w == 1 && kernel_h == 1 && stride_w == 1 && stride_h == 1 && dilation_w == 1 && dilation_h == 1)
        {
            int ret = convolution_1x1_int8_xsmtvdot_create_weight_tm(weight_data, weight_data_int8_1x1_xsmtvdot_tm, num_input, num_output);
            if (ret != 0)
                return ret;
            weight_data_int8_1x1_xsmtvdot_mode = weight_data_int8_1x1_xsmtvdot_tm.empty() ? CONVOLUTION_1X1_INT8_XSMTVDOT_PATH_NONE : xsmtvdot_pipeline_mode;
        }
#endif
        return 0;
    }
#endif

#if NCNN_ZFH
    if (support_fp16_storage && opt.use_fp16_storage)
    {
        return create_pipeline_fp16s(opt);
    }
#endif

#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
#endif

    int elempack = 1;
    int out_elempack = 1;
#if __riscv_vector
    if (opt.use_packing_layout)
    {
        elempack = num_input % packn == 0 ? packn : 1;
        out_elempack = num_output % packn == 0 ? packn : 1;
    }
#endif

#if __riscv_vector
    // packn
    if (elempack == packn && out_elempack == packn)
    {
        if (opt.use_winograd_convolution && (opt.use_winograd23_convolution || opt.use_winograd43_convolution || opt.use_winograd63_convolution) && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            if ((opt.use_winograd63_convolution && num_input >= packn * 2 && num_output >= packn * 2 && num_input <= packn * 16 && num_output <= packn * 16) || (!opt.use_winograd43_convolution && !opt.use_winograd23_convolution))
                conv3x3s1_winograd63_transform_kernel_packn_rvv(weight_data, weight_winograd63_data, num_input, num_output, opt);
            else if ((opt.use_winograd43_convolution && num_input >= packn * 2 && num_output >= packn * 2) || (!opt.use_winograd63_convolution && !opt.use_winograd23_convolution))
                conv3x3s1_winograd43_transform_kernel_packn_rvv(weight_data, weight_winograd43_data, num_input, num_output, opt);
            else // if (opt.use_winograd23_convolution)
                conv3x3s1_winograd23_transform_kernel_packn_rvv(weight_data, weight_winograd23_data, num_input, num_output, opt);
        }
        else
        {
            convolution_transform_kernel_packed_rvv(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);
        }
    }

    // pack1ton
    if (elempack == 1 && out_elempack == packn)
    {
        convolution_transform_kernel_packed_rvv(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);
    }

    // packnto1
    if (elempack == packn && out_elempack == 1)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            convolution_im2col_sgemm_transform_kernel_packnto1_rvv(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h);
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            convolution_im2col_sgemm_transform_kernel_packnto1_rvv(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h);
        }
        else if (opt.use_sgemm_convolution)
        {
            convolution_im2col_sgemm_transform_kernel_packnto1_rvv(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h);
        }
        else
        {
            convolution_transform_kernel_packed_rvv(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h, elempack, out_elempack);
        }
    }
#endif // __riscv_vector

    // pack1
    if (elempack == 1 && out_elempack == 1)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            convolution_im2col_sgemm_transform_kernel_rvv(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h);
        }
        else if (opt.use_winograd_convolution && (opt.use_winograd23_convolution || opt.use_winograd43_convolution) && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            if ((opt.use_winograd43_convolution && num_input >= 16 && num_output >= 16) || !opt.use_winograd23_convolution)
            {
                conv3x3s1_winograd43_transform_kernel_rvv(weight_data, weight_winograd43_data, num_input, num_output, opt);
            }
            else if (opt.use_winograd23_convolution)
            {
                conv3x3s1_winograd23_transform_kernel_rvv(weight_data, weight_winograd23_data, num_input, num_output, opt);
            }
        }
        else if (opt.use_sgemm_convolution)
        {
            convolution_im2col_sgemm_transform_kernel_rvv(weight_data, weight_data_tm, num_input, num_output, kernel_w, kernel_h);
        }
        else
        {
            weight_data_tm = weight_data;
        }
    }

    if (opt.lightmode)
        weight_data.release();

    return 0;
}

int Convolution_riscv::destroy_pipeline(const Option& opt)
{
    if (activation)
    {
        activation->destroy_pipeline(opt);
        delete activation;
        activation = 0;
    }

#if NCNN_INT8
    weight_data_int8_1x1_packn_tm.release();
    weight_data_int8_3x3s1_packn_tm.release();
    weight_data_int8_1x1_xsmtvdot_tm.release();
    weight_data_int8_1x1_xsmtvdot_mode = CONVOLUTION_1X1_INT8_XSMTVDOT_PATH_NONE;
#endif

    return 0;
}

int Convolution_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
#if NCNN_INT8
    const bool coverage_enabled = riscv_int8_coverage_enabled() != 0;
    if (coverage_enabled)
        riscv_int8_coverage_register_atexit_once();
    const bool packing_audit_enabled = riscv_int8_packing_audit_enabled() != 0;
    if (packing_audit_enabled)
        riscv_int8_packing_audit_register_atexit_once();

    if (coverage_enabled && opt.use_int8_inference && !int8_scale_term)
    {
        g_riscv_int8_coverage_counters.conv_int8_fallback_count.fetch_add(1, std::memory_order_relaxed);
        g_riscv_int8_coverage_counters.fallback_reason_int8_scale_term_0.fetch_add(1, std::memory_order_relaxed);
    }

    if (opt.use_int8_inference && int8_scale_term)
    {
#if NCNN_RISCV_INT8_CONV_STATS
        const size_t stats_elemsize = bottom_blob.elemsize;
        const int stats_elempack = bottom_blob.elempack;
        if (stats_elemsize == 1u)
            g_riscv_int8_elemsize_1.fetch_add(1, std::memory_order_relaxed);
        else if (stats_elemsize == 2u)
            g_riscv_int8_elemsize_2.fetch_add(1, std::memory_order_relaxed);
        else if (stats_elemsize == 4u)
            g_riscv_int8_elemsize_4.fetch_add(1, std::memory_order_relaxed);
        if (stats_elempack != 1)
            g_riscv_int8_elempack_ne1.fetch_add(1, std::memory_order_relaxed);

        double stats_cast_ms = 0.0;
        double stats_pack_ms = 0.0;
        double stats_quant_ms = 0.0;
        int stats_cast_calls = 0;
        int stats_pack_calls = 0;
        int stats_quant_calls = 0;
        auto stats_dump = [&](const char* tag)
        {
            NCNN_LOGE("int8 stats [%s]: elemsize=%zu elempack=%d totals: e1=%d e2=%d e4=%d packne1=%d cast_ms=%.3f pack_ms=%.3f quant_ms=%.3f cast_calls=%d pack_calls=%d quant_calls=%d",
                      tag, stats_elemsize, stats_elempack,
                      g_riscv_int8_elemsize_1.load(std::memory_order_relaxed),
                      g_riscv_int8_elemsize_2.load(std::memory_order_relaxed),
                      g_riscv_int8_elemsize_4.load(std::memory_order_relaxed),
                      g_riscv_int8_elempack_ne1.load(std::memory_order_relaxed),
                      stats_cast_ms, stats_pack_ms, stats_quant_ms,
                      stats_cast_calls, stats_pack_calls, stats_quant_calls);
        };
#endif

        Mat bottom_blob_fp32 = bottom_blob;
        if (bottom_blob_fp32.elembits() == 16)
        {
            Option opt_pack1 = opt;
            opt_pack1.blob_allocator = opt.workspace_allocator;

#if NCNN_RISCV_INT8_CONV_STATS
            double t0 = get_current_time();
#endif
            cast_float16_to_float32(bottom_blob, bottom_blob_fp32, opt_pack1);
#if NCNN_RISCV_INT8_CONV_STATS
            stats_cast_ms += get_current_time() - t0;
            stats_cast_calls += 1;
#endif
        }

        const bool int8_uniform = bottom_blob_int8_scales.w == 1;

#if __riscv_vector
        auto prepare_int8_pack1 = [&](Mat& bottom_blob_int8_pack1) -> int
        {
            if (riscv_int8_conv_prep_disabled())
                return -100;

            auto ensure_pack1 = [&](Mat& m) -> int
            {
                if (m.empty())
                    return -100;
                if (m.elempack == 1)
                    return 0;

                #if NCNN_RISCV_INT8_CONV_DEBUG
                const int inpack = m.elempack;
                #endif
                Option opt_pack1 = opt;
                opt_pack1.blob_allocator = opt.workspace_allocator;
                Mat tmp;
                convert_packing(m, tmp, 1, opt_pack1);
                if (tmp.empty())
                    return -100;
                m = tmp;
#if NCNN_RISCV_INT8_CONV_DEBUG
                static int g_rvv_int8_force_pack1 = 0;
                if (g_rvv_int8_force_pack1 < 8)
                {
                    NCNN_LOGE("rvv int8 prep: force pack1 (input pack=%d)", inpack);
                    g_rvv_int8_force_pack1++;
                }
#endif
                return 0;
            };

            if (bottom_blob_fp32.elembits() == 8)
            {
                if (bottom_blob_fp32.elempack == 1)
                {
                    bottom_blob_int8_pack1 = bottom_blob_fp32;
                    return ensure_pack1(bottom_blob_int8_pack1);
                }

                Option opt_pack1 = opt;
                opt_pack1.blob_allocator = opt.workspace_allocator;

#if NCNN_RISCV_INT8_CONV_STATS
                double t0 = get_current_time();
#endif
                convert_packing(bottom_blob_fp32, bottom_blob_int8_pack1, 1, opt_pack1);
#if NCNN_RISCV_INT8_CONV_STATS
                stats_pack_ms += get_current_time() - t0;
                stats_pack_calls += 1;
#endif
                if (bottom_blob_int8_pack1.empty())
                    return -100;
#if NCNN_RISCV_INT8_CONV_DEBUG
                static int g_rvv_int8_prep_warn = 0;
                if (bottom_blob_int8_pack1.elempack != 1 && g_rvv_int8_prep_warn < 8)
                {
                    NCNN_LOGE("rvv int8 prep warn: int8 input elempack=%d -> out elempack=%d",
                              bottom_blob_fp32.elempack, bottom_blob_int8_pack1.elempack);
                    g_rvv_int8_prep_warn++;
                }
#endif
                return ensure_pack1(bottom_blob_int8_pack1);
            }

            Option opt_g = opt;
            opt_g.blob_allocator = opt.workspace_allocator;
            // Force pack1 output from Quantize for int8 fast paths.
            opt_g.use_packing_layout = false;

            if (bottom_blob_fp32.elempack != 1)
            {
                if (int8_uniform && 0)
                {
#if NCNN_RISCV_INT8_CONV_STATS
                    double t0 = get_current_time();
#endif
                    int ret = quantize_to_int8_pack1(bottom_blob_fp32, bottom_blob_int8_pack1, bottom_blob_int8_scales[0], opt_g);
#if NCNN_RISCV_INT8_CONV_STATS
                    stats_quant_ms += get_current_time() - t0;
                    stats_quant_calls += 1;
#endif
                    return ret;
                }

                Mat bottom_blob_fp32_pack1;
                Option opt_pack1 = opt;
                opt_pack1.blob_allocator = opt.workspace_allocator;

#if NCNN_RISCV_INT8_CONV_STATS
                double t0 = get_current_time();
#endif
                convert_packing(bottom_blob_fp32, bottom_blob_fp32_pack1, 1, opt_pack1);
#if NCNN_RISCV_INT8_CONV_STATS
                stats_pack_ms += get_current_time() - t0;
                stats_pack_calls += 1;
#endif
                if (bottom_blob_fp32_pack1.empty())
                    return -100;

#if NCNN_RISCV_INT8_CONV_STATS
                t0 = get_current_time();
#endif
                quantize_to_int8(bottom_blob_fp32_pack1, bottom_blob_int8_pack1, bottom_blob_int8_scales, opt_g);
#if NCNN_RISCV_INT8_CONV_STATS
                stats_quant_ms += get_current_time() - t0;
                stats_quant_calls += 1;
#endif
                if (bottom_blob_int8_pack1.empty())
                    return -100;
#if NCNN_RISCV_INT8_CONV_DEBUG
                static int g_rvv_int8_prep_warn = 0;
                if (bottom_blob_int8_pack1.elempack != 1 && g_rvv_int8_prep_warn < 8)
                {
                    NCNN_LOGE("rvv int8 prep warn: fp32 input elempack=%d -> out elempack=%d",
                              bottom_blob_fp32.elempack, bottom_blob_int8_pack1.elempack);
                    g_rvv_int8_prep_warn++;
                }
#endif
                return ensure_pack1(bottom_blob_int8_pack1);
            }

#if NCNN_RISCV_INT8_CONV_STATS
            double t0 = get_current_time();
#endif
            quantize_to_int8(bottom_blob_fp32, bottom_blob_int8_pack1, bottom_blob_int8_scales, opt_g);
#if NCNN_RISCV_INT8_CONV_STATS
            stats_quant_ms += get_current_time() - t0;
            stats_quant_calls += 1;
#endif
            if (bottom_blob_int8_pack1.empty())
                return -100;
#if NCNN_RISCV_INT8_CONV_DEBUG
            static int g_rvv_int8_prep_warn = 0;
            if (bottom_blob_int8_pack1.elempack != 1 && g_rvv_int8_prep_warn < 8)
            {
                NCNN_LOGE("rvv int8 prep warn: fp32 input elempack=%d -> out elempack=%d",
                          bottom_blob_fp32.elempack, bottom_blob_int8_pack1.elempack);
                g_rvv_int8_prep_warn++;
            }
#endif
            return ensure_pack1(bottom_blob_int8_pack1);
        };

        auto prepare_int8_packed = [&](Mat& bottom_blob_int8_packed, int target_elempack) -> int
        {
            if (riscv_int8_conv_prep_disabled())
                return -100;

            if (target_elempack <= 1)
                return -100;

            if (bottom_blob_fp32.elembits() == 8)
            {
                bottom_blob_int8_packed = bottom_blob_fp32;
                if (bottom_blob_int8_packed.elempack == target_elempack)
                    return 0;

                Option opt_pack = opt;
                opt_pack.blob_allocator = opt.workspace_allocator;
                Mat tmp;
                convert_packing(bottom_blob_int8_packed, tmp, target_elempack, opt_pack);
                if (tmp.empty())
                    return -100;
                bottom_blob_int8_packed = tmp;
                return 0;
            }

            Mat bottom_blob_fp32_packed = bottom_blob_fp32;
            if (bottom_blob_fp32_packed.elempack != target_elempack)
            {
                Option opt_pack = opt;
                opt_pack.blob_allocator = opt.workspace_allocator;

                Mat tmp;
                convert_packing(bottom_blob_fp32, tmp, target_elempack, opt_pack);
                if (tmp.empty())
                    return -100;

                bottom_blob_fp32_packed = tmp;
            }

            Option opt_q = opt;
            // Temporary packed int8 input is short-lived and should come from workspace allocator.
            opt_q.blob_allocator = opt.workspace_allocator;
            return quantize_to_int8_packed(bottom_blob_fp32_packed, bottom_blob_int8_packed, bottom_blob_int8_scales[0], opt_q);
        };

        const int disable_rvv_int8_conv = riscv_int8_conv_disabled();
        const int disable_rvv_int8_conv1x1 = riscv_int8_conv1x1_disabled();
        const int disable_rvv_int8_conv_packn = riscv_int8_conv_packn_disabled();
        const int disable_rvv_int8_conv1x1_packn = riscv_int8_conv1x1_packn_disabled();
        const int disable_rvv_int8_conv3x3 = riscv_int8_conv3x3_disabled();
        const int disable_rvv_int8_conv3x3s1 = riscv_int8_conv3x3s1_disabled();
        const int disable_rvv_int8_conv3x3s1_packn = riscv_int8_conv3x3s1_packn_disabled();
        const int enable_rvv_int8_conv3x3s1_packn = riscv_int8_conv3x3s1_packn_enabled();
        const int disable_rvv_int8_conv3x3s2 = riscv_int8_conv3x3s2_disabled();
        const int disable_rvv_int8_prep = riscv_int8_conv_prep_disabled();
        const int disable_rvv_int8_post_vrvv = riscv_int8_conv_post_vrvv_disabled();
        const int disable_rvv_int8_packout = riscv_int8_conv_packout_disabled();
        const int disable_rvv_int8_packout_omp_q = riscv_int8_conv_packout_omp_q_disabled();
        const int disable_rvv_int8_packout_segstore = riscv_int8_conv_packout_segstore_disabled();
        const int disable_rvv_int8_s2_vlseg2 = riscv_int8_conv_s2_vlseg2_disabled();
        const int num_input = weight_data_size / num_output / (kernel_w * kernel_h);
        const int channels_unpacked = bottom_blob_fp32.c * bottom_blob_fp32.elempack;
        const bool fallback_dims3 = bottom_blob_fp32.dims == 3;
        const bool fallback_dilation1 = dilation_w == 1 && dilation_h == 1;
        const bool fallback_stride_1_or_2 = (stride_w == 1 && stride_h == 1) || (stride_w == 2 && stride_h == 2);
        const bool fallback_kernel_1_or_3 = (kernel_w == 1 && kernel_h == 1) || (kernel_w == 3 && kernel_h == 3);
        const bool fallback_group_or_channel_match = num_input == channels_unpacked;

        if (!disable_rvv_int8_conv && !disable_rvv_int8_conv1x1 && !disable_rvv_int8_prep && int8_uniform && bottom_blob_fp32.dims == 3 && num_input == channels_unpacked && kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            {
                const int vlenb = csrr_vlenb();
                const int packn_fp32 = std::max(1, vlenb / 4);
                const int in_elempack = bottom_blob_fp32.elempack;
                const bool use_int8_requantize = int8_scale_term > 100;
                const bool packout_layout_candidate = opt.use_packing_layout && !use_int8_requantize && num_output % packn_fp32 == 0;
                const bool packout_candidate = packout_layout_candidate && activation_type == 0;
                if (packing_audit_enabled)
                {
                    const int bottom_scalar_elemsize = in_elempack > 0 ? (int)(bottom_blob_fp32.elemsize / (size_t)in_elempack) : 0;
                    riscv_int8_packing_audit_record(kernel_w, kernel_h, stride_w, stride_h, dilation_w, dilation_h,
                                                    in_elempack, bottom_scalar_elemsize, packout_layout_candidate ? packn_fp32 : 1, activation_type);
                }

                if (!disable_rvv_int8_conv_packn && !disable_rvv_int8_conv1x1_packn
                    && !disable_rvv_int8_packout && !disable_rvv_int8_packout_omp_q
                    && !use_int8_requantize && packout_candidate
                    && packn_fp32 <= 8
                    && in_elempack > 1 && num_input % in_elempack == 0
                    && !weight_data_int8_1x1_packn_tm.empty())
                {
                    Mat bottom_blob_unbordered_packn;
                    if (prepare_int8_packed(bottom_blob_unbordered_packn, in_elempack) == 0)
                    {
                        Mat bottom_blob_bordered_packn;
                        Option opt_pad = opt;
                        make_padding(bottom_blob_unbordered_packn, bottom_blob_bordered_packn, opt_pad);
                        if (!bottom_blob_bordered_packn.empty() && bottom_blob_bordered_packn.elempack == in_elempack)
                        {
                            const int w = bottom_blob_bordered_packn.w;
                            const int h = bottom_blob_bordered_packn.h;
                            const int channels = bottom_blob_bordered_packn.c;
                            if (channels * in_elempack == num_input)
                            {
                                const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
                                const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;
                                const int outw = (w - kernel_extent_w) / stride_w + 1;
                                const int outh = (h - kernel_extent_h) / stride_h + 1;
                                const int out_elempack = packn_fp32;

                                top_blob.create(outw, outh, num_output / out_elempack, (size_t)4u * out_elempack, out_elempack, opt.blob_allocator);
                                if (top_blob.empty())
                                    return -100;

                                if (coverage_enabled)
                                {
                                    g_riscv_int8_coverage_counters.conv_int8_packout_used_count.fetch_add(1, std::memory_order_relaxed);
                                    g_riscv_int8_coverage_counters.conv_int8_packout_q_omp_used_count.fetch_add(1, std::memory_order_relaxed);
                                }

                                #pragma omp parallel num_threads(opt.num_threads)
                                {
                                    int* sums_lanes[16];
                                    for (int lane = 0; lane < out_elempack; lane++)
                                        sums_lanes[lane] = riscv_get_sums_buffer_slot((size_t)vlenb, lane);

                                    #pragma omp for schedule(static)
                                    for (int qg = 0; qg < num_output / out_elempack; qg++)
                                    {
                                        Mat outc = top_blob.channel(qg);
                                        const int p0 = qg * out_elempack;
                                        const signed char* kptr_group = weight_data_int8_1x1_packn_tm.row<const signed char>(qg);
                                        const int force_fma = riscv_int8_conv_force_fma();
                                        const int no_fma = riscv_int8_conv_no_fma();

                                        float scale_in_lanes[16];
                                        float bias_lanes[16];
                                        for (int lane = 0; lane < out_elempack; lane++)
                                        {
                                            const int p = p0 + lane;
                                            scale_in_lanes[lane] = weight_data_int8_scales[p] != 0 ? 1.f / (bottom_blob_int8_scales[0] * weight_data_int8_scales[p]) : 0.f;
                                            bias_lanes[lane] = bias_term ? bias_data[p] : 0.f;
                                        }

                                        for (int i = 0; i < outh; i++)
                                        {
                                            float* outptr = outc.row<float>(i);

                                            for (int j = 0; j < outw; )
                                            {
                                                size_t vl = __riscv_vsetvl_e8m1(outw - j);
                                                vint32m4_t _sum0 = __riscv_vmv_v_x_i32m4(0, vl);
                                                vint32m4_t _sum1 = __riscv_vmv_v_x_i32m4(0, vl);
                                                vint32m4_t _sum2 = __riscv_vmv_v_x_i32m4(0, vl);
                                                vint32m4_t _sum3 = __riscv_vmv_v_x_i32m4(0, vl);
                                                vint32m4_t _sum4 = __riscv_vmv_v_x_i32m4(0, vl);
                                                vint32m4_t _sum5 = __riscv_vmv_v_x_i32m4(0, vl);
                                                vint32m4_t _sum6 = __riscv_vmv_v_x_i32m4(0, vl);
                                                vint32m4_t _sum7 = __riscv_vmv_v_x_i32m4(0, vl);

                                                for (int q = 0; q < channels; q++)
                                                {
                                                    const signed char* sptr = bottom_blob_bordered_packn.channel(q).row<signed char>(i) + (size_t)j * in_elempack;
                                                    const signed char* kptr_q = kptr_group + (size_t)q * in_elempack * out_elempack;

                                                    for (int k = 0; k < in_elempack; k++)
                                                    {
                                                        vint8m1_t _val8 = __riscv_vlse8_v_i8m1(sptr + k, in_elempack, vl);
                                                        vint16m2_t _val16 = __riscv_vsext_vf2_i16m2(_val8, vl);
                                                        const signed char* kptr_qk = kptr_q + (size_t)k * out_elempack;
                                                        if (out_elempack > 0) _sum0 = __riscv_vwmacc_vx_i32m4(_sum0, (short)kptr_qk[0], _val16, vl);
                                                        if (out_elempack > 1) _sum1 = __riscv_vwmacc_vx_i32m4(_sum1, (short)kptr_qk[1], _val16, vl);
                                                        if (out_elempack > 2) _sum2 = __riscv_vwmacc_vx_i32m4(_sum2, (short)kptr_qk[2], _val16, vl);
                                                        if (out_elempack > 3) _sum3 = __riscv_vwmacc_vx_i32m4(_sum3, (short)kptr_qk[3], _val16, vl);
                                                        if (out_elempack > 4) _sum4 = __riscv_vwmacc_vx_i32m4(_sum4, (short)kptr_qk[4], _val16, vl);
                                                        if (out_elempack > 5) _sum5 = __riscv_vwmacc_vx_i32m4(_sum5, (short)kptr_qk[5], _val16, vl);
                                                        if (out_elempack > 6) _sum6 = __riscv_vwmacc_vx_i32m4(_sum6, (short)kptr_qk[6], _val16, vl);
                                                        if (out_elempack > 7) _sum7 = __riscv_vwmacc_vx_i32m4(_sum7, (short)kptr_qk[7], _val16, vl);
                                                    }
                                                }

                                                if (out_elempack > 0) __riscv_vse32_v_i32m4(sums_lanes[0], _sum0, vl);
                                                if (out_elempack > 1) __riscv_vse32_v_i32m4(sums_lanes[1], _sum1, vl);
                                                if (out_elempack > 2) __riscv_vse32_v_i32m4(sums_lanes[2], _sum2, vl);
                                                if (out_elempack > 3) __riscv_vse32_v_i32m4(sums_lanes[3], _sum3, vl);
                                                if (out_elempack > 4) __riscv_vse32_v_i32m4(sums_lanes[4], _sum4, vl);
                                                if (out_elempack > 5) __riscv_vse32_v_i32m4(sums_lanes[5], _sum5, vl);
                                                if (out_elempack > 6) __riscv_vse32_v_i32m4(sums_lanes[6], _sum6, vl);
                                                if (out_elempack > 7) __riscv_vse32_v_i32m4(sums_lanes[7], _sum7, vl);

                                                riscv_int8_store_fp32_packx_from_sums(outptr + (size_t)j * out_elempack, sums_lanes, out_elempack, (int)vl, scale_in_lanes, bias_lanes, bias_term, force_fma, no_fma, disable_rvv_int8_packout_segstore, disable_rvv_int8_post_vrvv, coverage_enabled);

                                                j += (int)vl;
                                            }
                                        }
                                    }
                                }

                                if (coverage_enabled)
                                    g_riscv_int8_coverage_counters.conv_int8_packn_1x1_fast_count.fetch_add(1, std::memory_order_relaxed);

                                return 0;
                            }
                        }
                    }
                }
            }

            Mat bottom_blob_unbordered;
            if (prepare_int8_pack1(bottom_blob_unbordered) != 0)
                return -100;

            Mat bottom_blob_bordered;
            Option opt_pad = opt;
            opt_pad.use_packing_layout = false;
            make_padding(bottom_blob_unbordered, bottom_blob_bordered, opt_pad);
            if (bottom_blob_bordered.empty())
                return -100;

            int w = bottom_blob_bordered.w;
            int h = bottom_blob_bordered.h;
            int channels = bottom_blob_bordered.c;

            const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
            const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

            int outw = (w - kernel_extent_w) / stride_w + 1;
            int outh = (h - kernel_extent_h) / stride_h + 1;

            bool use_int8_requantize = int8_scale_term > 100;

#if NCNN_RISCV_INT8_CONV_DEBUG
            static int g_rvv_int8_conv1x1_seen = 0;
            if (!g_rvv_int8_conv1x1_seen)
            {
                NCNN_LOGE("riscv int8 1x1 conv rvv fast path");
                g_rvv_int8_conv1x1_seen = 1;
            }
            static int g_rvv_int8_conv1x1_info = 0;
            if (!g_rvv_int8_conv1x1_info)
            {
                NCNN_LOGE("rvv int8 1x1 info w=%d h=%d c=%d elempack=%d num_input=%d wscale0=%f inscale0=%f w_elemsize=%zu",
                          w, h, channels, bottom_blob_bordered.elempack, num_input,
                          weight_data_int8_scales.empty() ? 0.f : weight_data_int8_scales[0],
                          bottom_blob_int8_scales.empty() ? 0.f : bottom_blob_int8_scales[0],
                          weight_data.elemsize);
                g_rvv_int8_conv1x1_info = 1;
            }
            static int g_rvv_int8_conv1x1_check = 0;
            static int g_rvv_int8_conv1x1_tail_check = 0;
            static int g_rvv_int8_conv1x1_ref_check = 0;
#endif

            const int vlenb = csrr_vlenb();
            const int packn_fp32 = std::max(1, vlenb / 4);
            const bool packout_layout_candidate = opt.use_packing_layout && !use_int8_requantize && num_output % packn_fp32 == 0;
            const int requested_out_elempack = packout_layout_candidate ? packn_fp32 : 1;
            if (packing_audit_enabled)
            {
                const int bottom_scalar_elemsize = bottom_blob_bordered.elempack > 0 ? (int)(bottom_blob_bordered.elemsize / (size_t)bottom_blob_bordered.elempack) : 0;
                riscv_int8_packing_audit_record(kernel_w, kernel_h, stride_w, stride_h, dilation_w, dilation_h,
                                                bottom_blob_bordered.elempack, bottom_scalar_elemsize, requested_out_elempack, activation_type);
            }
            bool packout_candidate = packout_layout_candidate;
            if (packout_candidate && activation_type != 0)
            {
                packout_candidate = false;
                if (coverage_enabled)
                    g_riscv_int8_coverage_counters.conv_int8_packout_blocked_activation_count.fetch_add(1, std::memory_order_relaxed);
            }
            const int out_elempack = (packout_candidate && !disable_rvv_int8_packout) ? packn_fp32 : 1;
            size_t out_elemsize = (use_int8_requantize ? (size_t)1u : (size_t)4u) * out_elempack;

            if (out_elempack == 1 && !weight_data_int8_1x1_xsmtvdot_tm.empty())
            {
                int xsmtvdot_ret = 1;
                if (weight_data_int8_1x1_xsmtvdot_mode == CONVOLUTION_1X1_INT8_XSMTVDOT_PATH_4X4K_APANEL_EXPERIMENTAL
                    || weight_data_int8_1x1_xsmtvdot_mode == CONVOLUTION_1X1_INT8_XSMTVDOT_PATH_4X4K_APANEL_CLUSTER0_MT_EXPERIMENTAL)
                {
                    xsmtvdot_ret = convolution_1x1_int8_xsmtvdot_forward_4x4k_apanel_experimental(bottom_blob_bordered, top_blob,
                                                                                                  weight_data_int8_1x1_xsmtvdot_tm,
                                                                                                  bias_data,
                                                                                                  bottom_blob_int8_scales,
                                                                                                  weight_data_int8_scales,
                                                                                                  top_blob_int8_scales,
                                                                                                  bias_term,
                                                                                                  int8_scale_term,
                                                                                                  activation_type,
                                                                                                  num_output,
                                                                                                  opt);
                }
                else if (weight_data_int8_1x1_xsmtvdot_mode == CONVOLUTION_1X1_INT8_XSMTVDOT_PATH_LEGACY
                         && !convolution_1x1_int8_xsmtvdot_legacy_safety_gated(bottom_blob_bordered.w, bottom_blob_bordered.h, bottom_blob_bordered.c, num_output))
                {
                    xsmtvdot_ret = convolution_1x1_int8_xsmtvdot_forward(bottom_blob_bordered, top_blob,
                                                                         weight_data_int8_1x1_xsmtvdot_tm,
                                                                         bias_data,
                                                                         bottom_blob_int8_scales,
                                                                         weight_data_int8_scales,
                                                                         top_blob_int8_scales,
                                                                         bias_term,
                                                                         int8_scale_term,
                                                                         activation_type,
                                                                         num_output,
                                                                         opt);
                }
                if (xsmtvdot_ret == 0)
                {
                    if (coverage_enabled)
                        g_riscv_int8_coverage_counters.conv_int8_pack1_1x1_fast_count.fetch_add(1, std::memory_order_relaxed);
                    return 0;
                }
                if (xsmtvdot_ret < 0)
                    return xsmtvdot_ret;
            }

            top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (coverage_enabled)
            {
                if (out_elempack > 1)
                    g_riscv_int8_coverage_counters.conv_int8_packout_used_count.fetch_add(1, std::memory_order_relaxed);
                else if (packout_candidate)
                    g_riscv_int8_coverage_counters.conv_int8_packout_forced_pack1_count.fetch_add(1, std::memory_order_relaxed);
            }

            if (!use_int8_requantize && out_elempack > 1 && !disable_rvv_int8_packout_omp_q)
            {
                if (coverage_enabled)
                    g_riscv_int8_coverage_counters.conv_int8_packout_q_omp_used_count.fetch_add(1, std::memory_order_relaxed);

                #pragma omp parallel num_threads(opt.num_threads)
                {
                    int* sums_lanes[16];

                    #pragma omp for schedule(static)
                    for (int qg = 0; qg < num_output / out_elempack; qg++)
                    {
                        Mat outc = top_blob.channel(qg);
                        const int p0 = qg * out_elempack;
                        const int force_fma = riscv_int8_conv_force_fma();
                        const int no_fma = riscv_int8_conv_no_fma();

                        const signed char* kptr_lanes[16];
                        float scale_in_lanes[16];
                        float bias_lanes[16];
                        for (int lane = 0; lane < out_elempack; lane++)
                        {
                            const int p = p0 + lane;
                            kptr_lanes[lane] = (const signed char*)weight_data + channels * p;
                            scale_in_lanes[lane] = weight_data_int8_scales[p] != 0 ? 1.f / (bottom_blob_int8_scales[0] * weight_data_int8_scales[p]) : 0.f;
                            bias_lanes[lane] = bias_term ? bias_data[p] : 0.f;
                            sums_lanes[lane] = riscv_get_sums_buffer_slot((size_t)vlenb, lane);
                        }

                        for (int i = 0; i < outh; i++)
                        {
                            float* outptr = outc.row<float>(i);

                            for (int j = 0; j < outw; )
                            {
                                size_t vl = __riscv_vsetvl_e8m1(outw - j);

                                for (int lane = 0; lane < out_elempack; lane++)
                                {
                                    vint32m4_t _sum = __riscv_vmv_v_x_i32m4(0, vl);
                                    const signed char* kptr_lane = kptr_lanes[lane];

                                    for (int q = 0; q < channels; q++)
                                    {
                                        const signed char* sptr = bottom_blob_bordered.channel(q).row<signed char>(i) + j;
                                        vint8m1_t _val8 = __riscv_vle8_v_i8m1(sptr, vl);
                                        vint16m2_t _val16 = __riscv_vsext_vf2_i16m2(_val8, vl);
                                        _sum = __riscv_vwmacc_vx_i32m4(_sum, (short)kptr_lane[q], _val16, vl);
                                    }

                                    __riscv_vse32_v_i32m4(sums_lanes[lane], _sum, vl);
                                }

                                riscv_int8_store_fp32_packx_from_sums(outptr + (size_t)j * out_elempack, sums_lanes, out_elempack, (int)vl, scale_in_lanes, bias_lanes, bias_term, force_fma, no_fma, disable_rvv_int8_packout_segstore, disable_rvv_int8_post_vrvv, coverage_enabled);

                                j += vl;
                            }
                        }
                    }
                }
            }
            else
            {
                if (coverage_enabled && !use_int8_requantize && out_elempack > 1)
                    g_riscv_int8_coverage_counters.conv_int8_packout_q_omp_forced_oldmap_count.fetch_add(1, std::memory_order_relaxed);

                #pragma omp parallel num_threads(opt.num_threads)
                {
                    int* sums = riscv_get_sums_buffer((size_t)vlenb);

                    #pragma omp for
                    for (int p = 0; p < num_output; p++)
                    {
                        const int out_channel_index = p / out_elempack;
                        const int out_lane = p % out_elempack;
                        Mat outc = top_blob.channel(out_channel_index);
                        const signed char* kptr = (const signed char*)weight_data + channels * p;

                        float scale_in = 0.f;
                        if (weight_data_int8_scales[p] != 0)
                            scale_in = 1.f / (bottom_blob_int8_scales[0] * weight_data_int8_scales[p]);

                        float bias = bias_term ? bias_data[p] : 0.f;
                        float scale_out = use_int8_requantize ? top_blob_int8_scales[0] : 0.f;
                        const int force_fma = riscv_int8_conv_force_fma();
                        const int no_fma = riscv_int8_conv_no_fma();

                        for (int i = 0; i < outh; i++)
                        {
                            if (use_int8_requantize)
                            {
                                signed char* outptr = outc.row<signed char>(i) + out_lane;

                                for (int j = 0; j < outw; )
                                {
                                    size_t vl = __riscv_vsetvl_e8m1(outw - j);
                                    vint32m4_t _sum = __riscv_vmv_v_x_i32m4(0, vl);

                                    for (int q = 0; q < channels; q++)
                                    {
                                        const signed char* sptr = bottom_blob_bordered.channel(q).row<signed char>(i) + j;
                                        vint8m1_t _val8 = __riscv_vle8_v_i8m1(sptr, vl);
                                        vint16m2_t _val16 = __riscv_vsext_vf2_i16m2(_val8, vl);
                                        _sum = __riscv_vwmacc_vx_i32m4(_sum, (short)kptr[q], _val16, vl);
                                    }

                                    __riscv_vse32_v_i32m4(sums, _sum, vl);

                                    for (size_t jj = 0; jj < vl; jj++)
                                    {
                                        float sumfp32 = riscv_int8_conv_accum_fp32(sums[jj], scale_in, bias, bias_term, force_fma, no_fma);
                                        sumfp32 = activation_ss(sumfp32, activation_type, activation_params);
                                        outptr[(j + jj) * out_elempack] = float2int8_rvv(sumfp32 * scale_out);
                                    }

                                    j += vl;
                                }
                            }
                            else
                            {
                                float* outptr = outc.row<float>(i) + out_lane;

                                for (int j = 0; j < outw; )
                                {
                                    size_t vl = __riscv_vsetvl_e8m1(outw - j);
                                    vint32m4_t _sum = __riscv_vmv_v_x_i32m4(0, vl);

                                    for (int q = 0; q < channels; q++)
                                    {
                                        const signed char* sptr = bottom_blob_bordered.channel(q).row<signed char>(i) + j;
                                        vint8m1_t _val8 = __riscv_vle8_v_i8m1(sptr, vl);
                                        vint16m2_t _val16 = __riscv_vsext_vf2_i16m2(_val8, vl);
                                        _sum = __riscv_vwmacc_vx_i32m4(_sum, (short)kptr[q], _val16, vl);
                                    }

                                    __riscv_vse32_v_i32m4(sums, _sum, vl);

#if NCNN_RISCV_INT8_CONV_DEBUG
                                    if (!g_rvv_int8_conv1x1_check && p == 0 && i == 0 && j == 0)
                                    {
                                        int sum_scalar = 0;
                                        for (int q = 0; q < channels; q++)
                                        {
                                            const signed char* sptr0 = bottom_blob_bordered.channel(q).row<signed char>(i) + j;
                                            sum_scalar += sptr0[0] * kptr[q];
                                        }
                                        NCNN_LOGE("rvv int8 1x1 check vec=%d scalar=%d scale_in=%f bias=%f",
                                                  sums[0], sum_scalar, scale_in, bias);
                                        g_rvv_int8_conv1x1_check = 1;
                                    }
                                    if (!g_rvv_int8_conv1x1_tail_check && p == 0 && i == 0 && j + (int)vl == outw)
                                    {
                                        int sum_scalar = 0;
                                        int j_tail = outw - 1;
                                        for (int q = 0; q < channels; q++)
                                        {
                                            const signed char* sptr0 = bottom_blob_bordered.channel(q).row<signed char>(i) + j_tail;
                                            sum_scalar += sptr0[0] * kptr[q];
                                        }
                                        NCNN_LOGE("rvv int8 1x1 tail vec=%d scalar=%d j=%d",
                                                  sums[vl - 1], sum_scalar, j_tail);
                                        g_rvv_int8_conv1x1_tail_check = 1;
                                    }
#endif
                                    riscv_int8_store_fp32_from_sums(outptr + (size_t)j * out_elempack, sums, (int)vl, scale_in, bias, bias_term, force_fma, no_fma, activation_type, activation_params, out_elempack, disable_rvv_int8_post_vrvv, coverage_enabled);

                                    j += vl;
                                }
                            }
                        }
                    }
                }
            }

#if NCNN_RISCV_INT8_CONV_STATS
            stats_dump("rvv1x1");
#endif
#if NCNN_RISCV_INT8_CONV_DEBUG
            if (g_rvv_int8_conv1x1_ref_check < 128)
            {
                Mat bottom_blob_ref = bottom_blob_fp32;
                if (bottom_blob_ref.elempack != 1)
                {
                    Option opt_pack1 = opt;
                    opt_pack1.blob_allocator = opt.workspace_allocator;
                    convert_packing(bottom_blob_fp32, bottom_blob_ref, 1, opt_pack1);
                }

                Option opt_ref = opt;
                opt_ref.use_packing_layout = false;
                Mat top_blob_ref;
                int ret = Convolution::forward_int8(bottom_blob_ref, top_blob_ref, opt_ref);
                if (ret == 0 && top_blob_ref.elemsize == top_blob.elemsize)
                {
                    const size_t total = top_blob.total();
                    float max_abs = 0.f;
                    double mean_abs = 0.0;
                    if (top_blob.elemsize == 4u)
                    {
                        const float* a = (const float*)top_blob.data;
                        const float* b = (const float*)top_blob_ref.data;
                        for (size_t i = 0; i < total; i++)
                        {
                            float d = fabsf(a[i] - b[i]);
                            max_abs = std::max(max_abs, d);
                            mean_abs += d;
                        }
                        mean_abs /= (double)total;
                    }
                    else if (top_blob.elemsize == 1u)
                    {
                        const signed char* a = (const signed char*)top_blob.data;
                        const signed char* b = (const signed char*)top_blob_ref.data;
                        for (size_t i = 0; i < total; i++)
                        {
                            float d = (float)abs((int)a[i] - (int)b[i]);
                            max_abs = std::max(max_abs, d);
                            mean_abs += d;
                        }
                        mean_abs /= (double)total;
                    }
                    NCNN_LOGE("rvv int8 1x1 refcheck[%d] name=%s max_abs=%f mean_abs=%f elems=%zu outw=%d outh=%d inch=%d inpack=%d outch=%d",
                              g_rvv_int8_conv1x1_ref_check, name.c_str(), max_abs, (float)mean_abs, total,
                              outw, outh, channels, bottom_blob_bordered.elempack, top_blob.c);
                }
                else
                {
                    NCNN_LOGE("rvv int8 1x1 refcheck[%d] skipped ret=%d ref_elemsize=%zu outw=%d outh=%d inch=%d inpack=%d outch=%d",
                              g_rvv_int8_conv1x1_ref_check, ret, top_blob_ref.elemsize,
                              outw, outh, channels, bottom_blob_bordered.elempack, top_blob.c);
                }
                g_rvv_int8_conv1x1_ref_check++;
            }
#endif
            if (coverage_enabled)
                g_riscv_int8_coverage_counters.conv_int8_pack1_1x1_fast_count.fetch_add(1, std::memory_order_relaxed);
            return 0;
        }

        if (!disable_rvv_int8_conv && !disable_rvv_int8_conv3x3 && !disable_rvv_int8_conv3x3s1 && !disable_rvv_int8_prep && int8_uniform && bottom_blob_fp32.dims == 3 && num_input == channels_unpacked && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            const bool use_int8_requantize_packn = int8_scale_term > 100;
            const int in_elempack = bottom_blob_fp32.elempack;
            const int vlenb_packn = csrr_vlenb();
            const int packn_fp32_packn = std::max(1, vlenb_packn / 4);
            const bool packout_layout_candidate_packn = opt.use_packing_layout && !use_int8_requantize_packn && num_output % packn_fp32_packn == 0;
            if (!disable_rvv_int8_conv_packn && !disable_rvv_int8_conv3x3s1_packn && enable_rvv_int8_conv3x3s1_packn
                && !disable_rvv_int8_packout && !disable_rvv_int8_packout_omp_q
                && packn_fp32_packn <= 8
                && in_elempack > 1 && num_input % in_elempack == 0
                && packout_layout_candidate_packn && activation_type == 0
                && !weight_data_int8_3x3s1_packn_tm.empty())
            {
                if (packing_audit_enabled)
                {
                    const int bottom_scalar_elemsize = in_elempack > 0 ? (int)(bottom_blob_fp32.elemsize / (size_t)in_elempack) : 0;
                    riscv_int8_packing_audit_record(kernel_w, kernel_h, stride_w, stride_h, dilation_w, dilation_h,
                                                    in_elempack, bottom_scalar_elemsize, packn_fp32_packn, activation_type);
                }

                Mat bottom_blob_fp32_packn = bottom_blob_fp32;
                if (bottom_blob_fp32_packn.elempack != in_elempack)
                {
                    Option opt_pack = opt;
                    opt_pack.blob_allocator = opt.workspace_allocator;
                    Mat tmp;
                    convert_packing(bottom_blob_fp32_packn, tmp, in_elempack, opt_pack);
                    if (tmp.empty())
                    {
                        if (coverage_enabled)
                            g_riscv_int8_coverage_counters.conv_int8_packn_3x3s1_skip_prepare_count.fetch_add(1, std::memory_order_relaxed);
                        goto PACKN_3X3S1_FALLBACK;
                    }
                    bottom_blob_fp32_packn = tmp;
                }

                {
                    Mat bottom_blob_bordered_fp32_packn;
                    Option opt_pad = opt;
                    opt_pad.use_packing_layout = false;
                    make_padding(bottom_blob_fp32_packn, bottom_blob_bordered_fp32_packn, opt_pad);
                    if (bottom_blob_bordered_fp32_packn.empty())
                    {
                        if (coverage_enabled)
                            g_riscv_int8_coverage_counters.conv_int8_packn_3x3s1_skip_padding_count.fetch_add(1, std::memory_order_relaxed);
                        goto PACKN_3X3S1_FALLBACK;
                    }

                    if (bottom_blob_bordered_fp32_packn.elempack != in_elempack)
                    {
                        Option opt_pack = opt;
                        opt_pack.blob_allocator = opt.workspace_allocator;
                        Mat tmp_packn;
                        convert_packing(bottom_blob_bordered_fp32_packn, tmp_packn, in_elempack, opt_pack);
                        if (tmp_packn.empty())
                        {
                            if (coverage_enabled)
                                g_riscv_int8_coverage_counters.conv_int8_packn_3x3s1_skip_padding_count.fetch_add(1, std::memory_order_relaxed);
                            goto PACKN_3X3S1_FALLBACK;
                        }
                        bottom_blob_bordered_fp32_packn = tmp_packn;
                    }

                    Mat bottom_blob_bordered_packn;
                    Option opt_q = opt;
                    opt_q.blob_allocator = opt.workspace_allocator;
                    if (quantize_to_int8_packed(bottom_blob_bordered_fp32_packn, bottom_blob_bordered_packn, bottom_blob_int8_scales[0], opt_q) != 0)
                    {
                        if (coverage_enabled)
                            g_riscv_int8_coverage_counters.conv_int8_packn_3x3s1_skip_prepare_count.fetch_add(1, std::memory_order_relaxed);
                        goto PACKN_3X3S1_FALLBACK;
                    }

                    if (!bottom_blob_bordered_packn.empty() && bottom_blob_bordered_packn.elempack == in_elempack)
                    {
                        const int w = bottom_blob_bordered_packn.w;
                        const int h = bottom_blob_bordered_packn.h;
                        const int channels = bottom_blob_bordered_packn.c;
                        if (channels * in_elempack == num_input)
                        {
                            const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
                            const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;
                            const int outw = (w - kernel_extent_w) / stride_w + 1;
                            const int outh = (h - kernel_extent_h) / stride_h + 1;
                            const int out_elempack = packn_fp32_packn;

                            top_blob.create(outw, outh, num_output / out_elempack, (size_t)4u * out_elempack, out_elempack, opt.blob_allocator);
                            if (top_blob.empty())
                                return -100;

                            if (coverage_enabled)
                            {
                                g_riscv_int8_coverage_counters.conv_int8_packout_used_count.fetch_add(1, std::memory_order_relaxed);
                                g_riscv_int8_coverage_counters.conv_int8_packout_q_omp_used_count.fetch_add(1, std::memory_order_relaxed);
                            }

                            #pragma omp parallel num_threads(opt.num_threads)
                            {
                                int* sums_lanes[16];
                                for (int lane = 0; lane < out_elempack; lane++)
                                    sums_lanes[lane] = riscv_get_sums_buffer_slot((size_t)vlenb_packn, lane);

                                #pragma omp for schedule(static)
                                for (int qg = 0; qg < num_output / out_elempack; qg++)
                                {
                                    Mat outc = top_blob.channel(qg);
                                    const int p0 = qg * out_elempack;
                                    const signed char* kptr_group = weight_data_int8_3x3s1_packn_tm.row<const signed char>(qg);
                                    const int force_fma = riscv_int8_conv_force_fma();
                                    const int no_fma = riscv_int8_conv_no_fma();

                                    float scale_in_lanes[16];
                                    float bias_lanes[16];
                                    for (int lane = 0; lane < out_elempack; lane++)
                                    {
                                        const int p = p0 + lane;
                                        scale_in_lanes[lane] = weight_data_int8_scales[p] != 0 ? 1.f / (bottom_blob_int8_scales[0] * weight_data_int8_scales[p]) : 0.f;
                                        bias_lanes[lane] = bias_term ? bias_data[p] : 0.f;
                                    }

                                    for (int i = 0; i < outh; i++)
                                    {
                                        const int in_y = i * stride_h;
                                        float* outptr = outc.row<float>(i);

                                        for (int j = 0; j < outw; )
                                        {
                                            size_t vl = __riscv_vsetvl_e8m1(outw - j);

                                            vint32m4_t _sum0 = __riscv_vmv_v_x_i32m4(0, vl);
                                            vint32m4_t _sum1 = __riscv_vmv_v_x_i32m4(0, vl);
                                            vint32m4_t _sum2 = __riscv_vmv_v_x_i32m4(0, vl);
                                            vint32m4_t _sum3 = __riscv_vmv_v_x_i32m4(0, vl);
                                            vint32m4_t _sum4 = __riscv_vmv_v_x_i32m4(0, vl);
                                            vint32m4_t _sum5 = __riscv_vmv_v_x_i32m4(0, vl);
                                            vint32m4_t _sum6 = __riscv_vmv_v_x_i32m4(0, vl);
                                            vint32m4_t _sum7 = __riscv_vmv_v_x_i32m4(0, vl);

                                            for (int q = 0; q < channels; q++)
                                            {
                                                const signed char* r0 = bottom_blob_bordered_packn.channel(q).row<signed char>(in_y + 0) + (size_t)j * in_elempack;
                                                const signed char* r1 = bottom_blob_bordered_packn.channel(q).row<signed char>(in_y + 1) + (size_t)j * in_elempack;
                                                const signed char* r2 = bottom_blob_bordered_packn.channel(q).row<signed char>(in_y + 2) + (size_t)j * in_elempack;

                                                for (int k = 0; k < in_elempack; k++)
                                                {
                                                    vint8m1_t _r00 = __riscv_vlse8_v_i8m1(r0 + (size_t)0 * in_elempack + k, in_elempack, vl);
                                                    vint8m1_t _r01 = __riscv_vlse8_v_i8m1(r0 + (size_t)1 * in_elempack + k, in_elempack, vl);
                                                    vint8m1_t _r02 = __riscv_vlse8_v_i8m1(r0 + (size_t)2 * in_elempack + k, in_elempack, vl);
                                                    vint8m1_t _r10 = __riscv_vlse8_v_i8m1(r1 + (size_t)0 * in_elempack + k, in_elempack, vl);
                                                    vint8m1_t _r11 = __riscv_vlse8_v_i8m1(r1 + (size_t)1 * in_elempack + k, in_elempack, vl);
                                                    vint8m1_t _r12 = __riscv_vlse8_v_i8m1(r1 + (size_t)2 * in_elempack + k, in_elempack, vl);
                                                    vint8m1_t _r20 = __riscv_vlse8_v_i8m1(r2 + (size_t)0 * in_elempack + k, in_elempack, vl);
                                                    vint8m1_t _r21 = __riscv_vlse8_v_i8m1(r2 + (size_t)1 * in_elempack + k, in_elempack, vl);
                                                    vint8m1_t _r22 = __riscv_vlse8_v_i8m1(r2 + (size_t)2 * in_elempack + k, in_elempack, vl);

                                                    vint16m2_t _r00_16 = __riscv_vsext_vf2_i16m2(_r00, vl);
                                                    vint16m2_t _r01_16 = __riscv_vsext_vf2_i16m2(_r01, vl);
                                                    vint16m2_t _r02_16 = __riscv_vsext_vf2_i16m2(_r02, vl);
                                                    vint16m2_t _r10_16 = __riscv_vsext_vf2_i16m2(_r10, vl);
                                                    vint16m2_t _r11_16 = __riscv_vsext_vf2_i16m2(_r11, vl);
                                                    vint16m2_t _r12_16 = __riscv_vsext_vf2_i16m2(_r12, vl);
                                                    vint16m2_t _r20_16 = __riscv_vsext_vf2_i16m2(_r20, vl);
                                                    vint16m2_t _r21_16 = __riscv_vsext_vf2_i16m2(_r21, vl);
                                                    vint16m2_t _r22_16 = __riscv_vsext_vf2_i16m2(_r22, vl);

                                                    const int scalar_index = q * in_elempack + k;
                                                    const signed char* wk = kptr_group + (size_t)scalar_index * 9 * out_elempack;

                                                    if (out_elempack > 0) _sum0 = __riscv_vwmacc_vx_i32m4(_sum0, (short)wk[0 * out_elempack + 0], _r00_16, vl);
                                                    if (out_elempack > 1) _sum1 = __riscv_vwmacc_vx_i32m4(_sum1, (short)wk[0 * out_elempack + 1], _r00_16, vl);
                                                    if (out_elempack > 2) _sum2 = __riscv_vwmacc_vx_i32m4(_sum2, (short)wk[0 * out_elempack + 2], _r00_16, vl);
                                                    if (out_elempack > 3) _sum3 = __riscv_vwmacc_vx_i32m4(_sum3, (short)wk[0 * out_elempack + 3], _r00_16, vl);
                                                    if (out_elempack > 4) _sum4 = __riscv_vwmacc_vx_i32m4(_sum4, (short)wk[0 * out_elempack + 4], _r00_16, vl);
                                                    if (out_elempack > 5) _sum5 = __riscv_vwmacc_vx_i32m4(_sum5, (short)wk[0 * out_elempack + 5], _r00_16, vl);
                                                    if (out_elempack > 6) _sum6 = __riscv_vwmacc_vx_i32m4(_sum6, (short)wk[0 * out_elempack + 6], _r00_16, vl);
                                                    if (out_elempack > 7) _sum7 = __riscv_vwmacc_vx_i32m4(_sum7, (short)wk[0 * out_elempack + 7], _r00_16, vl);

                                                    if (out_elempack > 0) _sum0 = __riscv_vwmacc_vx_i32m4(_sum0, (short)wk[1 * out_elempack + 0], _r01_16, vl);
                                                    if (out_elempack > 1) _sum1 = __riscv_vwmacc_vx_i32m4(_sum1, (short)wk[1 * out_elempack + 1], _r01_16, vl);
                                                    if (out_elempack > 2) _sum2 = __riscv_vwmacc_vx_i32m4(_sum2, (short)wk[1 * out_elempack + 2], _r01_16, vl);
                                                    if (out_elempack > 3) _sum3 = __riscv_vwmacc_vx_i32m4(_sum3, (short)wk[1 * out_elempack + 3], _r01_16, vl);
                                                    if (out_elempack > 4) _sum4 = __riscv_vwmacc_vx_i32m4(_sum4, (short)wk[1 * out_elempack + 4], _r01_16, vl);
                                                    if (out_elempack > 5) _sum5 = __riscv_vwmacc_vx_i32m4(_sum5, (short)wk[1 * out_elempack + 5], _r01_16, vl);
                                                    if (out_elempack > 6) _sum6 = __riscv_vwmacc_vx_i32m4(_sum6, (short)wk[1 * out_elempack + 6], _r01_16, vl);
                                                    if (out_elempack > 7) _sum7 = __riscv_vwmacc_vx_i32m4(_sum7, (short)wk[1 * out_elempack + 7], _r01_16, vl);

                                                    if (out_elempack > 0) _sum0 = __riscv_vwmacc_vx_i32m4(_sum0, (short)wk[2 * out_elempack + 0], _r02_16, vl);
                                                    if (out_elempack > 1) _sum1 = __riscv_vwmacc_vx_i32m4(_sum1, (short)wk[2 * out_elempack + 1], _r02_16, vl);
                                                    if (out_elempack > 2) _sum2 = __riscv_vwmacc_vx_i32m4(_sum2, (short)wk[2 * out_elempack + 2], _r02_16, vl);
                                                    if (out_elempack > 3) _sum3 = __riscv_vwmacc_vx_i32m4(_sum3, (short)wk[2 * out_elempack + 3], _r02_16, vl);
                                                    if (out_elempack > 4) _sum4 = __riscv_vwmacc_vx_i32m4(_sum4, (short)wk[2 * out_elempack + 4], _r02_16, vl);
                                                    if (out_elempack > 5) _sum5 = __riscv_vwmacc_vx_i32m4(_sum5, (short)wk[2 * out_elempack + 5], _r02_16, vl);
                                                    if (out_elempack > 6) _sum6 = __riscv_vwmacc_vx_i32m4(_sum6, (short)wk[2 * out_elempack + 6], _r02_16, vl);
                                                    if (out_elempack > 7) _sum7 = __riscv_vwmacc_vx_i32m4(_sum7, (short)wk[2 * out_elempack + 7], _r02_16, vl);

                                                    if (out_elempack > 0) _sum0 = __riscv_vwmacc_vx_i32m4(_sum0, (short)wk[3 * out_elempack + 0], _r10_16, vl);
                                                    if (out_elempack > 1) _sum1 = __riscv_vwmacc_vx_i32m4(_sum1, (short)wk[3 * out_elempack + 1], _r10_16, vl);
                                                    if (out_elempack > 2) _sum2 = __riscv_vwmacc_vx_i32m4(_sum2, (short)wk[3 * out_elempack + 2], _r10_16, vl);
                                                    if (out_elempack > 3) _sum3 = __riscv_vwmacc_vx_i32m4(_sum3, (short)wk[3 * out_elempack + 3], _r10_16, vl);
                                                    if (out_elempack > 4) _sum4 = __riscv_vwmacc_vx_i32m4(_sum4, (short)wk[3 * out_elempack + 4], _r10_16, vl);
                                                    if (out_elempack > 5) _sum5 = __riscv_vwmacc_vx_i32m4(_sum5, (short)wk[3 * out_elempack + 5], _r10_16, vl);
                                                    if (out_elempack > 6) _sum6 = __riscv_vwmacc_vx_i32m4(_sum6, (short)wk[3 * out_elempack + 6], _r10_16, vl);
                                                    if (out_elempack > 7) _sum7 = __riscv_vwmacc_vx_i32m4(_sum7, (short)wk[3 * out_elempack + 7], _r10_16, vl);

                                                    if (out_elempack > 0) _sum0 = __riscv_vwmacc_vx_i32m4(_sum0, (short)wk[4 * out_elempack + 0], _r11_16, vl);
                                                    if (out_elempack > 1) _sum1 = __riscv_vwmacc_vx_i32m4(_sum1, (short)wk[4 * out_elempack + 1], _r11_16, vl);
                                                    if (out_elempack > 2) _sum2 = __riscv_vwmacc_vx_i32m4(_sum2, (short)wk[4 * out_elempack + 2], _r11_16, vl);
                                                    if (out_elempack > 3) _sum3 = __riscv_vwmacc_vx_i32m4(_sum3, (short)wk[4 * out_elempack + 3], _r11_16, vl);
                                                    if (out_elempack > 4) _sum4 = __riscv_vwmacc_vx_i32m4(_sum4, (short)wk[4 * out_elempack + 4], _r11_16, vl);
                                                    if (out_elempack > 5) _sum5 = __riscv_vwmacc_vx_i32m4(_sum5, (short)wk[4 * out_elempack + 5], _r11_16, vl);
                                                    if (out_elempack > 6) _sum6 = __riscv_vwmacc_vx_i32m4(_sum6, (short)wk[4 * out_elempack + 6], _r11_16, vl);
                                                    if (out_elempack > 7) _sum7 = __riscv_vwmacc_vx_i32m4(_sum7, (short)wk[4 * out_elempack + 7], _r11_16, vl);

                                                    if (out_elempack > 0) _sum0 = __riscv_vwmacc_vx_i32m4(_sum0, (short)wk[5 * out_elempack + 0], _r12_16, vl);
                                                    if (out_elempack > 1) _sum1 = __riscv_vwmacc_vx_i32m4(_sum1, (short)wk[5 * out_elempack + 1], _r12_16, vl);
                                                    if (out_elempack > 2) _sum2 = __riscv_vwmacc_vx_i32m4(_sum2, (short)wk[5 * out_elempack + 2], _r12_16, vl);
                                                    if (out_elempack > 3) _sum3 = __riscv_vwmacc_vx_i32m4(_sum3, (short)wk[5 * out_elempack + 3], _r12_16, vl);
                                                    if (out_elempack > 4) _sum4 = __riscv_vwmacc_vx_i32m4(_sum4, (short)wk[5 * out_elempack + 4], _r12_16, vl);
                                                    if (out_elempack > 5) _sum5 = __riscv_vwmacc_vx_i32m4(_sum5, (short)wk[5 * out_elempack + 5], _r12_16, vl);
                                                    if (out_elempack > 6) _sum6 = __riscv_vwmacc_vx_i32m4(_sum6, (short)wk[5 * out_elempack + 6], _r12_16, vl);
                                                    if (out_elempack > 7) _sum7 = __riscv_vwmacc_vx_i32m4(_sum7, (short)wk[5 * out_elempack + 7], _r12_16, vl);

                                                    if (out_elempack > 0) _sum0 = __riscv_vwmacc_vx_i32m4(_sum0, (short)wk[6 * out_elempack + 0], _r20_16, vl);
                                                    if (out_elempack > 1) _sum1 = __riscv_vwmacc_vx_i32m4(_sum1, (short)wk[6 * out_elempack + 1], _r20_16, vl);
                                                    if (out_elempack > 2) _sum2 = __riscv_vwmacc_vx_i32m4(_sum2, (short)wk[6 * out_elempack + 2], _r20_16, vl);
                                                    if (out_elempack > 3) _sum3 = __riscv_vwmacc_vx_i32m4(_sum3, (short)wk[6 * out_elempack + 3], _r20_16, vl);
                                                    if (out_elempack > 4) _sum4 = __riscv_vwmacc_vx_i32m4(_sum4, (short)wk[6 * out_elempack + 4], _r20_16, vl);
                                                    if (out_elempack > 5) _sum5 = __riscv_vwmacc_vx_i32m4(_sum5, (short)wk[6 * out_elempack + 5], _r20_16, vl);
                                                    if (out_elempack > 6) _sum6 = __riscv_vwmacc_vx_i32m4(_sum6, (short)wk[6 * out_elempack + 6], _r20_16, vl);
                                                    if (out_elempack > 7) _sum7 = __riscv_vwmacc_vx_i32m4(_sum7, (short)wk[6 * out_elempack + 7], _r20_16, vl);

                                                    if (out_elempack > 0) _sum0 = __riscv_vwmacc_vx_i32m4(_sum0, (short)wk[7 * out_elempack + 0], _r21_16, vl);
                                                    if (out_elempack > 1) _sum1 = __riscv_vwmacc_vx_i32m4(_sum1, (short)wk[7 * out_elempack + 1], _r21_16, vl);
                                                    if (out_elempack > 2) _sum2 = __riscv_vwmacc_vx_i32m4(_sum2, (short)wk[7 * out_elempack + 2], _r21_16, vl);
                                                    if (out_elempack > 3) _sum3 = __riscv_vwmacc_vx_i32m4(_sum3, (short)wk[7 * out_elempack + 3], _r21_16, vl);
                                                    if (out_elempack > 4) _sum4 = __riscv_vwmacc_vx_i32m4(_sum4, (short)wk[7 * out_elempack + 4], _r21_16, vl);
                                                    if (out_elempack > 5) _sum5 = __riscv_vwmacc_vx_i32m4(_sum5, (short)wk[7 * out_elempack + 5], _r21_16, vl);
                                                    if (out_elempack > 6) _sum6 = __riscv_vwmacc_vx_i32m4(_sum6, (short)wk[7 * out_elempack + 6], _r21_16, vl);
                                                    if (out_elempack > 7) _sum7 = __riscv_vwmacc_vx_i32m4(_sum7, (short)wk[7 * out_elempack + 7], _r21_16, vl);

                                                    if (out_elempack > 0) _sum0 = __riscv_vwmacc_vx_i32m4(_sum0, (short)wk[8 * out_elempack + 0], _r22_16, vl);
                                                    if (out_elempack > 1) _sum1 = __riscv_vwmacc_vx_i32m4(_sum1, (short)wk[8 * out_elempack + 1], _r22_16, vl);
                                                    if (out_elempack > 2) _sum2 = __riscv_vwmacc_vx_i32m4(_sum2, (short)wk[8 * out_elempack + 2], _r22_16, vl);
                                                    if (out_elempack > 3) _sum3 = __riscv_vwmacc_vx_i32m4(_sum3, (short)wk[8 * out_elempack + 3], _r22_16, vl);
                                                    if (out_elempack > 4) _sum4 = __riscv_vwmacc_vx_i32m4(_sum4, (short)wk[8 * out_elempack + 4], _r22_16, vl);
                                                    if (out_elempack > 5) _sum5 = __riscv_vwmacc_vx_i32m4(_sum5, (short)wk[8 * out_elempack + 5], _r22_16, vl);
                                                    if (out_elempack > 6) _sum6 = __riscv_vwmacc_vx_i32m4(_sum6, (short)wk[8 * out_elempack + 6], _r22_16, vl);
                                                    if (out_elempack > 7) _sum7 = __riscv_vwmacc_vx_i32m4(_sum7, (short)wk[8 * out_elempack + 7], _r22_16, vl);
                                                }
                                            }

                                            if (out_elempack > 0) __riscv_vse32_v_i32m4(sums_lanes[0], _sum0, vl);
                                            if (out_elempack > 1) __riscv_vse32_v_i32m4(sums_lanes[1], _sum1, vl);
                                            if (out_elempack > 2) __riscv_vse32_v_i32m4(sums_lanes[2], _sum2, vl);
                                            if (out_elempack > 3) __riscv_vse32_v_i32m4(sums_lanes[3], _sum3, vl);
                                            if (out_elempack > 4) __riscv_vse32_v_i32m4(sums_lanes[4], _sum4, vl);
                                            if (out_elempack > 5) __riscv_vse32_v_i32m4(sums_lanes[5], _sum5, vl);
                                            if (out_elempack > 6) __riscv_vse32_v_i32m4(sums_lanes[6], _sum6, vl);
                                            if (out_elempack > 7) __riscv_vse32_v_i32m4(sums_lanes[7], _sum7, vl);

                                            riscv_int8_store_fp32_packx_from_sums(outptr + (size_t)j * out_elempack, sums_lanes, out_elempack, (int)vl, scale_in_lanes, bias_lanes, bias_term, force_fma, no_fma, disable_rvv_int8_packout_segstore, disable_rvv_int8_post_vrvv, coverage_enabled);

                                            j += (int)vl;
                                        }
                                    }
                                }
                            }

                            if (coverage_enabled)
                                g_riscv_int8_coverage_counters.conv_int8_packn_3x3s1_fast_count.fetch_add(1, std::memory_order_relaxed);

                            return 0;
                        }
                        else if (coverage_enabled)
                        {
                            g_riscv_int8_coverage_counters.conv_int8_packn_3x3s1_skip_channel_mismatch_count.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                    else if (coverage_enabled)
                    {
                        g_riscv_int8_coverage_counters.conv_int8_packn_3x3s1_skip_padding_count.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
PACKN_3X3S1_FALLBACK:

            Mat bottom_blob_unbordered;
            if (prepare_int8_pack1(bottom_blob_unbordered) != 0)
                return -100;

            Mat bottom_blob_bordered;
            Option opt_pad = opt;
            opt_pad.use_packing_layout = false;
            make_padding(bottom_blob_unbordered, bottom_blob_bordered, opt_pad);
            if (bottom_blob_bordered.empty())
                return -100;

            int w = bottom_blob_bordered.w;
            int h = bottom_blob_bordered.h;
            int channels = bottom_blob_bordered.c;

            const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
            const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

            int outw = (w - kernel_extent_w) / stride_w + 1;
            int outh = (h - kernel_extent_h) / stride_h + 1;

            bool use_int8_requantize = int8_scale_term > 100;

#if NCNN_RISCV_INT8_CONV_DEBUG
            static int g_rvv_int8_conv3x3s1_seen = 0;
            if (!g_rvv_int8_conv3x3s1_seen)
            {
                NCNN_LOGE("riscv int8 3x3s1 conv rvv fast path");
                g_rvv_int8_conv3x3s1_seen = 1;
            }
            static int g_rvv_int8_conv3x3s1_info = 0;
            if (!g_rvv_int8_conv3x3s1_info)
            {
                NCNN_LOGE("rvv int8 3x3s1 info w=%d h=%d c=%d elempack=%d num_input=%d wscale0=%f inscale0=%f w_elemsize=%zu",
                          w, h, channels, bottom_blob_bordered.elempack, num_input,
                          weight_data_int8_scales.empty() ? 0.f : weight_data_int8_scales[0],
                          bottom_blob_int8_scales.empty() ? 0.f : bottom_blob_int8_scales[0],
                          weight_data.elemsize);
                g_rvv_int8_conv3x3s1_info = 1;
            }
            static int g_rvv_int8_conv3x3s1_check = 0;
            static int g_rvv_int8_conv3x3s1_tail_check = 0;
            static int g_rvv_int8_conv3x3s1_ref_check = 0;
#endif

            const int vlenb = csrr_vlenb();
            const int packn_fp32 = std::max(1, vlenb / 4);
            const bool packout_layout_candidate = opt.use_packing_layout && !use_int8_requantize && num_output % packn_fp32 == 0;
            const int requested_out_elempack = packout_layout_candidate ? packn_fp32 : 1;
            if (packing_audit_enabled)
            {
                const int bottom_scalar_elemsize = bottom_blob_bordered.elempack > 0 ? (int)(bottom_blob_bordered.elemsize / (size_t)bottom_blob_bordered.elempack) : 0;
                riscv_int8_packing_audit_record(kernel_w, kernel_h, stride_w, stride_h, dilation_w, dilation_h,
                                                bottom_blob_bordered.elempack, bottom_scalar_elemsize, requested_out_elempack, activation_type);
            }
            bool packout_candidate = packout_layout_candidate;
            if (packout_candidate && activation_type != 0)
            {
                packout_candidate = false;
                if (coverage_enabled)
                    g_riscv_int8_coverage_counters.conv_int8_packout_blocked_activation_count.fetch_add(1, std::memory_order_relaxed);
            }
            const int out_elempack = (packout_candidate && !disable_rvv_int8_packout) ? packn_fp32 : 1;
            size_t out_elemsize = (use_int8_requantize ? (size_t)1u : (size_t)4u) * out_elempack;

            top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (coverage_enabled)
            {
                if (out_elempack > 1)
                    g_riscv_int8_coverage_counters.conv_int8_packout_used_count.fetch_add(1, std::memory_order_relaxed);
                else if (packout_candidate)
                    g_riscv_int8_coverage_counters.conv_int8_packout_forced_pack1_count.fetch_add(1, std::memory_order_relaxed);
            }

            if (!use_int8_requantize && out_elempack > 1 && !disable_rvv_int8_packout_omp_q)
            {
                if (coverage_enabled)
                    g_riscv_int8_coverage_counters.conv_int8_packout_q_omp_used_count.fetch_add(1, std::memory_order_relaxed);

                #pragma omp parallel num_threads(opt.num_threads)
                {
                    int* sums_lanes[16];

                    #pragma omp for schedule(static)
                    for (int qg = 0; qg < num_output / out_elempack; qg++)
                    {
                        Mat outc = top_blob.channel(qg);
                        const int p0 = qg * out_elempack;
                        const int force_fma = riscv_int8_conv_force_fma();
                        const int no_fma = riscv_int8_conv_no_fma();

                        const signed char* kptr_lanes[16];
                        float scale_in_lanes[16];
                        float bias_lanes[16];
                        for (int lane = 0; lane < out_elempack; lane++)
                        {
                            const int p = p0 + lane;
                            kptr_lanes[lane] = (const signed char*)weight_data + channels * p * 9;
                            scale_in_lanes[lane] = weight_data_int8_scales[p] != 0 ? 1.f / (bottom_blob_int8_scales[0] * weight_data_int8_scales[p]) : 0.f;
                            bias_lanes[lane] = bias_term ? bias_data[p] : 0.f;
                            sums_lanes[lane] = riscv_get_sums_buffer_slot((size_t)vlenb, lane);
                        }

                        for (int i = 0; i < outh; i++)
                        {
                            int in_y = i * stride_h;
                            float* outptr = outc.row<float>(i);

                            for (int j = 0; j < outw; )
                            {
                                size_t vl = __riscv_vsetvl_e8m1(outw - j);

                                for (int lane = 0; lane < out_elempack; lane++)
                                {
                                    vint32m4_t _sum = __riscv_vmv_v_x_i32m4(0, vl);
                                    const signed char* kptr_lane = kptr_lanes[lane];

                                    for (int q = 0; q < channels; q++)
                                    {
                                        const signed char* kptr_q = kptr_lane + q * 9;
                                        const signed char* r0 = bottom_blob_bordered.channel(q).row<signed char>(in_y + 0) + j;
                                        const signed char* r1 = bottom_blob_bordered.channel(q).row<signed char>(in_y + 1) + j;
                                        const signed char* r2 = bottom_blob_bordered.channel(q).row<signed char>(in_y + 2) + j;

                                        vint8m1_t _r00 = __riscv_vle8_v_i8m1(r0 + 0, vl);
                                        vint8m1_t _r01 = __riscv_vle8_v_i8m1(r0 + 1, vl);
                                        vint8m1_t _r02 = __riscv_vle8_v_i8m1(r0 + 2, vl);
                                        vint8m1_t _r10 = __riscv_vle8_v_i8m1(r1 + 0, vl);
                                        vint8m1_t _r11 = __riscv_vle8_v_i8m1(r1 + 1, vl);
                                        vint8m1_t _r12 = __riscv_vle8_v_i8m1(r1 + 2, vl);
                                        vint8m1_t _r20 = __riscv_vle8_v_i8m1(r2 + 0, vl);
                                        vint8m1_t _r21 = __riscv_vle8_v_i8m1(r2 + 1, vl);
                                        vint8m1_t _r22 = __riscv_vle8_v_i8m1(r2 + 2, vl);

                                        vint16m2_t _r00_16 = __riscv_vsext_vf2_i16m2(_r00, vl);
                                        vint16m2_t _r01_16 = __riscv_vsext_vf2_i16m2(_r01, vl);
                                        vint16m2_t _r02_16 = __riscv_vsext_vf2_i16m2(_r02, vl);
                                        vint16m2_t _r10_16 = __riscv_vsext_vf2_i16m2(_r10, vl);
                                        vint16m2_t _r11_16 = __riscv_vsext_vf2_i16m2(_r11, vl);
                                        vint16m2_t _r12_16 = __riscv_vsext_vf2_i16m2(_r12, vl);
                                        vint16m2_t _r20_16 = __riscv_vsext_vf2_i16m2(_r20, vl);
                                        vint16m2_t _r21_16 = __riscv_vsext_vf2_i16m2(_r21, vl);
                                        vint16m2_t _r22_16 = __riscv_vsext_vf2_i16m2(_r22, vl);

                                        _sum = __riscv_vwmacc_vx_i32m4(_sum, (short)kptr_q[0], _r00_16, vl);
                                        _sum = __riscv_vwmacc_vx_i32m4(_sum, (short)kptr_q[1], _r01_16, vl);
                                        _sum = __riscv_vwmacc_vx_i32m4(_sum, (short)kptr_q[2], _r02_16, vl);
                                        _sum = __riscv_vwmacc_vx_i32m4(_sum, (short)kptr_q[3], _r10_16, vl);
                                        _sum = __riscv_vwmacc_vx_i32m4(_sum, (short)kptr_q[4], _r11_16, vl);
                                        _sum = __riscv_vwmacc_vx_i32m4(_sum, (short)kptr_q[5], _r12_16, vl);
                                        _sum = __riscv_vwmacc_vx_i32m4(_sum, (short)kptr_q[6], _r20_16, vl);
                                        _sum = __riscv_vwmacc_vx_i32m4(_sum, (short)kptr_q[7], _r21_16, vl);
                                        _sum = __riscv_vwmacc_vx_i32m4(_sum, (short)kptr_q[8], _r22_16, vl);
                                    }

                                    __riscv_vse32_v_i32m4(sums_lanes[lane], _sum, vl);
                                }

                                riscv_int8_store_fp32_packx_from_sums(outptr + (size_t)j * out_elempack, sums_lanes, out_elempack, (int)vl, scale_in_lanes, bias_lanes, bias_term, force_fma, no_fma, disable_rvv_int8_packout_segstore, disable_rvv_int8_post_vrvv, coverage_enabled);
                                j += vl;
                            }
                        }
                    }
                }
            }
            else
            {
                if (coverage_enabled && !use_int8_requantize && out_elempack > 1)
                    g_riscv_int8_coverage_counters.conv_int8_packout_q_omp_forced_oldmap_count.fetch_add(1, std::memory_order_relaxed);

                #pragma omp parallel num_threads(opt.num_threads)
                {
                    int* sums = riscv_get_sums_buffer((size_t)vlenb);

                    #pragma omp for
                    for (int p = 0; p < num_output; p++)
                    {
                        const int out_channel_index = p / out_elempack;
                        const int out_lane = p % out_elempack;
                        Mat outc = top_blob.channel(out_channel_index);
                        const signed char* kptr = (const signed char*)weight_data + channels * p * 9;

                        float scale_in = 0.f;
                        if (weight_data_int8_scales[p] != 0)
                            scale_in = 1.f / (bottom_blob_int8_scales[0] * weight_data_int8_scales[p]);

                        float bias = bias_term ? bias_data[p] : 0.f;
                        float scale_out = use_int8_requantize ? top_blob_int8_scales[0] : 0.f;
                        const int force_fma = riscv_int8_conv_force_fma();
                        const int no_fma = riscv_int8_conv_no_fma();

                        for (int i = 0; i < outh; i++)
                        {
                            int in_y = i * stride_h;

                            if (use_int8_requantize)
                            {
                                signed char* outptr = outc.row<signed char>(i) + out_lane;

                                for (int j = 0; j < outw; )
                                {
                                    size_t vl = __riscv_vsetvl_e8m1(outw - j);
                                    vint32m4_t _sum = __riscv_vmv_v_x_i32m4(0, vl);

                                    for (int q = 0; q < channels; q++)
                                    {
                                        const signed char* kptr_q = kptr + q * 9;
                                        const signed char* r0 = bottom_blob_bordered.channel(q).row<signed char>(in_y + 0) + j;
                                        const signed char* r1 = bottom_blob_bordered.channel(q).row<signed char>(in_y + 1) + j;
                                        const signed char* r2 = bottom_blob_bordered.channel(q).row<signed char>(in_y + 2) + j;

                                        vint8m1_t _r00 = __riscv_vle8_v_i8m1(r0 + 0, vl);
                                        vint8m1_t _r01 = __riscv_vle8_v_i8m1(r0 + 1, vl);
                                        vint8m1_t _r02 = __riscv_vle8_v_i8m1(r0 + 2, vl);
                                        vint8m1_t _r10 = __riscv_vle8_v_i8m1(r1 + 0, vl);
                                        vint8m1_t _r11 = __riscv_vle8_v_i8m1(r1 + 1, vl);
                                        vint8m1_t _r12 = __riscv_vle8_v_i8m1(r1 + 2, vl);
                                        vint8m1_t _r20 = __riscv_vle8_v_i8m1(r2 + 0, vl);
                                        vint8m1_t _r21 = __riscv_vle8_v_i8m1(r2 + 1, vl);
                                        vint8m1_t _r22 = __riscv_vle8_v_i8m1(r2 + 2, vl);

                                        vint16m2_t _r00_16 = __riscv_vsext_vf2_i16m2(_r00, vl);
                                        vint16m2_t _r01_16 = __riscv_vsext_vf2_i16m2(_r01, vl);
                                        vint16m2_t _r02_16 = __riscv_vsext_vf2_i16m2(_r02, vl);
                                        vint16m2_t _r10_16 = __riscv_vsext_vf2_i16m2(_r10, vl);
                                        vint16m2_t _r11_16 = __riscv_vsext_vf2_i16m2(_r11, vl);
                                        vint16m2_t _r12_16 = __riscv_vsext_vf2_i16m2(_r12, vl);
                                        vint16m2_t _r20_16 = __riscv_vsext_vf2_i16m2(_r20, vl);
                                        vint16m2_t _r21_16 = __riscv_vsext_vf2_i16m2(_r21, vl);
                                        vint16m2_t _r22_16 = __riscv_vsext_vf2_i16m2(_r22, vl);

                                        _sum = __riscv_vwmacc_vx_i32m4(_sum, (short)kptr_q[0], _r00_16, vl);
                                        _sum = __riscv_vwmacc_vx_i32m4(_sum, (short)kptr_q[1], _r01_16, vl);
                                        _sum = __riscv_vwmacc_vx_i32m4(_sum, (short)kptr_q[2], _r02_16, vl);
                                        _sum = __riscv_vwmacc_vx_i32m4(_sum, (short)kptr_q[3], _r10_16, vl);
                                        _sum = __riscv_vwmacc_vx_i32m4(_sum, (short)kptr_q[4], _r11_16, vl);
                                        _sum = __riscv_vwmacc_vx_i32m4(_sum, (short)kptr_q[5], _r12_16, vl);
                                        _sum = __riscv_vwmacc_vx_i32m4(_sum, (short)kptr_q[6], _r20_16, vl);
                                        _sum = __riscv_vwmacc_vx_i32m4(_sum, (short)kptr_q[7], _r21_16, vl);
                                        _sum = __riscv_vwmacc_vx_i32m4(_sum, (short)kptr_q[8], _r22_16, vl);
                                    }

                                    __riscv_vse32_v_i32m4(sums, _sum, vl);

                                    for (size_t jj = 0; jj < vl; jj++)
                                    {
                                        float sumfp32 = riscv_int8_conv_accum_fp32(sums[jj], scale_in, bias, bias_term, force_fma, no_fma);
                                        sumfp32 = activation_ss(sumfp32, activation_type, activation_params);
                                        outptr[(j + jj) * out_elempack] = float2int8_rvv(sumfp32 * scale_out);
                                    }
                                    j += vl;
                                }
                            }
                            else
                            {
                                float* outptr = outc.row<float>(i) + out_lane;

                                for (int j = 0; j < outw; )
                                {
                                    size_t vl = __riscv_vsetvl_e8m1(outw - j);
                                    vint32m4_t _sum = __riscv_vmv_v_x_i32m4(0, vl);

                                    for (int q = 0; q < channels; q++)
                                    {
                                        const signed char* kptr_q = kptr + q * 9;
                                        const signed char* r0 = bottom_blob_bordered.channel(q).row<signed char>(in_y + 0) + j;
                                        const signed char* r1 = bottom_blob_bordered.channel(q).row<signed char>(in_y + 1) + j;
                                        const signed char* r2 = bottom_blob_bordered.channel(q).row<signed char>(in_y + 2) + j;

                                        vint8m1_t _r00 = __riscv_vle8_v_i8m1(r0 + 0, vl);
                                        vint8m1_t _r01 = __riscv_vle8_v_i8m1(r0 + 1, vl);
                                        vint8m1_t _r02 = __riscv_vle8_v_i8m1(r0 + 2, vl);
                                        vint8m1_t _r10 = __riscv_vle8_v_i8m1(r1 + 0, vl);
                                        vint8m1_t _r11 = __riscv_vle8_v_i8m1(r1 + 1, vl);
                                        vint8m1_t _r12 = __riscv_vle8_v_i8m1(r1 + 2, vl);
                                        vint8m1_t _r20 = __riscv_vle8_v_i8m1(r2 + 0, vl);
                                        vint8m1_t _r21 = __riscv_vle8_v_i8m1(r2 + 1, vl);
                                        vint8m1_t _r22 = __riscv_vle8_v_i8m1(r2 + 2, vl);

                                        vint16m2_t _r00_16 = __riscv_vsext_vf2_i16m2(_r00, vl);
                                        vint16m2_t _r01_16 = __riscv_vsext_vf2_i16m2(_r01, vl);
                                        vint16m2_t _r02_16 = __riscv_vsext_vf2_i16m2(_r02, vl);
                                        vint16m2_t _r10_16 = __riscv_vsext_vf2_i16m2(_r10, vl);
                                        vint16m2_t _r11_16 = __riscv_vsext_vf2_i16m2(_r11, vl);
                                        vint16m2_t _r12_16 = __riscv_vsext_vf2_i16m2(_r12, vl);
                                        vint16m2_t _r20_16 = __riscv_vsext_vf2_i16m2(_r20, vl);
                                        vint16m2_t _r21_16 = __riscv_vsext_vf2_i16m2(_r21, vl);
                                        vint16m2_t _r22_16 = __riscv_vsext_vf2_i16m2(_r22, vl);

                                        _sum = __riscv_vwmacc_vx_i32m4(_sum, (short)kptr_q[0], _r00_16, vl);
                                        _sum = __riscv_vwmacc_vx_i32m4(_sum, (short)kptr_q[1], _r01_16, vl);
                                        _sum = __riscv_vwmacc_vx_i32m4(_sum, (short)kptr_q[2], _r02_16, vl);
                                        _sum = __riscv_vwmacc_vx_i32m4(_sum, (short)kptr_q[3], _r10_16, vl);
                                        _sum = __riscv_vwmacc_vx_i32m4(_sum, (short)kptr_q[4], _r11_16, vl);
                                        _sum = __riscv_vwmacc_vx_i32m4(_sum, (short)kptr_q[5], _r12_16, vl);
                                        _sum = __riscv_vwmacc_vx_i32m4(_sum, (short)kptr_q[6], _r20_16, vl);
                                        _sum = __riscv_vwmacc_vx_i32m4(_sum, (short)kptr_q[7], _r21_16, vl);
                                        _sum = __riscv_vwmacc_vx_i32m4(_sum, (short)kptr_q[8], _r22_16, vl);
                                    }

                                    __riscv_vse32_v_i32m4(sums, _sum, vl);

                                    riscv_int8_store_fp32_from_sums(outptr + (size_t)j * out_elempack, sums, (int)vl, scale_in, bias, bias_term, force_fma, no_fma, activation_type, activation_params, out_elempack, disable_rvv_int8_post_vrvv, coverage_enabled);
                                    j += vl;
                                }
                            }
                        }
                    }
                }
            }

#if NCNN_RISCV_INT8_CONV_STATS
            stats_dump("rvv3x3s1");
#endif
#if NCNN_RISCV_INT8_CONV_DEBUG
            auto refcheck_3x3s1 = [&](const char* tag)
            {
                Mat bottom_blob_ref = bottom_blob_fp32;
                if (bottom_blob_ref.elempack != 1)
                {
                    Option opt_pack1 = opt;
                    opt_pack1.blob_allocator = opt.workspace_allocator;
                    convert_packing(bottom_blob_fp32, bottom_blob_ref, 1, opt_pack1);
                }

                Option opt_ref = opt;
                opt_ref.use_packing_layout = false;
                Mat top_blob_ref;
                int ret = Convolution::forward_int8(bottom_blob_ref, top_blob_ref, opt_ref);
                if (ret == 0 && top_blob_ref.elemsize == top_blob.elemsize)
                {
                    const size_t total = top_blob.total();
                    float max_abs = 0.f;
                    double mean_abs = 0.0;
                    if (top_blob.elemsize == 4u)
                    {
                        const float* a = (const float*)top_blob.data;
                        const float* b = (const float*)top_blob_ref.data;
                        for (size_t i = 0; i < total; i++)
                        {
                            float d = fabsf(a[i] - b[i]);
                            max_abs = std::max(max_abs, d);
                            mean_abs += d;
                        }
                        mean_abs /= (double)total;
                    }
                    else if (top_blob.elemsize == 1u)
                    {
                        const signed char* a = (const signed char*)top_blob.data;
                        const signed char* b = (const signed char*)top_blob_ref.data;
                        for (size_t i = 0; i < total; i++)
                        {
                            float d = (float)abs((int)a[i] - (int)b[i]);
                            max_abs = std::max(max_abs, d);
                            mean_abs += d;
                        }
                        mean_abs /= (double)total;
                    }
                    NCNN_LOGE("rvv int8 3x3s1 refcheck[%s] name=%s max_abs=%f mean_abs=%f elems=%zu outw=%d outh=%d inch=%d inpack=%d outch=%d",
                              tag, name.c_str(), max_abs, (float)mean_abs, total, outw, outh, channels, bottom_blob_bordered.elempack, num_output);
                }
                else
                {
                    NCNN_LOGE("rvv int8 3x3s1 refcheck[%s] skipped ret=%d ref_elemsize=%zu outw=%d outh=%d inch=%d inpack=%d outch=%d",
                              tag, ret, top_blob_ref.elemsize, outw, outh, channels, bottom_blob_bordered.elempack, num_output);
                }
            };

            if (g_rvv_int8_conv3x3s1_ref_check < 128)
            {
                char tag[16];
                snprintf(tag, sizeof(tag), "idx%d", g_rvv_int8_conv3x3s1_ref_check);
                refcheck_3x3s1(tag);
                g_rvv_int8_conv3x3s1_ref_check++;
            }
#endif
            if (coverage_enabled)
                g_riscv_int8_coverage_counters.conv_int8_pack1_3x3s1_fast_count.fetch_add(1, std::memory_order_relaxed);
            return 0;
        }

        if (!disable_rvv_int8_conv && !disable_rvv_int8_conv3x3 && !disable_rvv_int8_conv3x3s2 && !disable_rvv_int8_prep && int8_uniform && bottom_blob_fp32.dims == 3 && num_input == channels_unpacked && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            Mat bottom_blob_unbordered;
            if (prepare_int8_pack1(bottom_blob_unbordered) != 0)
                return -100;

            Mat bottom_blob_bordered;
            Option opt_pad = opt;
            opt_pad.use_packing_layout = false;
            make_padding(bottom_blob_unbordered, bottom_blob_bordered, opt_pad);
            if (bottom_blob_bordered.empty())
                return -100;

            int w = bottom_blob_bordered.w;
            int h = bottom_blob_bordered.h;
            int channels = bottom_blob_bordered.c;

            const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
            const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

            int outw = (w - kernel_extent_w) / stride_w + 1;
            int outh = (h - kernel_extent_h) / stride_h + 1;

            bool use_int8_requantize = int8_scale_term > 100;

#if NCNN_RISCV_INT8_CONV_DEBUG
            static int g_rvv_int8_conv3x3s2_seen = 0;
            if (!g_rvv_int8_conv3x3s2_seen)
            {
                NCNN_LOGE("riscv int8 3x3s2 conv rvv fast path");
                g_rvv_int8_conv3x3s2_seen = 1;
            }
            static int g_rvv_int8_conv3x3s2_info = 0;
            if (!g_rvv_int8_conv3x3s2_info)
            {
                NCNN_LOGE("rvv int8 3x3s2 info w=%d h=%d c=%d elempack=%d num_input=%d wscale0=%f inscale0=%f w_elemsize=%zu",
                          w, h, channels, bottom_blob_bordered.elempack, num_input,
                          weight_data_int8_scales.empty() ? 0.f : weight_data_int8_scales[0],
                          bottom_blob_int8_scales.empty() ? 0.f : bottom_blob_int8_scales[0],
                          weight_data.elemsize);
                g_rvv_int8_conv3x3s2_info = 1;
            }
            static int g_rvv_int8_conv3x3s2_check = 0;
            static int g_rvv_int8_conv3x3s2_tail_check = 0;
            static int g_rvv_int8_conv3x3s2_ref_check = 0;
#endif

            const int vlenb = csrr_vlenb();
            const int packn_fp32 = std::max(1, vlenb / 4);
            const bool packout_layout_candidate = opt.use_packing_layout && !use_int8_requantize && num_output % packn_fp32 == 0;
            const int requested_out_elempack = packout_layout_candidate ? packn_fp32 : 1;
            if (packing_audit_enabled)
            {
                const int bottom_scalar_elemsize = bottom_blob_bordered.elempack > 0 ? (int)(bottom_blob_bordered.elemsize / (size_t)bottom_blob_bordered.elempack) : 0;
                riscv_int8_packing_audit_record(kernel_w, kernel_h, stride_w, stride_h, dilation_w, dilation_h,
                                                bottom_blob_bordered.elempack, bottom_scalar_elemsize, requested_out_elempack, activation_type);
            }
            bool packout_candidate = packout_layout_candidate;
            if (packout_candidate && activation_type != 0)
            {
                packout_candidate = false;
                if (coverage_enabled)
                    g_riscv_int8_coverage_counters.conv_int8_packout_blocked_activation_count.fetch_add(1, std::memory_order_relaxed);
            }
            const int out_elempack = (packout_candidate && !disable_rvv_int8_packout) ? packn_fp32 : 1;
            size_t out_elemsize = (use_int8_requantize ? (size_t)1u : (size_t)4u) * out_elempack;

            top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

            if (coverage_enabled)
            {
                if (out_elempack > 1)
                    g_riscv_int8_coverage_counters.conv_int8_packout_used_count.fetch_add(1, std::memory_order_relaxed);
                else if (packout_candidate)
                    g_riscv_int8_coverage_counters.conv_int8_packout_forced_pack1_count.fetch_add(1, std::memory_order_relaxed);
            }

            #pragma omp parallel num_threads(opt.num_threads)
            {
                int* sums = riscv_get_sums_buffer((size_t)vlenb);

                #pragma omp for
                for (int p = 0; p < num_output; p++)
                {
                    const int out_channel_index = p / out_elempack;
                    const int out_lane = p % out_elempack;
                    Mat outc = top_blob.channel(out_channel_index);
                    const signed char* kptr = (const signed char*)weight_data + channels * p * 9;

                    float scale_in = 0.f;
                    if (weight_data_int8_scales[p] != 0)
                        scale_in = 1.f / (bottom_blob_int8_scales[0] * weight_data_int8_scales[p]);

                    float bias = bias_term ? bias_data[p] : 0.f;
                    float scale_out = use_int8_requantize ? top_blob_int8_scales[0] : 0.f;
                    const int force_fma = riscv_int8_conv_force_fma();
                    const int no_fma = riscv_int8_conv_no_fma();

                    for (int i = 0; i < outh; i++)
                    {
                        int in_y = i * stride_h;

                        if (use_int8_requantize)
                        {
                            signed char* outptr = outc.row<signed char>(i) + out_lane;

                            for (int j = 0; j < outw; )
                            {
                                const int remain = outw - j;
                                const int in_x = j * 2;
                                int max_pairs = 0;
                                if (!disable_rvv_int8_s2_vlseg2 && in_x + 2 < w)
                                    max_pairs = (w - (in_x + 2)) / 2;
                                int use_s2_vlseg2 = max_pairs > 0;
                                size_t vl = use_s2_vlseg2 ? __riscv_vsetvl_e8m1((size_t)std::min(remain, max_pairs)) : __riscv_vsetvl_e8m1((size_t)remain);
                                if (vl == 0)
                                {
                                    use_s2_vlseg2 = 0;
                                    vl = __riscv_vsetvl_e8m1((size_t)remain);
                                }
                                vint32m4_t _sum = __riscv_vmv_v_x_i32m4(0, vl);

                                for (int q = 0; q < channels; q++)
                                {
                                    const signed char* kptr_q = kptr + q * 9;

                                    const signed char* r0 = bottom_blob_bordered.channel(q).row<signed char>(in_y + 0) + j * 2;
                                    const signed char* r1 = bottom_blob_bordered.channel(q).row<signed char>(in_y + 1) + j * 2;
                                    const signed char* r2 = bottom_blob_bordered.channel(q).row<signed char>(in_y + 2) + j * 2;

                                    vint8m1_t _r00 = riscv_int8_load_stride2_i8m1(r0 + 0, vl, use_s2_vlseg2, coverage_enabled);
                                    vint8m1_t _r01 = riscv_int8_load_stride2_i8m1(r0 + 1, vl, use_s2_vlseg2, coverage_enabled);
                                    vint8m1_t _r02 = riscv_int8_load_stride2_i8m1(r0 + 2, vl, use_s2_vlseg2, coverage_enabled);
                                    vint8m1_t _r10 = riscv_int8_load_stride2_i8m1(r1 + 0, vl, use_s2_vlseg2, coverage_enabled);
                                    vint8m1_t _r11 = riscv_int8_load_stride2_i8m1(r1 + 1, vl, use_s2_vlseg2, coverage_enabled);
                                    vint8m1_t _r12 = riscv_int8_load_stride2_i8m1(r1 + 2, vl, use_s2_vlseg2, coverage_enabled);
                                    vint8m1_t _r20 = riscv_int8_load_stride2_i8m1(r2 + 0, vl, use_s2_vlseg2, coverage_enabled);
                                    vint8m1_t _r21 = riscv_int8_load_stride2_i8m1(r2 + 1, vl, use_s2_vlseg2, coverage_enabled);
                                    vint8m1_t _r22 = riscv_int8_load_stride2_i8m1(r2 + 2, vl, use_s2_vlseg2, coverage_enabled);

                                    vint16m2_t _r00_16 = __riscv_vsext_vf2_i16m2(_r00, vl);
                                    vint16m2_t _r01_16 = __riscv_vsext_vf2_i16m2(_r01, vl);
                                    vint16m2_t _r02_16 = __riscv_vsext_vf2_i16m2(_r02, vl);
                                    vint16m2_t _r10_16 = __riscv_vsext_vf2_i16m2(_r10, vl);
                                    vint16m2_t _r11_16 = __riscv_vsext_vf2_i16m2(_r11, vl);
                                    vint16m2_t _r12_16 = __riscv_vsext_vf2_i16m2(_r12, vl);
                                    vint16m2_t _r20_16 = __riscv_vsext_vf2_i16m2(_r20, vl);
                                    vint16m2_t _r21_16 = __riscv_vsext_vf2_i16m2(_r21, vl);
                                    vint16m2_t _r22_16 = __riscv_vsext_vf2_i16m2(_r22, vl);

                                    _sum = __riscv_vwmacc_vx_i32m4(_sum, (short)kptr_q[0], _r00_16, vl);
                                    _sum = __riscv_vwmacc_vx_i32m4(_sum, (short)kptr_q[1], _r01_16, vl);
                                    _sum = __riscv_vwmacc_vx_i32m4(_sum, (short)kptr_q[2], _r02_16, vl);
                                    _sum = __riscv_vwmacc_vx_i32m4(_sum, (short)kptr_q[3], _r10_16, vl);
                                    _sum = __riscv_vwmacc_vx_i32m4(_sum, (short)kptr_q[4], _r11_16, vl);
                                    _sum = __riscv_vwmacc_vx_i32m4(_sum, (short)kptr_q[5], _r12_16, vl);
                                    _sum = __riscv_vwmacc_vx_i32m4(_sum, (short)kptr_q[6], _r20_16, vl);
                                    _sum = __riscv_vwmacc_vx_i32m4(_sum, (short)kptr_q[7], _r21_16, vl);
                                    _sum = __riscv_vwmacc_vx_i32m4(_sum, (short)kptr_q[8], _r22_16, vl);
                                }

                                __riscv_vse32_v_i32m4(sums, _sum, vl);

                                for (size_t jj = 0; jj < vl; jj++)
                                {
                                    float sumfp32 = riscv_int8_conv_accum_fp32(sums[jj], scale_in, bias, bias_term, force_fma, no_fma);
                                    sumfp32 = activation_ss(sumfp32, activation_type, activation_params);
                                    outptr[(j + jj) * out_elempack] = float2int8_rvv(sumfp32 * scale_out);
                                }

                                j += vl;
                            }
                        }
                        else
                        {
                            float* outptr = outc.row<float>(i) + out_lane;

                            for (int j = 0; j < outw; )
                            {
                                const int remain = outw - j;
                                const int in_x = j * 2;
                                int max_pairs = 0;
                                if (!disable_rvv_int8_s2_vlseg2 && in_x + 2 < w)
                                    max_pairs = (w - (in_x + 2)) / 2;
                                int use_s2_vlseg2 = max_pairs > 0;
                                size_t vl = use_s2_vlseg2 ? __riscv_vsetvl_e8m1((size_t)std::min(remain, max_pairs)) : __riscv_vsetvl_e8m1((size_t)remain);
                                if (vl == 0)
                                {
                                    use_s2_vlseg2 = 0;
                                    vl = __riscv_vsetvl_e8m1((size_t)remain);
                                }
                                vint32m4_t _sum = __riscv_vmv_v_x_i32m4(0, vl);

                                for (int q = 0; q < channels; q++)
                                {
                                    const signed char* kptr_q = kptr + q * 9;

                                    const signed char* r0 = bottom_blob_bordered.channel(q).row<signed char>(in_y + 0) + j * 2;
                                    const signed char* r1 = bottom_blob_bordered.channel(q).row<signed char>(in_y + 1) + j * 2;
                                    const signed char* r2 = bottom_blob_bordered.channel(q).row<signed char>(in_y + 2) + j * 2;

                                    vint8m1_t _r00 = riscv_int8_load_stride2_i8m1(r0 + 0, vl, use_s2_vlseg2, coverage_enabled);
                                    vint8m1_t _r01 = riscv_int8_load_stride2_i8m1(r0 + 1, vl, use_s2_vlseg2, coverage_enabled);
                                    vint8m1_t _r02 = riscv_int8_load_stride2_i8m1(r0 + 2, vl, use_s2_vlseg2, coverage_enabled);
                                    vint8m1_t _r10 = riscv_int8_load_stride2_i8m1(r1 + 0, vl, use_s2_vlseg2, coverage_enabled);
                                    vint8m1_t _r11 = riscv_int8_load_stride2_i8m1(r1 + 1, vl, use_s2_vlseg2, coverage_enabled);
                                    vint8m1_t _r12 = riscv_int8_load_stride2_i8m1(r1 + 2, vl, use_s2_vlseg2, coverage_enabled);
                                    vint8m1_t _r20 = riscv_int8_load_stride2_i8m1(r2 + 0, vl, use_s2_vlseg2, coverage_enabled);
                                    vint8m1_t _r21 = riscv_int8_load_stride2_i8m1(r2 + 1, vl, use_s2_vlseg2, coverage_enabled);
                                    vint8m1_t _r22 = riscv_int8_load_stride2_i8m1(r2 + 2, vl, use_s2_vlseg2, coverage_enabled);

                                    vint16m2_t _r00_16 = __riscv_vsext_vf2_i16m2(_r00, vl);
                                    vint16m2_t _r01_16 = __riscv_vsext_vf2_i16m2(_r01, vl);
                                    vint16m2_t _r02_16 = __riscv_vsext_vf2_i16m2(_r02, vl);
                                    vint16m2_t _r10_16 = __riscv_vsext_vf2_i16m2(_r10, vl);
                                    vint16m2_t _r11_16 = __riscv_vsext_vf2_i16m2(_r11, vl);
                                    vint16m2_t _r12_16 = __riscv_vsext_vf2_i16m2(_r12, vl);
                                    vint16m2_t _r20_16 = __riscv_vsext_vf2_i16m2(_r20, vl);
                                    vint16m2_t _r21_16 = __riscv_vsext_vf2_i16m2(_r21, vl);
                                    vint16m2_t _r22_16 = __riscv_vsext_vf2_i16m2(_r22, vl);

                                    _sum = __riscv_vwmacc_vx_i32m4(_sum, (short)kptr_q[0], _r00_16, vl);
                                    _sum = __riscv_vwmacc_vx_i32m4(_sum, (short)kptr_q[1], _r01_16, vl);
                                    _sum = __riscv_vwmacc_vx_i32m4(_sum, (short)kptr_q[2], _r02_16, vl);
                                    _sum = __riscv_vwmacc_vx_i32m4(_sum, (short)kptr_q[3], _r10_16, vl);
                                    _sum = __riscv_vwmacc_vx_i32m4(_sum, (short)kptr_q[4], _r11_16, vl);
                                    _sum = __riscv_vwmacc_vx_i32m4(_sum, (short)kptr_q[5], _r12_16, vl);
                                    _sum = __riscv_vwmacc_vx_i32m4(_sum, (short)kptr_q[6], _r20_16, vl);
                                    _sum = __riscv_vwmacc_vx_i32m4(_sum, (short)kptr_q[7], _r21_16, vl);
                                    _sum = __riscv_vwmacc_vx_i32m4(_sum, (short)kptr_q[8], _r22_16, vl);
                                }

                                __riscv_vse32_v_i32m4(sums, _sum, vl);

#if NCNN_RISCV_INT8_CONV_DEBUG
                                if (!g_rvv_int8_conv3x3s2_check && p == 0 && i == 0 && j == 0)
                                {
                                    int sum_scalar = 0;
                                    for (int q = 0; q < channels; q++)
                                    {
                                        const signed char* kptr_q = kptr + q * 9;
                                        const signed char* r0 = bottom_blob_bordered.channel(q).row<signed char>(in_y + 0) + j * 2;
                                        const signed char* r1 = bottom_blob_bordered.channel(q).row<signed char>(in_y + 1) + j * 2;
                                        const signed char* r2 = bottom_blob_bordered.channel(q).row<signed char>(in_y + 2) + j * 2;
                                        sum_scalar += r0[0] * kptr_q[0];
                                        sum_scalar += r0[1] * kptr_q[1];
                                        sum_scalar += r0[2] * kptr_q[2];
                                        sum_scalar += r1[0] * kptr_q[3];
                                        sum_scalar += r1[1] * kptr_q[4];
                                        sum_scalar += r1[2] * kptr_q[5];
                                        sum_scalar += r2[0] * kptr_q[6];
                                        sum_scalar += r2[1] * kptr_q[7];
                                        sum_scalar += r2[2] * kptr_q[8];
                                    }
                                    NCNN_LOGE("rvv int8 3x3s2 check vec=%d scalar=%d scale_in=%f bias=%f",
                                              sums[0], sum_scalar, scale_in, bias);
                                    g_rvv_int8_conv3x3s2_check = 1;
                                }
                                if (!g_rvv_int8_conv3x3s2_tail_check && p == 0 && i == 0 && j + (int)vl == outw)
                                {
                                    int sum_scalar = 0;
                                    int j_tail = outw - 1;
                                    for (int q = 0; q < channels; q++)
                                    {
                                        const signed char* kptr_q = kptr + q * 9;
                                        const signed char* r0 = bottom_blob_bordered.channel(q).row<signed char>(in_y + 0) + j_tail * 2;
                                        const signed char* r1 = bottom_blob_bordered.channel(q).row<signed char>(in_y + 1) + j_tail * 2;
                                        const signed char* r2 = bottom_blob_bordered.channel(q).row<signed char>(in_y + 2) + j_tail * 2;
                                        sum_scalar += r0[0] * kptr_q[0];
                                        sum_scalar += r0[1] * kptr_q[1];
                                        sum_scalar += r0[2] * kptr_q[2];
                                        sum_scalar += r1[0] * kptr_q[3];
                                        sum_scalar += r1[1] * kptr_q[4];
                                        sum_scalar += r1[2] * kptr_q[5];
                                        sum_scalar += r2[0] * kptr_q[6];
                                        sum_scalar += r2[1] * kptr_q[7];
                                        sum_scalar += r2[2] * kptr_q[8];
                                    }
                                    NCNN_LOGE("rvv int8 3x3s2 tail vec=%d scalar=%d j=%d",
                                              sums[vl - 1], sum_scalar, j_tail);
                                    g_rvv_int8_conv3x3s2_tail_check = 1;
                                }
#endif
                                riscv_int8_store_fp32_from_sums(outptr + (size_t)j * out_elempack, sums, (int)vl, scale_in, bias, bias_term, force_fma, no_fma, activation_type, activation_params, out_elempack, disable_rvv_int8_post_vrvv, coverage_enabled);

                                j += vl;
                            }
                        }
                    }
                }
            }

#if NCNN_RISCV_INT8_CONV_STATS
            stats_dump("rvv3x3s2");
#endif
#if NCNN_RISCV_INT8_CONV_DEBUG
            auto refcheck_3x3s2 = [&](const char* tag)
            {
                Mat bottom_blob_ref = bottom_blob_fp32;
                if (bottom_blob_ref.elempack != 1)
                {
                    Option opt_pack1 = opt;
                    opt_pack1.blob_allocator = opt.workspace_allocator;
                    convert_packing(bottom_blob_fp32, bottom_blob_ref, 1, opt_pack1);
                }

                Option opt_ref = opt;
                opt_ref.use_packing_layout = false;
                Mat top_blob_ref;
                int ret = Convolution::forward_int8(bottom_blob_ref, top_blob_ref, opt_ref);
                if (ret == 0 && top_blob_ref.elemsize == top_blob.elemsize)
                {
                    const size_t total = top_blob.total();
                    float max_abs = 0.f;
                    double mean_abs = 0.0;
                    if (top_blob.elemsize == 4u)
                    {
                        const float* a = (const float*)top_blob.data;
                        const float* b = (const float*)top_blob_ref.data;
                        for (size_t i = 0; i < total; i++)
                        {
                            float d = fabsf(a[i] - b[i]);
                            max_abs = std::max(max_abs, d);
                            mean_abs += d;
                        }
                        mean_abs /= (double)total;
                    }
                    else if (top_blob.elemsize == 1u)
                    {
                        const signed char* a = (const signed char*)top_blob.data;
                        const signed char* b = (const signed char*)top_blob_ref.data;
                        for (size_t i = 0; i < total; i++)
                        {
                            float d = (float)abs((int)a[i] - (int)b[i]);
                            max_abs = std::max(max_abs, d);
                            mean_abs += d;
                        }
                        mean_abs /= (double)total;
                    }
                    NCNN_LOGE("rvv int8 3x3s2 refcheck[%s] name=%s max_abs=%f mean_abs=%f elems=%zu outw=%d outh=%d inch=%d inpack=%d outch=%d",
                              tag, name.c_str(), max_abs, (float)mean_abs, total, outw, outh, channels, bottom_blob_bordered.elempack, num_output);
                }
                else
                {
                    NCNN_LOGE("rvv int8 3x3s2 refcheck[%s] skipped ret=%d ref_elemsize=%zu outw=%d outh=%d inch=%d inpack=%d outch=%d",
                              tag, ret, top_blob_ref.elemsize, outw, outh, channels, bottom_blob_bordered.elempack, num_output);
                }
            };

            if (g_rvv_int8_conv3x3s2_ref_check < 128)
            {
                char tag[16];
                snprintf(tag, sizeof(tag), "idx%d", g_rvv_int8_conv3x3s2_ref_check);
                refcheck_3x3s2(tag);
                g_rvv_int8_conv3x3s2_ref_check++;
            }
#endif
            if (coverage_enabled)
                g_riscv_int8_coverage_counters.conv_int8_pack1_3x3s2_fast_count.fetch_add(1, std::memory_order_relaxed);
            return 0;
        }
#endif // __riscv_vector

        Mat bottom_blob_unpacked_fp32 = bottom_blob_fp32;
        if (bottom_blob_unpacked_fp32.elempack != 1)
        {
            Option opt_pack1 = opt;
            opt_pack1.blob_allocator = opt.workspace_allocator;

#if NCNN_RISCV_INT8_CONV_STATS
            double t0 = get_current_time();
#endif
            convert_packing(bottom_blob_fp32, bottom_blob_unpacked_fp32, 1, opt_pack1);
#if NCNN_RISCV_INT8_CONV_STATS
            stats_pack_ms += get_current_time() - t0;
            stats_pack_calls += 1;
#endif
            if (bottom_blob_unpacked_fp32.empty())
                return -100;
        }

        if (packing_audit_enabled)
        {
            int requested_out_elempack = 1;
#if __riscv_vector
            if (opt.use_packing_layout && int8_scale_term <= 100)
            {
                const int packn_fp32 = std::max(1, csrr_vlenb() / 4);
                if (num_output % packn_fp32 == 0)
                    requested_out_elempack = packn_fp32;
            }
#endif
            const int in_elempack = bottom_blob_fp32.elempack;
            const int bottom_scalar_elemsize = in_elempack > 0 ? (int)(bottom_blob_fp32.elemsize / (size_t)in_elempack) : 0;
            riscv_int8_packing_audit_record(kernel_w, kernel_h, stride_w, stride_h, dilation_w, dilation_h,
                                            in_elempack, bottom_scalar_elemsize, requested_out_elempack, activation_type);
        }

        Option opt_unpacked = opt;
        opt_unpacked.use_packing_layout = false;
        if (coverage_enabled)
        {
            g_riscv_int8_coverage_counters.conv_int8_fallback_count.fetch_add(1, std::memory_order_relaxed);
            if (!fallback_dims3)
                g_riscv_int8_coverage_counters.fallback_reason_dims_ne3.fetch_add(1, std::memory_order_relaxed);
            if (!fallback_dilation1)
                g_riscv_int8_coverage_counters.fallback_reason_dilation_ne1.fetch_add(1, std::memory_order_relaxed);
            if (!fallback_stride_1_or_2)
                g_riscv_int8_coverage_counters.fallback_reason_stride_not_1_or_2.fetch_add(1, std::memory_order_relaxed);
            if (!fallback_kernel_1_or_3)
                g_riscv_int8_coverage_counters.fallback_reason_kernel_not_1_or_3.fetch_add(1, std::memory_order_relaxed);
            if (!fallback_group_or_channel_match)
                g_riscv_int8_coverage_counters.fallback_reason_group_or_channel_mismatch.fetch_add(1, std::memory_order_relaxed);
            if (!int8_uniform)
                g_riscv_int8_coverage_counters.fallback_reason_int8_uniform_false.fetch_add(1, std::memory_order_relaxed);
            if (bottom_blob_fp32.elempack != 1)
                g_riscv_int8_coverage_counters.fallback_reason_bottom_elempack_ne1.fetch_add(1, std::memory_order_relaxed);
            if (disable_rvv_int8_conv)
                g_riscv_int8_coverage_counters.fallback_reason_disable_global.fetch_add(1, std::memory_order_relaxed);
            if (disable_rvv_int8_prep)
                g_riscv_int8_coverage_counters.fallback_reason_disable_prep.fetch_add(1, std::memory_order_relaxed);
            if (kernel_w == 1 && kernel_h == 1 && disable_rvv_int8_conv1x1)
                g_riscv_int8_coverage_counters.fallback_reason_disable_1x1.fetch_add(1, std::memory_order_relaxed);
            if (kernel_w == 3 && kernel_h == 3 && stride_w == 1 && stride_h == 1 && (disable_rvv_int8_conv3x3 || disable_rvv_int8_conv3x3s1))
                g_riscv_int8_coverage_counters.fallback_reason_disable_3x3s1.fetch_add(1, std::memory_order_relaxed);
            if (kernel_w == 3 && kernel_h == 3 && stride_w == 2 && stride_h == 2 && (disable_rvv_int8_conv3x3 || disable_rvv_int8_conv3x3s2))
                g_riscv_int8_coverage_counters.fallback_reason_disable_3x3s2.fetch_add(1, std::memory_order_relaxed);
        }
#if NCNN_RISCV_INT8_CONV_STATS
        stats_dump("fallback");
#endif
        return Convolution::forward_int8(bottom_blob_unpacked_fp32, top_blob, opt_unpacked);
    }
#endif

    // flattened blob, implement as InnerProduct
    if (bottom_blob.dims == 1 && kernel_w == 1 && kernel_h == 1)
    {
        Mat bottom_blob_3d;
        if (bottom_blob.elemsize % 16 == 0)
        {
            bottom_blob_3d = bottom_blob;
            bottom_blob_3d.dims = 3;
            bottom_blob_3d.w = 1;
            bottom_blob_3d.h = 1;
            bottom_blob_3d.c = bottom_blob.w;
            bottom_blob_3d.cstep = 1;
        }
        else
        {
            bottom_blob_3d = bottom_blob.reshape(1, 1, bottom_blob.w, opt.workspace_allocator);
        }

        Mat top_blob_3d;
        int ret = forward(bottom_blob_3d, top_blob_3d, opt);
        if (ret != 0)
            return ret;

        if (top_blob_3d.elemsize % 16 == 0)
        {
            top_blob = top_blob_3d;
            top_blob.dims = 1;
            top_blob.w = top_blob_3d.c;
            top_blob.h = 1;
            top_blob.c = 1;
            top_blob.cstep = top_blob_3d.c;
        }
        else
        {
            top_blob = top_blob_3d.reshape(top_blob_3d.c, opt.blob_allocator);
        }

        return 0;
    }

#if NCNN_ZFH
    int elembits = bottom_blob.elembits();

    if (opt.use_fp16_storage && elembits == 16)
    {
        if (opt.use_fp16_arithmetic)
            return forward_fp16sa(bottom_blob, top_blob, opt);
        else
            return forward_fp16s(bottom_blob, top_blob, opt);
    }
#endif

#if __riscv_vector
    const int packn = csrr_vlenb() / 4;
#endif

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    //     NCNN_LOGE("Convolution input %d x %d  pad = %d %d  ksize=%d %d  stride=%d %d", w, h, pad_w, pad_h, kernel_w, kernel_h, stride_w, stride_h);

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    Mat bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;

    int outw = (w - kernel_extent_w) / stride_w + 1;
    int outh = (h - kernel_extent_h) / stride_h + 1;
    int out_elempack = 1;
#if __riscv_vector
    if (opt.use_packing_layout)
    {
        out_elempack = num_output % packn == 0 ? packn : 1;
    }
#endif
    size_t out_elemsize = elemsize / elempack * out_elempack;

    top_blob.create(outw, outh, num_output / out_elempack, out_elemsize, out_elempack, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    const int num_input = channels * elempack;

#if __riscv_vector
    if (elempack == packn && out_elempack == packn)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_sgemm_packn_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv1x1s2_sgemm_packn_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (opt.use_winograd_convolution && (opt.use_winograd23_convolution || opt.use_winograd43_convolution || opt.use_winograd63_convolution) && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            if ((opt.use_winograd63_convolution && num_input >= packn * 2 && num_output >= packn * 2 && num_input <= packn * 16 && num_output <= packn * 16) || (!opt.use_winograd43_convolution && !opt.use_winograd23_convolution))
                conv3x3s1_winograd63_packn_rvv(bottom_blob_bordered, top_blob, weight_winograd63_data, bias_data, opt);
            else if ((opt.use_winograd43_convolution && num_input >= packn * 2 && num_output >= packn * 2) || (!opt.use_winograd63_convolution && !opt.use_winograd23_convolution))
                conv3x3s1_winograd43_packn_rvv(bottom_blob_bordered, top_blob, weight_winograd43_data, bias_data, opt);
            else // if (opt.use_winograd23_convolution)
                conv3x3s1_winograd23_packn_rvv(bottom_blob_bordered, top_blob, weight_winograd23_data, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (opt.use_sgemm_convolution)
        {
            convolution_im2col_sgemm_packn_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            convolution_packn_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
        }
    }

    if (elempack == 1 && out_elempack == packn)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_sgemm_pack1ton_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv3x3s1_pack1ton_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv3x3s2_pack1ton_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 7 && kernel_h == 7 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv7x7s2_pack1ton_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (opt.use_sgemm_convolution)
        {
            convolution_im2col_sgemm_pack1ton_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            convolution_pack1ton_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
        }
    }

    if (elempack == packn && out_elempack == 1)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_sgemm_packnto1_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 2 && stride_h == 2)
        {
            conv1x1s2_sgemm_packnto1_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (opt.use_sgemm_convolution)
        {
            convolution_im2col_sgemm_packnto1_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            convolution_packnto1_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, activation_type, activation_params, opt);
        }
    }
#endif // __riscv_vector

    if (elempack == 1 && out_elempack == 1)
    {
        if (kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            conv1x1s1_sgemm_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (opt.use_winograd_convolution && (opt.use_winograd43_convolution || opt.use_winograd23_convolution) && kernel_w == 3 && kernel_h == 3 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
        {
            if ((opt.use_winograd43_convolution && num_input >= 16 && num_output >= 16) || !opt.use_winograd23_convolution)
            {
                conv3x3s1_winograd43_rvv(bottom_blob_bordered, top_blob, weight_winograd43_data, bias_data, opt);
            }
            else if (opt.use_winograd23_convolution)
            {
                conv3x3s1_winograd23_rvv(bottom_blob_bordered, top_blob, weight_winograd23_data, bias_data, opt);
            }

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else if (opt.use_sgemm_convolution)
        {
            convolution_im2col_sgemm_rvv(bottom_blob_bordered, top_blob, weight_data_tm, bias_data, kernel_w, kernel_h, dilation_w, dilation_h, stride_w, stride_h, opt);

            if (activation)
            {
                activation->forward_inplace(top_blob, opt);
            }
        }
        else
        {
            const int maxk = kernel_w * kernel_h;

            // kernel offsets
            std::vector<int> _space_ofs(maxk);
            int* space_ofs = &_space_ofs[0];
            {
                int p1 = 0;
                int p2 = 0;
                int gap = w * dilation_h - kernel_w * dilation_w;
                for (int i = 0; i < kernel_h; i++)
                {
                    for (int j = 0; j < kernel_w; j++)
                    {
                        space_ofs[p1] = p2;
                        p1++;
                        p2 += dilation_w;
                    }
                    p2 += gap;
                }
            }

            // num_output
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int p = 0; p < num_output; p++)
            {
                float* outptr = top_blob.channel(p);

                for (int i = 0; i < outh; i++)
                {
                    for (int j = 0; j < outw; j++)
                    {
                        float sum = 0.f;

                        if (bias_term)
                        {
                            sum = bias_data[p];
                        }

                        const float* kptr = (const float*)weight_data_tm + maxk * channels * p;

                        // channels
                        for (int q = 0; q < channels; q++)
                        {
                            const Mat m = bottom_blob_bordered.channel(q);
                            const float* sptr = m.row(i * stride_h) + j * stride_w;

                            for (int k = 0; k < maxk; k++)
                            {
                                float val = sptr[space_ofs[k]];
                                float wt = kptr[k];
                                sum += val * wt;
                            }

                            kptr += maxk;
                        }

                        sum = activation_ss(sum, activation_type, activation_params);

                        outptr[j] = sum;
                    }

                    outptr += outw;
                }
            }
        }
    }

    return 0;
}

int Convolution_riscv::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    const Mat& _weight_data = bottom_blobs[1];
    Mat& top_blob = top_blobs[0];

    const int _kernel_w = _weight_data.w;
    const int _kernel_h = _weight_data.h;
    const int _num_output = _weight_data.c * _weight_data.elempack;

    Mat weight_data_flattened;
    flatten(_weight_data, weight_data_flattened, opt);
    if (weight_data_flattened.empty())
        return -100;

#if NCNN_ZFH
    if (opt.use_fp16_storage && support_fp16_storage && weight_data_flattened.elembits() == 16)
    {
        Mat weight_data_flattened_fp32;
        cast_float16_to_float32(weight_data_flattened, weight_data_flattened_fp32, opt);
        weight_data_flattened = weight_data_flattened_fp32;
    }
#endif // NCNN_ZFH

    // weight_data_flattened as pack1
    weight_data_flattened.w *= weight_data_flattened.elempack;
    weight_data_flattened.elemsize /= weight_data_flattened.elempack;
    weight_data_flattened.elempack = 1;

    Mat bias_data_flattened;
    if (bias_term)
    {
        const Mat& _bias_data = bottom_blobs[2];
        flatten(_bias_data, bias_data_flattened, opt);
        if (bias_data_flattened.empty())
            return -100;

#if NCNN_ZFH
        if (opt.use_fp16_storage && support_fp16_storage && bias_data_flattened.elembits() == 16)
        {
            Mat bias_data_flattened_fp32;
            cast_float16_to_float32(bias_data_flattened, bias_data_flattened_fp32, opt);
            bias_data_flattened = bias_data_flattened_fp32;
        }
#endif // NCNN_ZFH

        // bias_data_flattened as pack1
        bias_data_flattened.w *= bias_data_flattened.elempack;
        bias_data_flattened.elemsize /= bias_data_flattened.elempack;
        bias_data_flattened.elempack = 1;
    }

    ncnn::Layer* op = ncnn::create_layer_cpu(ncnn::LayerType::Convolution);

    ncnn::ParamDict pd;
    pd.set(0, _num_output);
    pd.set(1, _kernel_w);
    pd.set(11, _kernel_h);
    pd.set(2, dilation_w);
    pd.set(12, dilation_h);
    pd.set(3, stride_w);
    pd.set(13, stride_h);
    pd.set(4, pad_left);
    pd.set(15, pad_right);
    pd.set(14, pad_top);
    pd.set(16, pad_bottom);
    pd.set(18, pad_value);
    pd.set(5, bias_term);
    pd.set(6, weight_data_flattened.w);
    pd.set(8, int8_scale_term);
    pd.set(9, activation_type);
    pd.set(10, activation_params);

    op->load_param(pd);

    ncnn::Mat weights[2];
    weights[0] = weight_data_flattened;
    weights[1] = bias_data_flattened;

    op->load_model(ncnn::ModelBinFromMatArray(weights));

    op->create_pipeline(opt);

    op->forward(bottom_blob, top_blob, opt);

    op->destroy_pipeline(opt);

    delete op;

    return 0;
}

} // namespace ncnn
