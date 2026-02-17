// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "convolution_riscv.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

#include "benchmark.h"
#include "cpu.h"
#include "layer_type.h"

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector
#include "riscv_activation.h"
#include "riscv_usability.h"

namespace ncnn {

#if NCNN_INT8
struct riscv_int8_coverage_counters_t
{
    std::atomic<unsigned long long> conv_int8_pack1_1x1_fast_count{0};
    std::atomic<unsigned long long> conv_int8_pack1_3x3s1_fast_count{0};
    std::atomic<unsigned long long> conv_int8_pack1_3x3s2_fast_count{0};
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

    NCNN_LOGE("riscv_int8_coverage conv_int8_pack1_1x1_fast_count=%llu conv_int8_pack1_3x3s1_fast_count=%llu conv_int8_pack1_3x3s2_fast_count=%llu post_vrvv_used_count=%llu post_scalar_used_count=%llu conv_int8_fallback_count=%llu "
              "fallback_reason_dims_ne3=%llu fallback_reason_dilation_ne1=%llu fallback_reason_stride_not_1_or_2=%llu fallback_reason_kernel_not_1_or_3=%llu "
              "fallback_reason_group_or_channel_mismatch=%llu fallback_reason_int8_scale_term_0=%llu fallback_reason_int8_uniform_false=%llu fallback_reason_bottom_elempack_ne1=%llu "
              "fallback_reason_disable_global=%llu fallback_reason_disable_prep=%llu fallback_reason_disable_1x1=%llu fallback_reason_disable_3x3s1=%llu fallback_reason_disable_3x3s2=%llu",
              load(g_riscv_int8_coverage_counters.conv_int8_pack1_1x1_fast_count),
              load(g_riscv_int8_coverage_counters.conv_int8_pack1_3x3s1_fast_count),
              load(g_riscv_int8_coverage_counters.conv_int8_pack1_3x3s2_fast_count),
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

static inline void riscv_int8_store_fp32_from_sums(float* outptr, const int* sums, int count, float scale_in, float bias, int bias_term, int force_fma, int no_fma, int activation_type, const Mat& activation_params, int disable_post_vrvv, int coverage_enabled)
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

            __riscv_vse32_v_f32m4(outptr + j, _sum, vl);
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
        outptr[j] = sumfp32;
    }

    if (coverage_enabled)
        g_riscv_int8_coverage_counters.post_scalar_used_count.fetch_add(1, std::memory_order_relaxed);
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

#if NCNN_INT8
    if (opt.use_int8_inference && weight_data.elemsize == (size_t)1u)
    {
        // TODO implement int8
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

    const int maxk = kernel_w * kernel_h;
    const int num_input = weight_data_size / maxk / num_output;

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

    return 0;
}

int Convolution_riscv::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
#if NCNN_INT8
    const bool coverage_enabled = riscv_int8_coverage_enabled() != 0;
    if (coverage_enabled)
        riscv_int8_coverage_register_atexit_once();

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

        const int disable_rvv_int8_conv = riscv_int8_conv_disabled();
        const int disable_rvv_int8_conv1x1 = riscv_int8_conv1x1_disabled();
        const int disable_rvv_int8_conv3x3 = riscv_int8_conv3x3_disabled();
        const int disable_rvv_int8_conv3x3s1 = riscv_int8_conv3x3s1_disabled();
        const int disable_rvv_int8_conv3x3s2 = riscv_int8_conv3x3s2_disabled();
        const int disable_rvv_int8_prep = riscv_int8_conv_prep_disabled();
        const int disable_rvv_int8_post_vrvv = riscv_int8_conv_post_vrvv_disabled();
        const int num_input = weight_data_size / num_output / (kernel_w * kernel_h);
        const int channels_unpacked = bottom_blob_fp32.c * bottom_blob_fp32.elempack;
        const bool fallback_dims3 = bottom_blob_fp32.dims == 3;
        const bool fallback_dilation1 = dilation_w == 1 && dilation_h == 1;
        const bool fallback_stride_1_or_2 = (stride_w == 1 && stride_h == 1) || (stride_w == 2 && stride_h == 2);
        const bool fallback_kernel_1_or_3 = (kernel_w == 1 && kernel_h == 1) || (kernel_w == 3 && kernel_h == 3);
        const bool fallback_group_or_channel_match = num_input == channels_unpacked;

        if (!disable_rvv_int8_conv && !disable_rvv_int8_conv1x1 && !disable_rvv_int8_prep && int8_uniform && bottom_blob_fp32.dims == 3 && num_input == channels_unpacked && kernel_w == 1 && kernel_h == 1 && dilation_w == 1 && dilation_h == 1 && stride_w == 1 && stride_h == 1)
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
            size_t out_elemsize = use_int8_requantize ? 1u : 4u;

            top_blob.create(outw, outh, num_output, out_elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

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

            #pragma omp parallel num_threads(opt.num_threads)
            {
                int* sums = riscv_get_sums_buffer((size_t)vlenb);

                #pragma omp for
                for (int p = 0; p < num_output; p++)
                {
                    Mat outc = top_blob.channel(p);
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
                            signed char* outptr = outc.row<signed char>(i);

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
                                    outptr[j + jj] = float2int8_rvv(sumfp32 * scale_out);
                                }

                                j += vl;
                            }
                        }
                        else
                        {
                            float* outptr = outc.row<float>(i);

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
                                riscv_int8_store_fp32_from_sums(outptr + j, sums, (int)vl, scale_in, bias, bias_term, force_fma, no_fma, activation_type, activation_params, disable_rvv_int8_post_vrvv, coverage_enabled);

                                j += vl;
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
            size_t out_elemsize = use_int8_requantize ? 1u : 4u;

            top_blob.create(outw, outh, num_output, out_elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

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

            #pragma omp parallel num_threads(opt.num_threads)
            {
                int* sums = riscv_get_sums_buffer((size_t)vlenb);

                #pragma omp for
                for (int p = 0; p < num_output; p++)
                {
                    Mat outc = top_blob.channel(p);
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
                            signed char* outptr = outc.row<signed char>(i);

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
                                    outptr[j + jj] = float2int8_rvv(sumfp32 * scale_out);
                                }

                                j += vl;
                            }
                        }
                        else
                        {
                            float* outptr = outc.row<float>(i);

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

#if NCNN_RISCV_INT8_CONV_DEBUG
                                if (!g_rvv_int8_conv3x3s1_check && p == 0 && i == 0 && j == 0)
                                {
                                    int sum_scalar = 0;
                                    for (int q = 0; q < channels; q++)
                                    {
                                        const signed char* kptr_q = kptr + q * 9;
                                        const signed char* r0 = bottom_blob_bordered.channel(q).row<signed char>(in_y + 0) + j;
                                        const signed char* r1 = bottom_blob_bordered.channel(q).row<signed char>(in_y + 1) + j;
                                        const signed char* r2 = bottom_blob_bordered.channel(q).row<signed char>(in_y + 2) + j;
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
                                    NCNN_LOGE("rvv int8 3x3s1 check vec=%d scalar=%d scale_in=%f bias=%f",
                                              sums[0], sum_scalar, scale_in, bias);
                                    g_rvv_int8_conv3x3s1_check = 1;
                                }
                                if (!g_rvv_int8_conv3x3s1_tail_check && p == 0 && i == 0 && j + (int)vl == outw)
                                {
                                    int sum_scalar = 0;
                                    int j_tail = outw - 1;
                                    for (int q = 0; q < channels; q++)
                                    {
                                        const signed char* kptr_q = kptr + q * 9;
                                        const signed char* r0 = bottom_blob_bordered.channel(q).row<signed char>(in_y + 0) + j_tail;
                                        const signed char* r1 = bottom_blob_bordered.channel(q).row<signed char>(in_y + 1) + j_tail;
                                        const signed char* r2 = bottom_blob_bordered.channel(q).row<signed char>(in_y + 2) + j_tail;
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
                                    NCNN_LOGE("rvv int8 3x3s1 tail vec=%d scalar=%d j=%d",
                                              sums[vl - 1], sum_scalar, j_tail);
                                    g_rvv_int8_conv3x3s1_tail_check = 1;
                                }
#endif
                                riscv_int8_store_fp32_from_sums(outptr + j, sums, (int)vl, scale_in, bias, bias_term, force_fma, no_fma, activation_type, activation_params, disable_rvv_int8_post_vrvv, coverage_enabled);

                                j += vl;
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
            size_t out_elemsize = use_int8_requantize ? 1u : 4u;

            top_blob.create(outw, outh, num_output, out_elemsize, opt.blob_allocator);
            if (top_blob.empty())
                return -100;

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

            #pragma omp parallel num_threads(opt.num_threads)
            {
                int* sums = riscv_get_sums_buffer((size_t)vlenb);

                #pragma omp for
                for (int p = 0; p < num_output; p++)
                {
                    Mat outc = top_blob.channel(p);
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
                            signed char* outptr = outc.row<signed char>(i);

                            for (int j = 0; j < outw; )
                            {
                                size_t vl = __riscv_vsetvl_e8m1(outw - j);
                                vint32m4_t _sum = __riscv_vmv_v_x_i32m4(0, vl);

                                for (int q = 0; q < channels; q++)
                                {
                                    const signed char* kptr_q = kptr + q * 9;

                                    const signed char* r0 = bottom_blob_bordered.channel(q).row<signed char>(in_y + 0) + j * 2;
                                    const signed char* r1 = bottom_blob_bordered.channel(q).row<signed char>(in_y + 1) + j * 2;
                                    const signed char* r2 = bottom_blob_bordered.channel(q).row<signed char>(in_y + 2) + j * 2;

                                    vint8m1_t _r00 = __riscv_vlse8_v_i8m1(r0 + 0, 2, vl);
                                    vint8m1_t _r01 = __riscv_vlse8_v_i8m1(r0 + 1, 2, vl);
                                    vint8m1_t _r02 = __riscv_vlse8_v_i8m1(r0 + 2, 2, vl);
                                    vint8m1_t _r10 = __riscv_vlse8_v_i8m1(r1 + 0, 2, vl);
                                    vint8m1_t _r11 = __riscv_vlse8_v_i8m1(r1 + 1, 2, vl);
                                    vint8m1_t _r12 = __riscv_vlse8_v_i8m1(r1 + 2, 2, vl);
                                    vint8m1_t _r20 = __riscv_vlse8_v_i8m1(r2 + 0, 2, vl);
                                    vint8m1_t _r21 = __riscv_vlse8_v_i8m1(r2 + 1, 2, vl);
                                    vint8m1_t _r22 = __riscv_vlse8_v_i8m1(r2 + 2, 2, vl);

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
                                    outptr[j + jj] = float2int8_rvv(sumfp32 * scale_out);
                                }

                                j += vl;
                            }
                        }
                        else
                        {
                            float* outptr = outc.row<float>(i);

                            for (int j = 0; j < outw; )
                            {
                                size_t vl = __riscv_vsetvl_e8m1(outw - j);
                                vint32m4_t _sum = __riscv_vmv_v_x_i32m4(0, vl);

                                for (int q = 0; q < channels; q++)
                                {
                                    const signed char* kptr_q = kptr + q * 9;

                                    const signed char* r0 = bottom_blob_bordered.channel(q).row<signed char>(in_y + 0) + j * 2;
                                    const signed char* r1 = bottom_blob_bordered.channel(q).row<signed char>(in_y + 1) + j * 2;
                                    const signed char* r2 = bottom_blob_bordered.channel(q).row<signed char>(in_y + 2) + j * 2;

                                    vint8m1_t _r00 = __riscv_vlse8_v_i8m1(r0 + 0, 2, vl);
                                    vint8m1_t _r01 = __riscv_vlse8_v_i8m1(r0 + 1, 2, vl);
                                    vint8m1_t _r02 = __riscv_vlse8_v_i8m1(r0 + 2, 2, vl);
                                    vint8m1_t _r10 = __riscv_vlse8_v_i8m1(r1 + 0, 2, vl);
                                    vint8m1_t _r11 = __riscv_vlse8_v_i8m1(r1 + 1, 2, vl);
                                    vint8m1_t _r12 = __riscv_vlse8_v_i8m1(r1 + 2, 2, vl);
                                    vint8m1_t _r20 = __riscv_vlse8_v_i8m1(r2 + 0, 2, vl);
                                    vint8m1_t _r21 = __riscv_vlse8_v_i8m1(r2 + 1, 2, vl);
                                    vint8m1_t _r22 = __riscv_vlse8_v_i8m1(r2 + 2, 2, vl);

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
                                riscv_int8_store_fp32_from_sums(outptr + j, sums, (int)vl, scale_in, bias, bias_term, force_fma, no_fma, activation_type, activation_params, disable_rvv_int8_post_vrvv, coverage_enabled);

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
