// Copyright 2026
// SPDX-License-Identifier: BSD-3-Clause

#include "convolution_1x1_int8_xsmtvdot.h"

#include "cpu.h"
#include "k1x_runtime_feature.h"

#include <cstddef>
#include <cstdlib>
#include <algorithm>
#include <vector>

#if __riscv_vector
#include <riscv_vector.h>
#endif // __riscv_vector

namespace ncnn {

static inline signed char xsmtvdot_float2int8(float v)
{
    int int32 = (int)(v >= 0.f ? v + 0.5f : v - 0.5f);
    if (int32 > 127) return 127;
    if (int32 < -127) return -127;
    return (signed char)int32;
}

static inline void xsmtvdot_store_int8_tile(const int* acc, const float scale_in[4], const float bias[4], float scale_out, signed char* const outptrs[4], int s)
{
#if __riscv_vector
    const ptrdiff_t acc_stride = (ptrdiff_t)(4 * sizeof(int));
    for (int n = 0; n < 4; n++)
    {
        signed char* outptr = outptrs[n] + s;
        int m = 0;
        while (m < 4)
        {
            const size_t vl = __riscv_vsetvl_e32m1((size_t)(4 - m));
            vint32m1_t _acc = __riscv_vlse32_v_i32m1(acc + m * 4 + n, acc_stride, vl);
            vfloat32m1_t _sum = __riscv_vfcvt_f_x_v_f32m1(_acc, vl);
            _sum = __riscv_vfmul_vf_f32m1(_sum, scale_in[n], vl);
            _sum = __riscv_vfadd_vf_f32m1(_sum, bias[n], vl);
            _sum = __riscv_vfmul_vf_f32m1(_sum, scale_out, vl);

            // Match xsmtvdot_float2int8(): sign-dependent half-away offset,
            // truncate toward zero, clamp to [-127, 127], then narrow/store.
            const vbool32_t _negative = __riscv_vmflt_vf_f32m1_b32(_sum, 0.f, vl);
            vfloat32m1_t _round = __riscv_vfmv_v_f_f32m1(0.5f, vl);
            _round = __riscv_vfmerge_vfm_f32m1(_round, -0.5f, _negative, vl);
            _sum = __riscv_vfadd_vv_f32m1(_sum, _round, vl);
            vint32m1_t _out32 = __riscv_vfcvt_rtz_x_f_v_i32m1(_sum, vl);
            _out32 = __riscv_vmax_vx_i32m1(_out32, -127, vl);
            _out32 = __riscv_vmin_vx_i32m1(_out32, 127, vl);
            vint16mf2_t _out16 = __riscv_vncvt_x_x_w_i16mf2(_out32, vl);
            vint8mf4_t _out8 = __riscv_vncvt_x_x_w_i8mf4(_out16, vl);
            __riscv_vse8_v_i8mf4(outptr + m, _out8, vl);

            m += (int)vl;
        }
    }
#else  // __riscv_vector
    for (int n = 0; n < 4; n++)
    {
        signed char* outptr = outptrs[n] + s;
        for (int m = 0; m < 4; m++)
        {
            const float sumfp32 = (float)acc[m * 4 + n] * scale_in[n] + bias[n];
            outptr[m] = xsmtvdot_float2int8(sumfp32 * scale_out);
        }
    }
#endif // __riscv_vector
}

static inline void xsmtvdot_store_fp32_tile(const int* acc, const float scale_in[4], const float bias[4], float* const outptrs[4], int s)
{
#if __riscv_vector
    const ptrdiff_t acc_stride = (ptrdiff_t)(4 * sizeof(int));
    for (int n = 0; n < 4; n++)
    {
        float* outptr = outptrs[n] + s;
        int m = 0;
        while (m < 4)
        {
            const size_t vl = __riscv_vsetvl_e32m1((size_t)(4 - m));
            vint32m1_t _acc = __riscv_vlse32_v_i32m1(acc + m * 4 + n, acc_stride, vl);
            vfloat32m1_t _sum = __riscv_vfcvt_f_x_v_f32m1(_acc, vl);
            _sum = __riscv_vfmul_vf_f32m1(_sum, scale_in[n], vl);
            _sum = __riscv_vfadd_vf_f32m1(_sum, bias[n], vl);
            __riscv_vse32_v_f32m1(outptr + m, _sum, vl);
            m += (int)vl;
        }
    }
#else  // __riscv_vector
    for (int n = 0; n < 4; n++)
    {
        float* outptr = outptrs[n] + s;
        for (int m = 0; m < 4; m++)
            outptr[m] = (float)acc[m * 4 + n] * scale_in[n] + bias[n];
    }
#endif // __riscv_vector
}

int convolution_1x1_int8_xsmtvdot_create_weight_tm(const Mat& weight_data, Mat& weight_data_tm, int num_input, int num_output)
{
    weight_data_tm.release();

    if (weight_data.empty() || weight_data.elemsize != (size_t)1u)
        return 0;
    if (num_input <= 0 || num_output <= 0)
        return 0;
    if (num_input % 8 != 0 || num_output % 4 != 0)
        return 0;

    const int kblocks = num_input / 8;
    const int out_groups = num_output / 4;

    weight_data_tm.create(32 * kblocks, out_groups, (size_t)1u);
    if (weight_data_tm.empty())
        return -100;

    const signed char* weight_ptr = (const signed char*)weight_data;
    for (int pg = 0; pg < out_groups; pg++)
    {
        signed char* gptr = weight_data_tm.row<signed char>(pg);
        const int p0 = pg * 4;

        for (int kb = 0; kb < kblocks; kb++)
        {
            signed char* bpack = gptr + (size_t)kb * 32;
            const int q0 = kb * 8;
            for (int n = 0; n < 4; n++)
            {
                const signed char* kptr = weight_ptr + (size_t)(p0 + n) * num_input + q0;
                for (int k = 0; k < 8; k++)
                    bpack[n * 8 + k] = kptr[k];
            }
        }
    }

    return 0;
}

int convolution_1x1_int8_xsmtvdot_pipeline_enabled(const Option& opt, int activation_type)
{
    return convolution_1x1_int8_xsmtvdot_pipeline_mode(opt, activation_type) != CONVOLUTION_1X1_INT8_XSMTVDOT_PATH_NONE;
}

int convolution_1x1_int8_xsmtvdot_pipeline_mode(const Option& opt, int activation_type)
{
    const char* apanel_env = getenv("NCNN_RISCV_INT8_XSMTVDOT_4X4K_APANEL_ENABLE");
    if (apanel_env && apanel_env[0] != '\0' && apanel_env[0] != '0')
    {
        if (opt.num_threads == 1 && k1x_xsmtvdot_policy_allows(opt, activation_type, opt.num_threads, 1, 1))
            return CONVOLUTION_1X1_INT8_XSMTVDOT_PATH_4X4K_APANEL_EXPERIMENTAL;

        if (opt.num_threads > 1 && k1x_xsmtvdot_policy_allows_cluster0_workers(opt, activation_type, opt.num_threads, 1, 1))
            return CONVOLUTION_1X1_INT8_XSMTVDOT_PATH_4X4K_APANEL_CLUSTER0_MT_EXPERIMENTAL;

        return CONVOLUTION_1X1_INT8_XSMTVDOT_PATH_NONE;
    }

    if (k1x_xsmtvdot_policy_allows(opt, activation_type, opt.num_threads, 1, 1))
    {
        return CONVOLUTION_1X1_INT8_XSMTVDOT_PATH_LEGACY;
    }

    return CONVOLUTION_1X1_INT8_XSMTVDOT_PATH_NONE;
}

int convolution_1x1_int8_xsmtvdot_h4b_effective_workers(const Option& opt, int size)
{
    if (opt.num_threads <= 1 || size <= 0)
        return 1;

    const int tile_blocks = size / 4;
    if (tile_blocks <= 1)
        return 1;

    const int worker_capacity = k1x_xsmtvdot_cluster0_worker_capacity();
    if (worker_capacity < 2)
        return 1;

    return std::max(1, std::min(std::min(opt.num_threads, worker_capacity), tile_blocks));
}

int convolution_1x1_int8_xsmtvdot_legacy_safety_gated(int w, int h, int channels, int num_output)
{
    const int size = w * h;

    // H2 Lane A intentionally gates only the current legacy path for the
    // Conv42-like small-spatial/high-channel regime identified by H0/H1.
    return size > 0 && size <= 1600 && channels >= 384 && num_output >= 128;
}

int convolution_1x1_int8_xsmtvdot_forward(const Mat& bottom_blob_int8,
                                          Mat& top_blob,
                                          const Mat& weight_data_tm,
                                          const Mat& bias_data,
                                          const Mat& bottom_blob_int8_scales,
                                          const Mat& weight_data_int8_scales,
                                          const Mat& top_blob_int8_scales,
                                          int bias_term,
                                          int int8_scale_term,
                                          int activation_type,
                                          int num_output,
                                          const Option& opt)
{
    if (opt.num_threads != 1)
        return 1;

    if (activation_type != 0)
        return 1;

    if (bottom_blob_int8.dims != 3 || bottom_blob_int8.elempack != 1 || bottom_blob_int8.elemsize != (size_t)1u)
        return 1;

    const int w = bottom_blob_int8.w;
    const int h = bottom_blob_int8.h;
    const int channels = bottom_blob_int8.c;
    const int size = w * h;

    if (w <= 0 || h <= 0 || channels <= 0 || num_output <= 0)
        return 1;
    if (channels % 8 != 0 || num_output % 4 != 0 || size % 4 != 0)
        return 1;
    if (weight_data_tm.empty() || weight_data_tm.w != 32 * (channels / 8) || weight_data_tm.h != num_output / 4)
        return 1;
    if (bottom_blob_int8_scales.w != 1 || weight_data_int8_scales.w < num_output)
        return 1;

    const bool use_int8_requantize = int8_scale_term > 100;
    if (use_int8_requantize && top_blob_int8_scales.w < 1)
        return 1;

    top_blob.create(w, h, num_output, use_int8_requantize ? (size_t)1u : (size_t)4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    std::vector<const signed char*> inptrs(channels);
    for (int q = 0; q < channels; q++)
        inptrs[q] = bottom_blob_int8.channel(q).row<const signed char>(0);

    const int kblocks = channels / 8;
    const float bottom_scale = bottom_blob_int8_scales[0];
    const float scale_out = use_int8_requantize ? top_blob_int8_scales[0] : 0.f;

    for (int pg = 0; pg < num_output / 4; pg++)
    {
        const signed char* bpack_group = weight_data_tm.row<const signed char>(pg);
        const int p0 = pg * 4;

        float scale_in[4];
        float bias[4];
        for (int n = 0; n < 4; n++)
        {
            const int p = p0 + n;
            scale_in[n] = weight_data_int8_scales[p] != 0 ? 1.f / (bottom_scale * weight_data_int8_scales[p]) : 0.f;
            bias[n] = bias_term ? bias_data[p] : 0.f;
        }

        signed char* outptr_int8[4] = {0, 0, 0, 0};
        float* outptr_fp32[4] = {0, 0, 0, 0};
        if (use_int8_requantize)
        {
            for (int n = 0; n < 4; n++)
                outptr_int8[n] = top_blob.channel(p0 + n).row<signed char>(0);
        }
        else
        {
            for (int n = 0; n < 4; n++)
                outptr_fp32[n] = top_blob.channel(p0 + n).row<float>(0);
        }

        for (int s = 0; s < size; s += 4)
        {
            int acc[16] = {0};
            signed char apack[32];
            int partial[16];

            for (int kb = 0; kb < kblocks; kb++)
            {
                for (int m = 0; m < 4; m++)
                {
                    for (int k = 0; k < 8; k++)
                        apack[m * 8 + k] = inptrs[kb * 8 + k][s + m];
                }

                ncnn_convolution_1x1_int8_xsmtvdot_4x4(apack, bpack_group + (size_t)kb * 32, partial);
                for (int i = 0; i < 16; i++)
                    acc[i] += partial[i];
            }

            if (use_int8_requantize)
            {
                xsmtvdot_store_int8_tile(acc, scale_in, bias, scale_out, outptr_int8, s);
            }
            else
            {
                xsmtvdot_store_fp32_tile(acc, scale_in, bias, outptr_fp32, s);
            }
        }
    }

    return 0;
}

int convolution_1x1_int8_xsmtvdot_forward_4x4k_apanel_experimental(const Mat& bottom_blob_int8,
                                                                   Mat& top_blob,
                                                                   const Mat& weight_data_tm,
                                                                   const Mat& bias_data,
                                                                   const Mat& bottom_blob_int8_scales,
                                                                   const Mat& weight_data_int8_scales,
                                                                   const Mat& top_blob_int8_scales,
                                                                   int bias_term,
                                                                   int int8_scale_term,
                                                                   int activation_type,
                                                                   int num_output,
                                                                   const Option& opt)
{
    if (opt.num_threads < 1)
        return 1;

    if (activation_type != 0)
        return 1;

    if (bottom_blob_int8.dims != 3 || bottom_blob_int8.elempack != 1 || bottom_blob_int8.elemsize != (size_t)1u)
        return 1;

    const int w = bottom_blob_int8.w;
    const int h = bottom_blob_int8.h;
    const int channels = bottom_blob_int8.c;
    const int size = w * h;

    if (w <= 0 || h <= 0 || channels <= 0 || num_output <= 0)
        return 1;
    if (channels % 8 != 0 || num_output % 4 != 0 || size % 4 != 0)
        return 1;
    if (weight_data_tm.empty() || weight_data_tm.w != 32 * (channels / 8) || weight_data_tm.h != num_output / 4)
        return 1;
    if (bottom_blob_int8_scales.w != 1 || weight_data_int8_scales.w < num_output)
        return 1;

    const bool use_int8_requantize = int8_scale_term > 100;
    if (use_int8_requantize && top_blob_int8_scales.w < 1)
        return 1;

    top_blob.create(w, h, num_output, use_int8_requantize ? (size_t)1u : (size_t)4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

    std::vector<const signed char*> inptrs(channels);
    for (int q = 0; q < channels; q++)
        inptrs[q] = bottom_blob_int8.channel(q).row<const signed char>(0);

    const int kblocks = channels / 8;

    const float bottom_scale = bottom_blob_int8_scales[0];
    const float scale_out = use_int8_requantize ? top_blob_int8_scales[0] : 0.f;

    std::vector<float> scale_in_data(num_output);
    std::vector<float> bias_data_local(num_output);
    for (int p = 0; p < num_output; p++)
    {
        scale_in_data[p] = weight_data_int8_scales[p] != 0 ? 1.f / (bottom_scale * weight_data_int8_scales[p]) : 0.f;
        bias_data_local[p] = bias_term ? bias_data[p] : 0.f;
    }

    std::vector<signed char*> outptr_int8_channels;
    std::vector<float*> outptr_fp32_channels;
    if (use_int8_requantize)
    {
        outptr_int8_channels.resize(num_output);
        for (int p = 0; p < num_output; p++)
            outptr_int8_channels[p] = top_blob.channel(p).row<signed char>(0);
    }
    else
    {
        outptr_fp32_channels.resize(num_output);
        for (int p = 0; p < num_output; p++)
            outptr_fp32_channels[p] = top_blob.channel(p).row<float>(0);
    }

    const int tile_blocks = size / 4;
    const int effective_workers = convolution_1x1_int8_xsmtvdot_h4b_effective_workers(opt, size);

    if (opt.num_threads > 1 && effective_workers > 1 && get_omp_num_threads() != 1)
        return 1;

    const signed char* const* inptrs_ptr = inptrs.data();
    const float* scale_in_ptr = scale_in_data.data();
    const float* bias_ptr = bias_data_local.data();

    const auto worker_body = [&](int tile_begin, int tile_end) {
        std::vector<signed char> apanel((size_t)kblocks * 32);

        for (int tile = tile_begin; tile < tile_end; tile++)
        {
            const int s = tile * 4;
            for (int kb = 0; kb < kblocks; kb++)
            {
                signed char* ablock = apanel.data() + (size_t)kb * 32;
                for (int m = 0; m < 4; m++)
                {
                    for (int k = 0; k < 8; k++)
                        ablock[m * 8 + k] = inptrs_ptr[kb * 8 + k][s + m];
                }
            }

            for (int pg = 0; pg < num_output / 4; pg++)
            {
                const signed char* bpack_group = weight_data_tm.row<const signed char>(pg);
                const int p0 = pg * 4;
                int acc[16];

                ncnn_convolution_1x1_int8_xsmtvdot_4x4_kloop(apanel.data(), bpack_group, kblocks, acc);

                if (use_int8_requantize)
                {
                    signed char* outptr_int8[4] = {
                        outptr_int8_channels[p0],
                        outptr_int8_channels[p0 + 1],
                        outptr_int8_channels[p0 + 2],
                        outptr_int8_channels[p0 + 3]};
                    xsmtvdot_store_int8_tile(acc, scale_in_ptr + p0, bias_ptr + p0, scale_out, outptr_int8, s);
                }
                else
                {
                    float* outptr_fp32[4] = {
                        outptr_fp32_channels[p0],
                        outptr_fp32_channels[p0 + 1],
                        outptr_fp32_channels[p0 + 2],
                        outptr_fp32_channels[p0 + 3]};
                    xsmtvdot_store_fp32_tile(acc, scale_in_ptr + p0, bias_ptr + p0, outptr_fp32, s);
                }
            }
        }
    };

    if (effective_workers <= 1)
    {
        worker_body(0, tile_blocks);
        return 0;
    }

    #pragma omp parallel for num_threads(effective_workers) schedule(static)
    for (int wi = 0; wi < effective_workers; wi++)
    {
        const int tile_begin = (int)((long long)tile_blocks * wi / effective_workers);
        const int tile_end = (int)((long long)tile_blocks * (wi + 1) / effective_workers);
        worker_body(tile_begin, tile_end);
    }

    return 0;
}

} // namespace ncnn
