// Copyright 2026
// SPDX-License-Identifier: BSD-3-Clause

#include "convolution_1x1_int8_xsmtvdot.h"

#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#include <setjmp.h>
#include <signal.h>
#include <unistd.h>
#endif

namespace ncnn {

static inline signed char xsmtvdot_float2int8(float v)
{
    int int32 = (int)(v >= 0.f ? v + 0.5f : v - 0.5f);
    if (int32 > 127) return 127;
    if (int32 < -127) return -127;
    return (signed char)int32;
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

#if defined(__riscv) && defined(__linux__)
static thread_local sigjmp_buf g_xsmtvdot_sigjmp;
static thread_local volatile sig_atomic_t g_xsmtvdot_sigill_seen = 0;

static void xsmtvdot_sigill_handler(int signo, siginfo_t* info, void* uctx)
{
    (void)signo;
    (void)info;
    (void)uctx;
    g_xsmtvdot_sigill_seen = 1;
    siglongjmp(g_xsmtvdot_sigjmp, 1);
}

static int parse_cpu_list(const char* text, cpu_set_t* set)
{
    const char* p = text;
    CPU_ZERO(set);

    if (!text || text[0] == '\0')
        return -1;

    while (*p)
    {
        char* end = 0;
        long first = strtol(p, &end, 10);
        long last = first;
        if (end == p || first < 0 || first >= CPU_SETSIZE)
            return -1;

        if (*end == '-')
        {
            p = end + 1;
            last = strtol(p, &end, 10);
            if (end == p || last < first || last >= CPU_SETSIZE)
                return -1;
        }

        for (long cpu = first; cpu <= last; cpu++)
            CPU_SET((int)cpu, set);

        if (*end == ',')
            p = end + 1;
        else if (*end == '\0')
            p = end;
        else
            return -1;
    }

    return 0;
}

static int cluster0_set(cpu_set_t* set)
{
    const char* env = getenv("NCNN_RISCV_INT8_XSMTVDOT_CLUSTER0_CPUS");
    if (env && env[0] != '\0')
        return parse_cpu_list(env, set);

    CPU_ZERO(set);
    CPU_SET(0, set);
    CPU_SET(1, set);
    CPU_SET(2, set);
    CPU_SET(3, set);
    return 0;
}

static int cpu_is_in_set(int cpu, const cpu_set_t* set)
{
    return cpu >= 0 && cpu < CPU_SETSIZE && CPU_ISSET(cpu, set);
}

static int observed_affinity_is_cluster0_only(const cpu_set_t* observed, const cpu_set_t* cluster0)
{
    int any = 0;
    for (int cpu = 0; cpu < CPU_SETSIZE; cpu++)
    {
        if (!CPU_ISSET(cpu, observed))
            continue;

        any = 1;
        if (!CPU_ISSET(cpu, cluster0))
            return 0;
    }

    return any;
}

static int xsmtvdot_canary_ok()
{
    struct sigaction sa;
    struct sigaction old_sa;
    std::memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = xsmtvdot_sigill_handler;
    sa.sa_flags = SA_SIGINFO;
    sigemptyset(&sa.sa_mask);

    if (sigaction(SIGILL, &sa, &old_sa) != 0)
        return 0;

    g_xsmtvdot_sigill_seen = 0;
    int ok = 0;
    if (sigsetjmp(g_xsmtvdot_sigjmp, 1) == 0)
    {
        ncnn_convolution_1x1_int8_xsmtvdot_canary();
        ok = g_xsmtvdot_sigill_seen ? 0 : 1;
    }

    sigaction(SIGILL, &old_sa, 0);
    return ok;
}
#endif

int convolution_1x1_int8_xsmtvdot_runtime_ready(void)
{
#if defined(__riscv) && defined(__linux__)
    const char* enable = getenv("NCNN_RISCV_INT8_XSMTVDOT_1X1_ENABLE");
    if (!enable || enable[0] == '\0' || enable[0] == '0')
        return 0;

    const char* affinity_confirmed = getenv("NCNN_RISCV_INT8_XSMTVDOT_CLUSTER0_AFFINITY_CONFIRMED");
    if (!affinity_confirmed || affinity_confirmed[0] != '1')
        return 0;

    cpu_set_t cluster0;
    if (cluster0_set(&cluster0) != 0)
        return 0;

    cpu_set_t observed;
    CPU_ZERO(&observed);
    if (sched_getaffinity(0, sizeof(cpu_set_t), &observed) != 0)
        return 0;
    if (!observed_affinity_is_cluster0_only(&observed, &cluster0))
        return 0;

    const int cpu_before = sched_getcpu();
    if (!cpu_is_in_set(cpu_before, &cluster0))
        return 0;

    static thread_local int canary_passed = 0;
    if (!canary_passed)
    {
        if (!xsmtvdot_canary_ok())
            return 0;
        canary_passed = 1;
    }

    const int cpu_after = sched_getcpu();
    if (!cpu_is_in_set(cpu_after, &cluster0))
        return 0;

    return 1;
#else
    return 0;
#endif
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
    if (!convolution_1x1_int8_xsmtvdot_runtime_ready())
        return 1;

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

        for (int s = 0; s < size; s += 4)
        {
            int acc[16] = {0};

            for (int kb = 0; kb < kblocks; kb++)
            {
                signed char apack[32];
                int partial[16];

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
                for (int n = 0; n < 4; n++)
                {
                    signed char* outptr = top_blob.channel(p0 + n).row<signed char>(0);
                    for (int m = 0; m < 4; m++)
                    {
                        const float sumfp32 = (float)acc[m * 4 + n] * scale_in[n] + bias[n];
                        outptr[s + m] = xsmtvdot_float2int8(sumfp32 * scale_out);
                    }
                }
            }
            else
            {
                for (int n = 0; n < 4; n++)
                {
                    float* outptr = top_blob.channel(p0 + n).row<float>(0);
                    for (int m = 0; m < 4; m++)
                        outptr[s + m] = (float)acc[m * 4 + n] * scale_in[n] + bias[n];
                }
            }
        }
    }

    return 0;
}

} // namespace ncnn
