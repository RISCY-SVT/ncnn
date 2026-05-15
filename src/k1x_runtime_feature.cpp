// Copyright 2026
// SPDX-License-Identifier: BSD-3-Clause

#include "k1x_runtime_feature.h"

#include "cpu.h"

#include <cstdlib>
#include <cstring>
#include <mutex>

#if defined(__riscv) && defined(__linux__)
#include <sched.h>
#include <setjmp.h>
#include <signal.h>
#include <unistd.h>

extern "C" {
void ncnn_convolution_1x1_int8_xsmtvdot_canary(void);
}
#endif

namespace ncnn {

namespace {

enum
{
    K1X_XSMTVDOT_CLUSTER0 = 1u << 0,
    K1X_XSMTVDOT_CLUSTER1 = 1u << 1
};

enum K1xXsmtvdotReason
{
    K1X_XSMTVDOT_REASON_UNINITIALIZED = 0,
    K1X_XSMTVDOT_REASON_READY,
    K1X_XSMTVDOT_REASON_ENV_DISABLED,
    K1X_XSMTVDOT_REASON_ENV_MALFORMED,
    K1X_XSMTVDOT_REASON_AFFINITY_NOT_CONFIRMED,
    K1X_XSMTVDOT_REASON_CURRENT_AFFINITY_UNSUPPORTED,
    K1X_XSMTVDOT_REASON_CLUSTER_UNKNOWN,
    K1X_XSMTVDOT_REASON_CANARY_FAILED,
    K1X_XSMTVDOT_REASON_OPTION_POLICY_DISALLOWED,
    K1X_XSMTVDOT_REASON_THREAD_POLICY_DISALLOWED,
    K1X_XSMTVDOT_REASON_DEFAULT_DISPATCH_DISABLED,
    K1X_XSMTVDOT_REASON_NOT_RISCV_LINUX
};

struct K1xXsmtvdotRuntimeState
{
    int initialized;
    int env_opt_in_cached;
    int env_affinity_confirmation_cached;
    int ruapu_spacemit_vmadot_visible;
    int hwcap_or_hwprobe_visible;
    int canary_required;
    unsigned int canary_attempted_mask;
    unsigned int canary_success_mask;
    unsigned int supported_cluster_mask;
    unsigned int tested_cluster_mask;
    unsigned int failed_or_trap_cluster_mask;
    unsigned int unknown_cluster_mask;
    unsigned int current_policy_flags;
    int last_reason_code;
#if defined(__riscv) && defined(__linux__)
    cpu_set_t cluster0_cpu_set;
    cpu_set_t current_affinity_snapshot;
#endif
};

static std::mutex g_k1x_xsmtvdot_lock;
static K1xXsmtvdotRuntimeState g_k1x_xsmtvdot_state;

static const char* k1x_xsmtvdot_reason_string(int reason)
{
    switch (reason)
    {
    case K1X_XSMTVDOT_REASON_READY:
        return "ready";
    case K1X_XSMTVDOT_REASON_ENV_DISABLED:
        return "env-disabled";
    case K1X_XSMTVDOT_REASON_ENV_MALFORMED:
        return "env-malformed";
    case K1X_XSMTVDOT_REASON_AFFINITY_NOT_CONFIRMED:
        return "affinity-not-confirmed";
    case K1X_XSMTVDOT_REASON_CURRENT_AFFINITY_UNSUPPORTED:
        return "current-affinity-unsupported";
    case K1X_XSMTVDOT_REASON_CLUSTER_UNKNOWN:
        return "cluster-unknown";
    case K1X_XSMTVDOT_REASON_CANARY_FAILED:
        return "canary-failed";
    case K1X_XSMTVDOT_REASON_OPTION_POLICY_DISALLOWED:
        return "option-policy-disallowed";
    case K1X_XSMTVDOT_REASON_THREAD_POLICY_DISALLOWED:
        return "thread-policy-disallowed";
    case K1X_XSMTVDOT_REASON_DEFAULT_DISPATCH_DISABLED:
        return "default-dispatch-disabled";
    case K1X_XSMTVDOT_REASON_NOT_RISCV_LINUX:
        return "not-riscv-linux";
    default:
        return "uninitialized";
    }
}

static int k1x_env_truthy(const char* value)
{
    if (!value || value[0] == '\0' || value[0] == '0')
        return 0;

    return 1;
}

#if defined(__riscv) && defined(__linux__)
static thread_local sigjmp_buf g_k1x_xsmtvdot_sigjmp;
static thread_local volatile sig_atomic_t g_k1x_xsmtvdot_sigill_seen = 0;

static void k1x_xsmtvdot_sigill_handler(int signo, siginfo_t* info, void* uctx)
{
    (void)signo;
    (void)info;
    (void)uctx;
    g_k1x_xsmtvdot_sigill_seen = 1;
    siglongjmp(g_k1x_xsmtvdot_sigjmp, 1);
}

static int k1x_parse_cluster_cpu_env(const char* text, cpu_set_t* set)
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

static int k1x_build_cluster0_cpu_set_locked(cpu_set_t* set)
{
    const char* env = getenv("NCNN_RISCV_INT8_XSMTVDOT_CLUSTER0_CPUS");
    if (env && env[0] != '\0')
        return k1x_parse_cluster_cpu_env(env, set);

    CPU_ZERO(set);
    CPU_SET(0, set);
    CPU_SET(1, set);
    CPU_SET(2, set);
    CPU_SET(3, set);
    return 0;
}

static int k1x_cpu_is_in_set(int cpu, const cpu_set_t* set)
{
    return cpu >= 0 && cpu < CPU_SETSIZE && CPU_ISSET(cpu, set);
}

static int k1x_observed_affinity_is_subset(const cpu_set_t* observed, const cpu_set_t* allowed)
{
    int any = 0;
    for (int cpu = 0; cpu < CPU_SETSIZE; cpu++)
    {
        if (!CPU_ISSET(cpu, observed))
            continue;

        any = 1;
        if (!CPU_ISSET(cpu, allowed))
            return 0;
    }

    return any;
}

static int k1x_xsmtvdot_canary_ok_locked()
{
    struct sigaction sa;
    struct sigaction old_sa;
    std::memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = k1x_xsmtvdot_sigill_handler;
    sa.sa_flags = SA_SIGINFO;
    sigemptyset(&sa.sa_mask);

    if (sigaction(SIGILL, &sa, &old_sa) != 0)
        return 0;

    g_k1x_xsmtvdot_sigill_seen = 0;
    int ok = 0;
    if (sigsetjmp(g_k1x_xsmtvdot_sigjmp, 1) == 0)
    {
        ncnn_convolution_1x1_int8_xsmtvdot_canary();
        ok = g_k1x_xsmtvdot_sigill_seen ? 0 : 1;
    }

    sigaction(SIGILL, &old_sa, 0);
    return ok;
}

static int k1x_xsmtvdot_current_affinity_cluster0_locked()
{
    CPU_ZERO(&g_k1x_xsmtvdot_state.current_affinity_snapshot);
    if (sched_getaffinity(0, sizeof(cpu_set_t), &g_k1x_xsmtvdot_state.current_affinity_snapshot) != 0)
        return 0;

    if (!k1x_observed_affinity_is_subset(&g_k1x_xsmtvdot_state.current_affinity_snapshot, &g_k1x_xsmtvdot_state.cluster0_cpu_set))
        return 0;

    const int cpu = sched_getcpu();
    if (!k1x_cpu_is_in_set(cpu, &g_k1x_xsmtvdot_state.cluster0_cpu_set))
        return 0;

    return 1;
}
#endif

static void k1x_xsmtvdot_initialize_baseline_locked()
{
    if (g_k1x_xsmtvdot_state.initialized)
        return;

    std::memset(&g_k1x_xsmtvdot_state, 0, sizeof(g_k1x_xsmtvdot_state));
    g_k1x_xsmtvdot_state.initialized = 1;
    g_k1x_xsmtvdot_state.canary_required = 1;
    g_k1x_xsmtvdot_state.unknown_cluster_mask = K1X_XSMTVDOT_CLUSTER0 | K1X_XSMTVDOT_CLUSTER1;
    g_k1x_xsmtvdot_state.last_reason_code = K1X_XSMTVDOT_REASON_UNINITIALIZED;
#if defined(__riscv) && defined(__linux__)
    CPU_ZERO(&g_k1x_xsmtvdot_state.cluster0_cpu_set);
    CPU_ZERO(&g_k1x_xsmtvdot_state.current_affinity_snapshot);
#endif
}

static int k1x_xsmtvdot_refresh_env_locked()
{
    k1x_xsmtvdot_initialize_baseline_locked();

    const char* enable = getenv("NCNN_RISCV_INT8_XSMTVDOT_1X1_ENABLE");
    g_k1x_xsmtvdot_state.env_opt_in_cached = k1x_env_truthy(enable);

    const char* affinity_confirmed = getenv("NCNN_RISCV_INT8_XSMTVDOT_CLUSTER0_AFFINITY_CONFIRMED");
    g_k1x_xsmtvdot_state.env_affinity_confirmation_cached = affinity_confirmed && affinity_confirmed[0] == '1';

#if defined(__riscv) && defined(__linux__)
    g_k1x_xsmtvdot_state.ruapu_spacemit_vmadot_visible = cpu_support_k1x_xsmtvdot();
    g_k1x_xsmtvdot_state.hwcap_or_hwprobe_visible = cpu_support_riscv_v();

    if (k1x_build_cluster0_cpu_set_locked(&g_k1x_xsmtvdot_state.cluster0_cpu_set) != 0)
    {
        g_k1x_xsmtvdot_state.last_reason_code = K1X_XSMTVDOT_REASON_ENV_MALFORMED;
        return 0;
    }
#else
    g_k1x_xsmtvdot_state.last_reason_code = K1X_XSMTVDOT_REASON_NOT_RISCV_LINUX;
    return 0;
#endif

    return 1;
}

static int k1x_xsmtvdot_prepare_cluster0_locked()
{
#if defined(__riscv) && defined(__linux__)
    if (!k1x_xsmtvdot_refresh_env_locked())
        return 0;

    if (!g_k1x_xsmtvdot_state.env_opt_in_cached)
    {
        g_k1x_xsmtvdot_state.last_reason_code = K1X_XSMTVDOT_REASON_ENV_DISABLED;
        return 0;
    }

    if (!g_k1x_xsmtvdot_state.env_affinity_confirmation_cached)
    {
        g_k1x_xsmtvdot_state.last_reason_code = K1X_XSMTVDOT_REASON_AFFINITY_NOT_CONFIRMED;
        return 0;
    }

    if (!k1x_xsmtvdot_current_affinity_cluster0_locked())
    {
        g_k1x_xsmtvdot_state.last_reason_code = K1X_XSMTVDOT_REASON_CURRENT_AFFINITY_UNSUPPORTED;
        return 0;
    }

    if (!(g_k1x_xsmtvdot_state.canary_attempted_mask & K1X_XSMTVDOT_CLUSTER0))
    {
        g_k1x_xsmtvdot_state.canary_attempted_mask |= K1X_XSMTVDOT_CLUSTER0;
        g_k1x_xsmtvdot_state.tested_cluster_mask |= K1X_XSMTVDOT_CLUSTER0;

        if (k1x_xsmtvdot_canary_ok_locked())
        {
            g_k1x_xsmtvdot_state.canary_success_mask |= K1X_XSMTVDOT_CLUSTER0;
            g_k1x_xsmtvdot_state.supported_cluster_mask |= K1X_XSMTVDOT_CLUSTER0;
            g_k1x_xsmtvdot_state.unknown_cluster_mask &= ~K1X_XSMTVDOT_CLUSTER0;
        }
        else
        {
            g_k1x_xsmtvdot_state.failed_or_trap_cluster_mask |= K1X_XSMTVDOT_CLUSTER0;
            g_k1x_xsmtvdot_state.unknown_cluster_mask &= ~K1X_XSMTVDOT_CLUSTER0;
        }
    }

    if (!(g_k1x_xsmtvdot_state.supported_cluster_mask & K1X_XSMTVDOT_CLUSTER0))
    {
        g_k1x_xsmtvdot_state.last_reason_code = K1X_XSMTVDOT_REASON_CANARY_FAILED;
        return 0;
    }

    if (!k1x_xsmtvdot_current_affinity_cluster0_locked())
    {
        g_k1x_xsmtvdot_state.last_reason_code = K1X_XSMTVDOT_REASON_CURRENT_AFFINITY_UNSUPPORTED;
        return 0;
    }

    g_k1x_xsmtvdot_state.last_reason_code = K1X_XSMTVDOT_REASON_READY;
    return 1;
#else
    g_k1x_xsmtvdot_state.last_reason_code = K1X_XSMTVDOT_REASON_NOT_RISCV_LINUX;
    return 0;
#endif
}

} // namespace

int k1x_xsmtvdot_runtime_init()
{
    std::lock_guard<std::mutex> lock(g_k1x_xsmtvdot_lock);

    if (!k1x_xsmtvdot_refresh_env_locked())
        return -1;

    if (!g_k1x_xsmtvdot_state.env_opt_in_cached)
    {
        g_k1x_xsmtvdot_state.last_reason_code = K1X_XSMTVDOT_REASON_ENV_DISABLED;
        return 0;
    }

    return 0;
}

int k1x_xsmtvdot_runtime_init_for_cluster_mask(unsigned int cluster_mask)
{
    std::lock_guard<std::mutex> lock(g_k1x_xsmtvdot_lock);

    if (cluster_mask & K1X_XSMTVDOT_CLUSTER1)
    {
        k1x_xsmtvdot_refresh_env_locked();
        g_k1x_xsmtvdot_state.last_reason_code = K1X_XSMTVDOT_REASON_CLUSTER_UNKNOWN;
        return -1;
    }

    if (cluster_mask & K1X_XSMTVDOT_CLUSTER0)
        return k1x_xsmtvdot_prepare_cluster0_locked() ? 0 : -1;

    return k1x_xsmtvdot_refresh_env_locked() ? 0 : -1;
}

int k1x_xsmtvdot_available()
{
    std::lock_guard<std::mutex> lock(g_k1x_xsmtvdot_lock);
    k1x_xsmtvdot_refresh_env_locked();
    return g_k1x_xsmtvdot_state.supported_cluster_mask != 0;
}

int k1x_xsmtvdot_available_for_cluster(int cluster_id)
{
    std::lock_guard<std::mutex> lock(g_k1x_xsmtvdot_lock);
    unsigned int cluster_mask = 0u;
    if (cluster_id == 0)
        cluster_mask = K1X_XSMTVDOT_CLUSTER0;
    else if (cluster_id == 1)
        cluster_mask = K1X_XSMTVDOT_CLUSTER1;

    if (cluster_mask == 0)
    {
        g_k1x_xsmtvdot_state.last_reason_code = K1X_XSMTVDOT_REASON_CLUSTER_UNKNOWN;
        return 0;
    }

    if (cluster_mask == K1X_XSMTVDOT_CLUSTER0 && !(g_k1x_xsmtvdot_state.canary_attempted_mask & K1X_XSMTVDOT_CLUSTER0))
        k1x_xsmtvdot_prepare_cluster0_locked();

    if (cluster_mask == K1X_XSMTVDOT_CLUSTER1)
        g_k1x_xsmtvdot_state.last_reason_code = K1X_XSMTVDOT_REASON_CLUSTER_UNKNOWN;

    return (g_k1x_xsmtvdot_state.supported_cluster_mask & cluster_mask) != 0;
}

int k1x_xsmtvdot_available_for_current_affinity()
{
    std::lock_guard<std::mutex> lock(g_k1x_xsmtvdot_lock);
    return k1x_xsmtvdot_prepare_cluster0_locked();
}

unsigned int k1x_xsmtvdot_supported_cluster_mask()
{
    std::lock_guard<std::mutex> lock(g_k1x_xsmtvdot_lock);
    return g_k1x_xsmtvdot_state.supported_cluster_mask;
}

int k1x_xsmtvdot_policy_allows(const Option& opt, int activation_type, int num_threads, int input_elempack, int output_elempack)
{
    std::lock_guard<std::mutex> lock(g_k1x_xsmtvdot_lock);

    if (k1x_xsmtvdot_default_dispatch_enabled())
    {
        g_k1x_xsmtvdot_state.last_reason_code = K1X_XSMTVDOT_REASON_DEFAULT_DISPATCH_DISABLED;
        return 0;
    }

    if (num_threads != 1 || opt.num_threads != 1)
    {
        g_k1x_xsmtvdot_state.last_reason_code = K1X_XSMTVDOT_REASON_THREAD_POLICY_DISALLOWED;
        return 0;
    }

    if (activation_type != 0 || input_elempack != 1 || output_elempack != 1)
    {
        g_k1x_xsmtvdot_state.last_reason_code = K1X_XSMTVDOT_REASON_OPTION_POLICY_DISALLOWED;
        return 0;
    }

    return k1x_xsmtvdot_prepare_cluster0_locked();
}

const char* k1x_xsmtvdot_runtime_reason()
{
    std::lock_guard<std::mutex> lock(g_k1x_xsmtvdot_lock);
    return k1x_xsmtvdot_reason_string(g_k1x_xsmtvdot_state.last_reason_code);
}

int k1x_xsmtvdot_default_dispatch_enabled()
{
    return 0;
}

} // namespace ncnn
