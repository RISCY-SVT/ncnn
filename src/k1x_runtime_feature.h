// Copyright 2026
// SPDX-License-Identifier: BSD-3-Clause

#ifndef NCNN_K1X_RUNTIME_FEATURE_H
#define NCNN_K1X_RUNTIME_FEATURE_H

#include "option.h"
#include "platform.h"

namespace ncnn {

NCNN_EXPORT int k1x_xsmtvdot_runtime_init();
NCNN_EXPORT int k1x_xsmtvdot_runtime_init_for_cluster_mask(unsigned int cluster_mask);
NCNN_EXPORT int k1x_xsmtvdot_available();
NCNN_EXPORT int k1x_xsmtvdot_available_for_cluster(int cluster_id);
NCNN_EXPORT int k1x_xsmtvdot_available_for_current_affinity();
NCNN_EXPORT unsigned int k1x_xsmtvdot_supported_cluster_mask();
NCNN_EXPORT int k1x_xsmtvdot_policy_allows(const Option& opt, int activation_type, int num_threads, int input_elempack, int output_elempack);
NCNN_EXPORT int k1x_xsmtvdot_policy_allows_cluster0_workers(const Option& opt, int activation_type, int num_threads, int input_elempack, int output_elempack);
NCNN_EXPORT int k1x_xsmtvdot_cluster0_worker_capacity();
NCNN_EXPORT const char* k1x_xsmtvdot_runtime_reason();
NCNN_EXPORT int k1x_xsmtvdot_default_dispatch_enabled();

} // namespace ncnn

#endif // NCNN_K1X_RUNTIME_FEATURE_H
