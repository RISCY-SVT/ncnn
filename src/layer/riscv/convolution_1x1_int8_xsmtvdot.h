// Copyright 2026
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_RISCV_CONVOLUTION_1X1_INT8_XSMTVDOT_H
#define LAYER_RISCV_CONVOLUTION_1X1_INT8_XSMTVDOT_H

#include "mat.h"
#include "option.h"

extern "C" {
void ncnn_convolution_1x1_int8_xsmtvdot_canary(void);
void ncnn_convolution_1x1_int8_xsmtvdot_4x4(const signed char* a, const signed char* packed_b, int* c);
}

namespace ncnn {

int convolution_1x1_int8_xsmtvdot_create_weight_tm(const Mat& weight_data, Mat& weight_data_tm, int num_input, int num_output);

int convolution_1x1_int8_xsmtvdot_pipeline_enabled(void);

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
                                          const Option& opt);

int convolution_1x1_int8_xsmtvdot_runtime_ready(void);

} // namespace ncnn

#endif // LAYER_RISCV_CONVOLUTION_1X1_INT8_XSMTVDOT_H
