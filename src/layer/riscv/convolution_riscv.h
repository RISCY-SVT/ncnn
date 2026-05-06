// Copyright 2021 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_CONVOLUTION_RISCV_H
#define LAYER_CONVOLUTION_RISCV_H

#include "convolution.h"

namespace ncnn {

class Convolution_riscv : public Convolution
{
public:
    Convolution_riscv();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

protected:
#if NCNN_ZFH
    int create_pipeline_fp16s(const Option& opt);
    int forward_fp16s(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
    int forward_fp16sa(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;
#endif

public:
    Layer* activation;

    Mat weight_data_tm;
    Mat weight_winograd23_data;
    Mat weight_winograd43_data;
    Mat weight_winograd63_data;
#if NCNN_INT8
    Mat weight_data_int8_1x1_packn_tm;
    Mat weight_data_int8_3x3s1_packn_tm;
    Mat weight_data_int8_1x1_xsmtvdot_tm;
#endif

    // fp16
    Mat bias_data_fp16;
};

} // namespace ncnn

#endif // LAYER_CONVOLUTION_RISCV_H
