// Copyright 2026
// SPDX-License-Identifier: BSD-3-Clause
//
// Standalone SpacemiT XSMTVDot probe/microkernel for the W1-IME-A
// experimental path. This file intentionally uses assembler-recognized
// canonical smt.vmadot mnemonics only.
//
// ABI notes:
// - lp64d C ABI entry/return.
// - Stack is not touched.
// - Integer clobbers: t0.
// - Vector clobbers: v0, v1, v28 and vector state/vtype/vl.
// - Callers must treat vector state as caller-saved across this helper.

    .text
    .align 2

    .globl ncnn_convolution_1x1_int8_xsmtvdot_canary
    .type ncnn_convolution_1x1_int8_xsmtvdot_canary, @function
ncnn_convolution_1x1_int8_xsmtvdot_canary:
    vsetvli t0, zero, e32, m2
    vxor.vv v28, v28, v28
    vsetvli t0, zero, e8, m1
    vmv.v.i v0, 0
    vmv.v.i v1, 0
    smt.vmadot v28, v0, v1
    ret
    .size ncnn_convolution_1x1_int8_xsmtvdot_canary, .-ncnn_convolution_1x1_int8_xsmtvdot_canary

    .globl ncnn_convolution_1x1_int8_xsmtvdot_4x4
    .type ncnn_convolution_1x1_int8_xsmtvdot_4x4, @function
ncnn_convolution_1x1_int8_xsmtvdot_4x4:
    vsetvli t0, zero, e32, m2
    vxor.vv v28, v28, v28
    vsetvli t0, zero, e8, m1
    vle8.v v0, (a0)
    vle8.v v1, (a1)
    smt.vmadot v28, v0, v1
    vsetvli t0, zero, e32, m2
    vse32.v v28, (a2)
    ret
    .size ncnn_convolution_1x1_int8_xsmtvdot_4x4, .-ncnn_convolution_1x1_int8_xsmtvdot_4x4
