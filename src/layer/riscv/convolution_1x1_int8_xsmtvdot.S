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

    .globl ncnn_convolution_1x1_int8_xsmtvdot_4x4_kloop
    .type ncnn_convolution_1x1_int8_xsmtvdot_4x4_kloop, @function
ncnn_convolution_1x1_int8_xsmtvdot_4x4_kloop:
    // a0 = A-panel, 32 bytes per kblock, layout m8k for 4 output pixels
    // a1 = packed B, 32 bytes per kblock, layout n4k8
    // a2 = kblocks
    // a3 = int32 output[16], layout m4n4
    // Integer clobbers: t0 and caller-saved argument registers a0/a1/a2.
    li t0, 16
    vsetvli t0, t0, e32, m2
    vxor.vv v28, v28, v28
    ble a2, zero, 2f
1:
    li t0, 32
    vsetvli t0, t0, e8, m1
    vle8.v v0, (a0)
    vle8.v v1, (a1)
    smt.vmadot v28, v0, v1
    addi a0, a0, 32
    addi a1, a1, 32
    addi a2, a2, -1
    bnez a2, 1b
2:
    li t0, 16
    vsetvli t0, t0, e32, m2
    vse32.v v28, (a3)
    ret
    .size ncnn_convolution_1x1_int8_xsmtvdot_4x4_kloop, .-ncnn_convolution_1x1_int8_xsmtvdot_4x4_kloop
