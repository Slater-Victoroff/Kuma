// aten.conv2d.default / aten.convolution.default — pointwise (1x1 kernel, groups=1)
// conv2d, restructured as a per-pixel channel-mixing accumulation.
//
// One thread per (batch, output spatial position) -- *not* per output element --
// computing every output channel for that position in one pass. The general direct
// kernel dispatches one thread per output element (b,oc,oh,ow), so every output channel
// independently re-reads the same input pixel's full channel vector from global memory.
// That's the real redundancy for pointwise convs (unlike weight reuse across spatial
// positions, which is small enough to already be served by the GPU's automatic cache --
// see the reverted general-kernel weight-cache attempt). Reading each input channel
// value exactly once per spatial position and reusing it across every output channel
// removes that redundancy. No shared memory or tiling needed at all -- pointwise convs
// have zero spatial overlap between neighboring output pixels (unlike depthwise), so
// there's nothing spatial to cache; this is purely a loop-order + per-thread
// accumulator fix.
//
// MAX_OUT_CHANNELS (64) bounds the per-thread accumulator array -- a too-large
// per-thread array risks register spilling, which would be exactly the kind of "added
// overhead, no offsetting benefit" mistake the weight-cache attempt made. ops/conv2d.ts
// checks out_channels against this (and that groups==1, kh==kw==1) and falls back to
// the general kernel otherwise, rather than guessing.
//
// Dispatch convention: callers use the ordinary OpContext.dispatchKernel (1D,
// dispatchElements = batch*out_h*out_w) -- one thread per spatial position, not per
// output element.

const MAX_OUT_CHANNELS: u32 = 64u;

struct Params {
    batch: u32,
    in_channels: u32,
    in_h: u32,
    in_w: u32,
    out_channels: u32,
    out_h: u32,
    out_w: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: u32,
    pad_w: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>; // (out_channels, in_channels) -- kh=kw=1, flattened
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let total = params.batch * params.out_h * params.out_w;
    let idx = gid.x + gid.y * (num_wg.x * 64u);
    if (idx >= total) {
        return;
    }

    let ow = idx % params.out_w;
    let oh = (idx / params.out_w) % params.out_h;
    let b = idx / (params.out_w * params.out_h);

    let iy_signed = i32(oh * params.stride_h) - i32(params.pad_h);
    let ix_signed = i32(ow * params.stride_w) - i32(params.pad_w);
    let in_bounds = iy_signed >= 0 && iy_signed < i32(params.in_h) && ix_signed >= 0 && ix_signed < i32(params.in_w);

    var acc: array<f32, 64>;
    for (var oc: u32 = 0u; oc < params.out_channels; oc = oc + 1u) {
        acc[oc] = bias[oc];
    }

    // Out-of-bounds (padded) position: every tap is zero, so the sum stays at bias --
    // skip the accumulation loop entirely rather than reading anything.
    if (in_bounds) {
        let iy = u32(iy_signed);
        let ix = u32(ix_signed);
        let in_plane = params.in_h * params.in_w;
        let in_base = (b * params.in_channels) * in_plane + iy * params.in_w + ix;
        for (var c: u32 = 0u; c < params.in_channels; c = c + 1u) {
            let v = input[in_base + c * in_plane];
            for (var oc: u32 = 0u; oc < params.out_channels; oc = oc + 1u) {
                acc[oc] = acc[oc] + v * weight[oc * params.in_channels + c];
            }
        }
    }

    let out_plane = params.out_h * params.out_w;
    let out_base = (b * params.out_channels) * out_plane + oh * params.out_w + ow;
    for (var oc: u32 = 0u; oc < params.out_channels; oc = oc + 1u) {
        out[out_base + oc * out_plane] = acc[oc];
    }
}
