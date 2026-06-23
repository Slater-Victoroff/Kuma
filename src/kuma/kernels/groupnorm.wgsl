// aten.native_group_norm.default — normalize each (batch, group) block of `group_size` =
// channels_per_group * H * W contiguous elements, then apply per-channel affine.
// NCHW layout; weight/bias are length `channels`. One invocation per (batch, group) row.

struct Params {
    batch: u32,
    channels: u32,
    spatial: u32,        // H * W
    groups: u32,
    channels_per_group: u32,
    eps: f32,
    rows: u32,           // batch * groups
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let row = gid.x + gid.y * (num_wg.x * 64u);
    if (row >= params.rows) {
        return;
    }
    let b = row / params.groups;
    let g = row % params.groups;
    let group_size = params.channels_per_group * params.spatial;
    let base = (b * params.channels + g * params.channels_per_group) * params.spatial;

    var mean: f32 = 0.0;
    for (var i: u32 = 0u; i < group_size; i = i + 1u) {
        mean = mean + input[base + i];
    }
    mean = mean / f32(group_size);

    var variance: f32 = 0.0;
    for (var i: u32 = 0u; i < group_size; i = i + 1u) {
        let d = input[base + i] - mean;
        variance = variance + d * d;
    }
    variance = variance / f32(group_size);
    let rstd = inverseSqrt(variance + params.eps);

    for (var cg: u32 = 0u; cg < params.channels_per_group; cg = cg + 1u) {
        let c = g * params.channels_per_group + cg;
        let w = weight[c];
        let bi = bias[c];
        let chan_base = base + cg * params.spatial;
        for (var s: u32 = 0u; s < params.spatial; s = s + 1u) {
            let normed = (input[chan_base + s] - mean) * rstd;
            out[chan_base + s] = normed * w + bi;
        }
    }
}
