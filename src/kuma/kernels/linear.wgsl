// aten.addmm.default / aten.mm.default — y = x @ weight^T + bias.
// x: (M, K), weight: (N, K), bias: (N) — one invocation per output element.

struct Params {
    m: u32,
    k: u32,
    n: u32,
};

@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let total = params.m * params.n;
    let idx = gid.x + gid.y * (num_wg.x * 64u);
    if (idx >= total) {
        return;
    }

    let col = idx % params.n;
    let row = idx / params.n;

    var acc: f32 = bias[col];
    for (var i: u32 = 0u; i < params.k; i = i + 1u) {
        acc = acc + x[row * params.k + i] * weight[col * params.k + i];
    }
    out[idx] = acc;
}
