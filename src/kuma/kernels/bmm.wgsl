// aten.bmm.default — batched matmul, a: (B, M, K), b: (B, K, N) -> out: (B, M, N). No bias, no transpose.

struct Params {
    batch: u32,
    m: u32,
    k: u32,
    n: u32,
};

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
) {
    let total = params.batch * params.m * params.n;
    let idx = gid.x + gid.y * (num_wg.x * 64u);
    if (idx >= total) {
        return;
    }

    let col = idx % params.n;
    let row = (idx / params.n) % params.m;
    let bat = idx / (params.n * params.m);

    let a_base = bat * params.m * params.k + row * params.k;
    let b_base = bat * params.k * params.n;

    var acc: f32 = 0.0;
    for (var i: u32 = 0u; i < params.k; i = i + 1u) {
        acc = acc + a[a_base + i] * b[b_base + i * params.n + col];
    }
    out[idx] = acc;
}
