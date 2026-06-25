// aten.addmm.default / aten.mm.default — y = x @ weight^T + bias.
// x: (M, K), weight: (N, K), bias: (N).
//
// Shared-memory-tiled matmul: one workgroup per 16x16 output tile, instead of the
// original one-thread-per-output-element version, which read x/weight straight from
// global memory inside its K-length loop -- every element of x got re-read from global
// memory N times (once per output column reusing that row) and every element of weight
// got re-read M times, with zero reuse. That's memory-bandwidth-bound, not
// compute-bound, which is exactly backwards for a matmul.
//
// Here, each workgroup loads one 16x16 tile of x and one 16x16 tile of weight into
// workgroup-shared memory *once*, and all 256 threads in the workgroup reuse those same
// 32 cached values for their own partial dot product before moving to the next K-tile --
// a 16x reduction in global memory traffic per operand, the standard fix for this class
// of kernel.
//
// Dispatch convention: callers must use OpContext.dispatchKernelGrid (not dispatchKernel)
// with grid = (ceil(N/16), ceil(M/16)) -- workgroup_id is read directly as 2D tile
// coordinates here, not folded into a linear index the way dispatchKernel's other
// kernels are. See ops/linear.ts's dispatchLinear.

const TILE: u32 = 16u;

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

var<workgroup> a_tile: array<f32, 256>; // a_tile[ty*TILE+tx] = x[row][k_base+tx], row fixed per ty
var<workgroup> b_tile: array<f32, 256>; // b_tile[ty*TILE+tx] = weight[col][k_base+ty], col fixed per tx

@compute @workgroup_size(16, 16)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let tx = local_id.x;
    let ty = local_id.y;
    let row = wg_id.y * TILE + ty;
    let col = wg_id.x * TILE + tx;

    var acc: f32 = 0.0;
    let num_tiles = (params.k + TILE - 1u) / TILE;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let k_base = t * TILE;

        let a_k = k_base + tx;
        a_tile[ty * TILE + tx] = select(0.0, x[row * params.k + a_k], row < params.m && a_k < params.k);

        let b_k = k_base + ty;
        b_tile[ty * TILE + tx] = select(0.0, weight[col * params.k + b_k], col < params.n && b_k < params.k);

        workgroupBarrier();

        for (var kk: u32 = 0u; kk < TILE; kk = kk + 1u) {
            acc = acc + a_tile[ty * TILE + kk] * b_tile[kk * TILE + tx];
        }

        workgroupBarrier();
    }

    if (row < params.m && col < params.n) {
        out[row * params.n + col] = acc + bias[col];
    }
}
