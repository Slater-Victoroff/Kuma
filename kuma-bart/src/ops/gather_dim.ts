import type { OpContext } from "../engine/context.js";
import type { ArgValue } from "../types/manifest.js";
import { KumaUnsupportedOpError } from "../errors.js";
import { contiguousStrides, normalizeDim, numElements } from "../engine/shape.js";

const RANK_CAP = 8;

/** aten.gather.default args: (input, dim, index) — `index` has the same shape as the
 * output; out[idx] = input[idx with axis `dim` replaced by index[idx]]. Different from
 * aten.index.Tensor (ops/gather.ts): that's a 1D-index row-gather along axis 0, this is
 * a per-output-element lookup along an arbitrary axis. Implemented like a strided
 * gather (same padding trick as permute.wgsl), except the gathered axis's contribution
 * comes from a dynamic index lookup instead of a static coordinate. gather_dim.wgsl
 * supports rank <= 8 (split across two vec4<u32> pairs) — the real usage here is rank 5
 * (MosaicNika's segment-select stacks all segments along a new leading axis before
 * gathering the right one per batch element). */
export function gatherDimHandler(ctx: OpContext): void {
  const node = ctx.node;
  const [inputRef, dimArg, indexRef] = node.args as [ArgValue, number, ArgValue];
  const input = ctx.resolve(inputRef);
  const index = ctx.resolve(indexRef);
  const rank = input.shape.length;
  const dim = normalizeDim(dimArg, rank);

  if (rank > RANK_CAP) {
    throw new KumaUnsupportedOpError(
      node.target,
      node.name,
      `gather_dim.wgsl supports rank <= ${RANK_CAP}, got rank ${rank}.`,
    );
  }

  const outShape = node.meta.shape ?? index.shape;
  const inStrides = contiguousStrides(input.shape);
  const gatherStride = inStrides[dim]!;

  const pad = RANK_CAP - rank;
  const paddedOutShape = new Array(pad).fill(1).concat(outShape);
  const paddedInStrides = new Array(pad).fill(0).concat(inStrides.map((s, d) => (d === dim ? 0 : s)));
  const n = numElements(outShape);

  const out = ctx.createBuffer(outShape);
  const params = ctx.uniform([...paddedOutShape, ...paddedInStrides, gatherStride, n]);
  ctx.dispatchKernel("gather_dim.wgsl", [input.buffer, index.buffer, out, params], n);

  // Same complex-pairing gap as ops/gather.ts -- a strided gather doesn't care whether
  // the values it's moving are the real or imaginary half, so the imaginary buffer (if
  // any) needs the identical gather dispatched against it too.
  let outImag: GPUBuffer | undefined;
  if (input.imag) {
    outImag = ctx.createBuffer(outShape);
    ctx.dispatchKernel("gather_dim.wgsl", [input.imag, index.buffer, outImag, params], n);
  }

  ctx.setOutput(out, outShape, outImag);
}
