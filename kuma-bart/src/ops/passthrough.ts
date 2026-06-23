import type { OpContext, OpHandler } from "../engine/context.js";

/** Metadata-only passthrough: same buffer, shape taken straight from this node's own
 * (authoritative) manifest meta — no kernel dispatch. Covers aten.alias.default,
 * aten.contiguous.default, aten.to.device, aten.to.dtype, aten.squeeze.dim,
 * aten.unsqueeze.default. `to.dtype` is included here even for the float32->int64 case
 * seen in this model (an index tensor feeding aten.index.Tensor) — kuma-bart keeps
 * "integer" tensors as f32-encoded values throughout and only truncates to u32 at the
 * point of use inside gather.wgsl, so no actual cast/copy is ever needed. */
export const passthroughHandler: OpHandler = (ctx) => {
  const [inputRef] = ctx.node.args;
  const input = ctx.resolve(inputRef!);
  const shape = ctx.node.meta.shape ?? input.shape;
  ctx.setOutput(input.buffer, shape, input.imag);
};

/** aten.zeros_like.default — same shape as input, freshly zero-initialized (a brand new
 * GPUBuffer is zero-initialized per the WebGPU spec, so this is also dispatch-free). */
export const zerosLikeHandler: OpHandler = (ctx) => {
  const [inputRef] = ctx.node.args;
  const input = ctx.resolve(inputRef!);
  const shape = ctx.node.meta.shape ?? input.shape;
  ctx.setOutput(ctx.createBuffer(shape), shape);
};
