import type { OpContext } from "../engine/context.js";
import type { ArgValue } from "../types/manifest.js";
import { numElements } from "../engine/shape.js";

/** aten.pixel_shuffle.default args: (input, upscale_factor). */
export function pixelShuffleHandler(ctx: OpContext): void {
  const node = ctx.node;
  const [inputRef, rArg] = node.args as [ArgValue, number];
  const input = ctx.resolve(inputRef);
  const [batch, inChannels, inH, inW] = input.shape as [number, number, number, number];
  const r = rArg;
  const outChannels = inChannels / (r * r);
  const outH = inH * r;
  const outW = inW * r;
  const outShape = node.meta.shape ?? [batch, outChannels, outH, outW];

  const out = ctx.createBuffer(outShape);
  const params = ctx.uniform([batch, outChannels, inH, inW, r, outH, outW]);
  ctx.dispatchKernel("pixel_shuffle.wgsl", [input.buffer, out, params], numElements(outShape));
  ctx.setOutput(out, outShape);
}
