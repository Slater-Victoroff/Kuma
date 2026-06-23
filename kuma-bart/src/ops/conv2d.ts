import type { OpContext, ResolvedTensor } from "../engine/context.js";
import type { ArgValue } from "../types/manifest.js";
import { KumaShapeError, KumaUnsupportedOpError } from "../errors.js";
import { numElements } from "../engine/shape.js";

function dispatchConv2d(
  ctx: OpContext,
  input: ResolvedTensor,
  weight: ResolvedTensor,
  biasRef: ArgValue,
  strideArg: number[],
  paddingArg: number[],
  dilationArg: number[],
  groups: number,
): void {
  const node = ctx.node;
  if (dilationArg[0] !== 1 || dilationArg[1] !== 1) {
    throw new KumaUnsupportedOpError(
      node.target,
      node.name,
      `dilation ${JSON.stringify(dilationArg)} is not supported — conv2d.wgsl has no dilation parameter.`,
    );
  }

  const outShape = node.meta.shape;
  if (!outShape || outShape.length !== 4) {
    throw new KumaShapeError(`Op "${node.target}" (node "${node.name}") is missing a 4D output shape in the manifest.`);
  }
  const [batch, outChannels, outH, outW] = outShape as [number, number, number, number];
  const inChannels = input.shape[1]!;
  const inH = input.shape[2]!;
  const inW = input.shape[3]!;
  const kh = weight.shape[2]!;
  const kw = weight.shape[3]!;
  const [strideH, strideW] = strideArg;
  const [padH, padW] = paddingArg;

  const bias = biasRef === null ? { buffer: ctx.zeros(outChannels * 4), shape: [outChannels] } : ctx.resolve(biasRef);

  const out = ctx.createBuffer(outShape);
  const params = ctx.uniform([
    batch,
    inChannels,
    inH,
    inW,
    outChannels,
    outH,
    outW,
    kh,
    kw,
    strideH!,
    strideW!,
    padH!,
    padW!,
    groups,
  ]);
  ctx.dispatchKernel("conv2d.wgsl", [input.buffer, weight.buffer, bias.buffer, out, params], numElements(outShape));
  ctx.setOutput(out, outShape);
}

/** aten.convolution.default args: (input, weight, bias, stride[2], padding[2],
 * dilation[2], transposed:bool, output_padding[2], groups:int). Trailing args at their
 * schema default (stride=1, padding=0, dilation=1, groups=1) are sometimes omitted by
 * the tracer entirely rather than serialized explicitly — same as aten.conv2d.default
 * below — so every arg past bias is read positionally with an explicit fallback. */
export function conv2dHandler(ctx: OpContext): void {
  const node = ctx.node;
  const inputRef = node.args[0] as ArgValue;
  const weightRef = node.args[1] as ArgValue;
  const biasRef = (node.args[2] as ArgValue | undefined) ?? null;
  const strideArg = (node.args[3] as number[] | undefined) ?? [1, 1];
  const paddingArg = (node.args[4] as number[] | undefined) ?? [0, 0];
  const dilationArg = (node.args[5] as number[] | undefined) ?? [1, 1];
  const transposedArg = (node.args[6] as boolean | undefined) ?? false;
  const groupsArg = (node.args[8] as number | undefined) ?? 1;

  if (transposedArg) {
    throw new KumaUnsupportedOpError(
      node.target,
      node.name,
      "ConvTranspose2d (transposed convolution) has no WGSL kernel yet.",
    );
  }

  const input = ctx.resolve(inputRef);
  const weight = ctx.resolve(weightRef);
  dispatchConv2d(ctx, input, weight, biasRef, strideArg, paddingArg, dilationArg, groupsArg);
}

/** aten.conv2d.default — the simpler non-transposed-capable overload:
 * (input, weight, bias, stride[2], padding[2], dilation[2], groups:int). Same kernel,
 * shorter arg list (no transposed/output_padding fields at all).
 *
 * Confirmed against the real model: most calls here are the bare 3-arg form
 * (input, weight, bias) with stride=1/padding=0/dilation=1/groups=1 all omitted
 * because they're at their PyTorch default — the tracer only serializes args up to
 * the last one actually passed non-default, not the full canonical schema. Every
 * trailing arg needs its own explicit default, not just a positional destructure. */
export function conv2dSimpleHandler(ctx: OpContext): void {
  const node = ctx.node;
  const inputRef = node.args[0] as ArgValue;
  const weightRef = node.args[1] as ArgValue;
  const biasRef = (node.args[2] as ArgValue | undefined) ?? null;
  const strideArg = (node.args[3] as number[] | undefined) ?? [1, 1];
  const paddingArg = (node.args[4] as number[] | undefined) ?? [0, 0];
  const dilationArg = (node.args[5] as number[] | undefined) ?? [1, 1];
  const groupsArg = (node.args[6] as number | undefined) ?? 1;

  const input = ctx.resolve(inputRef);
  const weight = ctx.resolve(weightRef);
  dispatchConv2d(ctx, input, weight, biasRef, strideArg, paddingArg, dilationArg, groupsArg);
}
