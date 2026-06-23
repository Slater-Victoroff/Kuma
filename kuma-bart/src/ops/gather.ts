import type { OpContext } from "../engine/context.js";
import type { ArgValue } from "../types/manifest.js";
import { KumaUnsupportedOpError } from "../errors.js";
import { numElements, outerInner } from "../engine/shape.js";

/** aten.index.Tensor args: (input, indices) where `indices` is a list with one entry
 * per dimension (None for dims not being indexed). Only the single-leading-axis,
 * single-1D-index-tensor pattern this model uses is supported — anything else fails
 * loudly rather than attempting general advanced indexing. */
export function gatherHandler(ctx: OpContext): void {
  const node = ctx.node;
  const [inputRef, indicesListArg] = node.args as [ArgValue, ArgValue[]];
  if (!Array.isArray(indicesListArg) || indicesListArg.length !== 1 || indicesListArg[0] === null) {
    throw new KumaUnsupportedOpError(
      node.target,
      node.name,
      `gather.wgsl only supports indexing a single leading axis (dim 0) with one index tensor, got indices=${JSON.stringify(indicesListArg)}.`,
    );
  }
  const input = ctx.resolve(inputRef);
  const indices = ctx.resolve(indicesListArg[0]!);
  if (indices.shape.length !== 1) {
    throw new KumaUnsupportedOpError(
      node.target,
      node.name,
      `gather.wgsl only supports a 1D index tensor, got shape ${JSON.stringify(indices.shape)}.`,
    );
  }

  const dim = 0;
  const inExtent = input.shape[dim]!;
  const numIndices = indices.shape[0]!;
  const { outer, inner } = outerInner(input.shape, dim);

  const outShape = node.meta.shape ?? [numIndices, ...input.shape.slice(1)];
  const n = numElements(outShape);

  const out = ctx.createBuffer(outShape);
  const params = ctx.uniform([outer, inner, inExtent, numIndices, n]);
  ctx.dispatchKernel("gather.wgsl", [input.buffer, indices.buffer, out, params], n);
  ctx.setOutput(out, outShape);
}
