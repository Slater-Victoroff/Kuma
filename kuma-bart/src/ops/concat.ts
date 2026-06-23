import type { OpContext, ResolvedTensor } from "../engine/context.js";
import type { ArgValue } from "../types/manifest.js";
import { KumaShapeError } from "../errors.js";
import { normalizeDim, numElements, outerInner } from "../engine/shape.js";

/** concat.wgsl only concats two inputs along one axis — an N-way concat (or stack, see
 * ops/stack.ts) lowers to a left-to-right chain of pairwise dispatches here. Shared so
 * stack can reuse it after a free unsqueeze. */
export function dispatchConcatChain(ctx: OpContext, tensors: ResolvedTensor[], dim: number): ResolvedTensor {
  let acc = tensors[0]!;
  for (let i = 1; i < tensors.length; i++) {
    const next = tensors[i]!;
    const { outer, inner } = outerInner(acc.shape, dim);
    const outShape = acc.shape.slice();
    outShape[dim] = acc.shape[dim]! + next.shape[dim]!;
    const n = numElements(outShape);

    const out = ctx.createBuffer(outShape);
    const params = ctx.uniform([outer, inner, acc.shape[dim]!, next.shape[dim]!, n]);
    ctx.dispatchKernel("concat.wgsl", [acc.buffer, next.buffer, out, params], n);
    acc = { buffer: out, shape: outShape };
  }
  return acc;
}

/** aten.cat.default args: ([tensors], dim). */
export function concatHandler(ctx: OpContext): void {
  const node = ctx.node;
  const [listArg, dimArg] = node.args as [ArgValue[], number];
  if (!Array.isArray(listArg) || listArg.length < 2) {
    throw new KumaShapeError(
      `Op "${node.target}" (node "${node.name}") needs at least 2 tensors to concat, got ` +
        `${Array.isArray(listArg) ? listArg.length : "a non-list args[0]"}.`,
    );
  }

  const tensors = listArg.map((ref) => ctx.resolve(ref));
  const dim = normalizeDim(dimArg, tensors[0]!.shape.length);
  const result = dispatchConcatChain(ctx, tensors, dim);
  ctx.setOutput(result.buffer, result.shape);
}
