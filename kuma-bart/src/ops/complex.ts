import type { OpContext } from "../engine/context.js";
import type { ArgValue } from "../types/manifest.js";
import { KumaShapeError } from "../errors.js";

/** Complex tensors are a JS-side concept only — a ResolvedTensor's `imag` field, never
 * an interleaved GPU buffer. aten.complex.default(real, imag) just pairs two already
 * -resolved real tensors; no kernel dispatch. */
export function complexHandler(ctx: OpContext): void {
  const [realRef, imagRef] = ctx.node.args as [ArgValue, ArgValue];
  const real = ctx.resolve(realRef);
  const imag = ctx.resolve(imagRef);
  ctx.setOutputTensor({ buffer: real.buffer, shape: real.shape, imag: imag.buffer });
}

export function realHandler(ctx: OpContext): void {
  const [inputRef] = ctx.node.args as [ArgValue];
  const input = ctx.resolve(inputRef);
  ctx.setOutputTensor({ buffer: input.buffer, shape: input.shape });
}

export function imagHandler(ctx: OpContext): void {
  const [inputRef] = ctx.node.args as [ArgValue];
  const input = ctx.resolve(inputRef);
  if (!input.imag) {
    throw new KumaShapeError(
      `Op "${ctx.node.target}" (node "${ctx.node.name}") expected a complex-paired tensor (with an ` +
        `imaginary part) but got a plain real tensor.`,
    );
  }
  ctx.setOutputTensor({ buffer: input.imag, shape: input.shape });
}
