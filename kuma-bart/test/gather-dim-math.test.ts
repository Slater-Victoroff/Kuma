import { describe, expect, it } from "vitest";
import { contiguousStrides, numElements } from "../src/engine/shape.js";

/** Independent reference: torch.gather's own definition via straightforward
 * multi-dimensional indexing (no flattening/stride tricks) — out[coords] =
 * input[coords with axis `dim` replaced by index[coords]]. */
function referenceGather(
  input: readonly number[],
  inputShape: readonly number[],
  index: readonly number[],
  outShape: readonly number[],
  dim: number,
): number[] {
  const inStrides = contiguousStrides(inputShape);
  const out = new Array(numElements(outShape));
  const rank = outShape.length;
  const coords = new Array(rank).fill(0);

  function recurse(axis: number): void {
    if (axis === rank) {
      const outIdx = coords.reduce((acc, c, d) => acc * outShape[d]! + c, 0);
      const inCoords = coords.slice();
      inCoords[dim] = index[outIdx]!;
      const inIdx = inCoords.reduce((acc, c, d) => acc + c * inStrides[d]!, 0);
      out[outIdx] = input[inIdx]!;
      return;
    }
    for (let i = 0; i < outShape[axis]!; i++) {
      coords[axis] = i;
      recurse(axis + 1);
    }
  }
  recurse(0);
  return out;
}

/** Mirrors exactly what gather_dim.wgsl computes (see kuma/src/kuma/kernels/gather_dim.wgsl
 * and ops/gather_dim.ts's param construction): decode the output's linear index into
 * per-axis coords, accumulate coord*stride for every axis except the gathered one
 * (whose stride is pre-zeroed), then add the dynamically-looked-up index*gatherStride. */
function stridedGather(
  input: readonly number[],
  paddedOutShape: readonly number[],
  paddedInStrides: readonly number[],
  gatherStride: number,
  index: readonly number[],
): number[] {
  const n = numElements(paddedOutShape);
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    let rem = i;
    let inOffset = 0;
    for (let d = 3; d >= 0; d--) {
      const extent = paddedOutShape[d]!;
      const coord = rem % extent;
      rem = Math.floor(rem / extent);
      inOffset += coord * paddedInStrides[d]!;
    }
    inOffset += index[i]! * gatherStride;
    out[i] = input[inOffset];
  }
  return out;
}

describe("gather_dim strided computation", () => {
  it("matches torch.gather's own definition via plain multi-dim indexing", () => {
    const inputShape = [2, 3, 4];
    const dim = 1;
    const outShape = [2, 5, 4];
    const input = Array.from({ length: numElements(inputShape) }, (_, i) => i + 1);
    // index[outIdx] in [0, inputShape[dim]) -- deterministic pseudo-random pattern.
    const index = Array.from({ length: numElements(outShape) }, (_, i) => (i * 7 + 3) % inputShape[dim]!);

    const expected = referenceGather(input, inputShape, index, outShape, dim);

    const inStrides = contiguousStrides(inputShape);
    const gatherStride = inStrides[dim]!;
    const paddedOutShape = [1, ...outShape];
    const paddedInStrides = [0, ...inStrides.map((s, d) => (d === dim ? 0 : s))];

    const actual = stridedGather(input, paddedOutShape, paddedInStrides, gatherStride, index);
    expect(actual).toEqual(expected);
  });

  it("matches when gathering along axis 0 (the exact MosaicNika segment-select pattern)", () => {
    // all_outputs: (num_segments=4, B=3, C=2) -- a tiny stand-in for (segments, B, C, H, W).
    const inputShape = [4, 3, 2];
    const dim = 0;
    const outShape = [1, 3, 2]; // index is broadcast from a [B]-shaped selector
    const input = Array.from({ length: numElements(inputShape) }, (_, i) => i);
    const segmentIds = [2, 0, 3]; // one segment choice per b in [0,B)
    // index broadcast: index[0,b,c] = segmentIds[b], constant across c.
    const index = Array.from({ length: numElements(outShape) }, (_, i) => {
      const b = Math.floor(i / outShape[2]!) % outShape[1]!;
      return segmentIds[b]!;
    });

    const expected = referenceGather(input, inputShape, index, outShape, dim);

    const inStrides = contiguousStrides(inputShape);
    const gatherStride = inStrides[dim]!;
    const paddedOutShape = [1, ...outShape];
    const paddedInStrides = [0, ...inStrides.map((s, d) => (d === dim ? 0 : s))];

    const actual = stridedGather(input, paddedOutShape, paddedInStrides, gatherStride, index);
    expect(actual).toEqual(expected);

    // Sanity: actual[b,c] really is input[segmentIds[b], b, c].
    for (let b = 0; b < 3; b++) {
      for (let c = 0; c < 2; c++) {
        const outIdx = b * 2 + c;
        const expectedVal = input[(segmentIds[b]! * 3 + b) * 2 + c]!;
        expect(actual[outIdx]).toBe(expectedVal);
      }
    }
  });
});
