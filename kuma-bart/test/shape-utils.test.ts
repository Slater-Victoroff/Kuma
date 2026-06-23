import { describe, expect, it } from "vitest";
import {
  clampSliceBounds,
  contiguousStrides,
  expandStrides,
  normalizeDim,
  numElements,
  outerInner,
} from "../src/engine/shape.js";
import { KumaShapeError } from "../src/errors.js";

describe("numElements", () => {
  it("multiplies dims", () => {
    expect(numElements([1, 3, 32, 32])).toBe(3072);
  });

  it("is 1 for an empty shape (scalar)", () => {
    expect(numElements([])).toBe(1);
  });
});

describe("contiguousStrides", () => {
  it("computes row-major strides for the simple.iph input shape", () => {
    expect(contiguousStrides([1, 3, 32, 32])).toEqual([3072, 1024, 32, 1]);
  });

  it("handles rank 1", () => {
    expect(contiguousStrides([5])).toEqual([1]);
  });
});

describe("normalizeDim", () => {
  it("passes through a valid positive dim", () => {
    expect(normalizeDim(1, 4)).toBe(1);
  });

  it("resolves a negative dim", () => {
    expect(normalizeDim(-1, 4)).toBe(3);
  });

  it("throws KumaShapeError for an out-of-range dim", () => {
    expect(() => normalizeDim(4, 4)).toThrow(KumaShapeError);
    expect(() => normalizeDim(-5, 4)).toThrow(KumaShapeError);
  });
});

describe("outerInner", () => {
  it("factors a 4D shape around the channel axis", () => {
    expect(outerInner([1, 3, 32, 32], 1)).toEqual({ outer: 1, inner: 1024 });
  });

  it("factors around the batch axis", () => {
    expect(outerInner([2, 3, 4], 0)).toEqual({ outer: 1, inner: 12 });
  });

  it("factors around the last axis", () => {
    expect(outerInner([2, 3, 4], 2)).toEqual({ outer: 6, inner: 1 });
  });
});

describe("expandStrides", () => {
  it("gives stride 0 for a broadcast axis, normal stride otherwise", () => {
    // (3,1,5) expanded to (3,4,5): axis 1 (size 1->4) broadcasts, others keep their stride.
    expect(expandStrides([3, 1, 5], [3, 4, 5])).toEqual([5, 0, 1]);
  });

  it("left-pads a lower-rank input (PyTorch's right-aligned expand convention)", () => {
    // (5,) expanded to (3,4,5): the new leading axes (3,4) are pure broadcasts.
    expect(expandStrides([5], [3, 4, 5])).toEqual([0, 0, 1]);
  });

  it("throws when expanding down to a lower rank", () => {
    expect(() => expandStrides([3, 4, 5], [4, 5])).toThrow(KumaShapeError);
  });

  it("throws when a non-1 axis would need to change size", () => {
    expect(() => expandStrides([3, 4, 5], [3, 9, 5])).toThrow(KumaShapeError);
  });
});

describe("clampSliceBounds", () => {
  it("defaults null start/end to the full extent", () => {
    expect(clampSliceBounds(null, null, 10)).toEqual({ start: 0, end: 10 });
  });

  it("clamps a PyTorch sentinel-huge end to the dim size", () => {
    expect(clampSliceBounds(0, Number.MAX_SAFE_INTEGER, 10)).toEqual({ start: 0, end: 10 });
  });

  it("resolves negative start/end", () => {
    expect(clampSliceBounds(-5, -1, 10)).toEqual({ start: 5, end: 9 });
  });

  it("clamps an out-of-range positive start", () => {
    expect(clampSliceBounds(20, null, 10)).toEqual({ start: 10, end: 10 });
  });
});
