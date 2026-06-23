import { describe, expect, it } from "vitest";
import { packParams, packTypedParams } from "../src/gpu/params.js";

describe("packParams", () => {
  it("pads a single field (e.g. add/mul/relu/gelu/reshape's {n}) up to 16 bytes", () => {
    const buf = packParams([42]);
    expect(buf.byteLength).toBe(16);
    expect(new Uint32Array(buf)[0]).toBe(42);
  });

  it("pads 3 fields (linear.wgsl's {m,k,n}) up to 16 bytes", () => {
    expect(packParams([1, 2, 3]).byteLength).toBe(16);
  });

  it("pads 5 fields (concat.wgsl's params) up to 32 bytes", () => {
    expect(packParams([1, 2, 3, 4, 5]).byteLength).toBe(32);
  });

  it("pads 7 fields (slice.wgsl's params) up to 32 bytes", () => {
    expect(packParams([1, 2, 3, 4, 5, 6, 7]).byteLength).toBe(32);
  });

  it("pads 9 fields (permute.wgsl's vec4<u32> + vec4<u32> + u32) up to 48 bytes", () => {
    const fields = Array.from({ length: 9 }, (_, i) => i);
    const buf = packParams(fields);
    expect(buf.byteLength).toBe(48);
    expect(Array.from(new Uint32Array(buf).slice(0, 9))).toEqual(fields);
  });

  it("pads 14 fields (conv2d.wgsl's params) up to 64 bytes", () => {
    expect(packParams(new Array(14).fill(7)).byteLength).toBe(64);
  });

  it("zero-fills the padding tail", () => {
    const view = new Uint32Array(packParams([1]));
    expect(Array.from(view)).toEqual([1, 0, 0, 0]);
  });
});

describe("packTypedParams", () => {
  it("writes clamp.wgsl's {n:u32, min_val:f32, max_val:f32} byte-exact", () => {
    const buf = packTypedParams([{ u32: 100 }, { f32: -1.5 }, { f32: 2.5 }]);
    expect(buf.byteLength).toBe(16);
    const view = new DataView(buf);
    expect(view.getUint32(0, true)).toBe(100);
    expect(view.getFloat32(4, true)).toBeCloseTo(-1.5);
    expect(view.getFloat32(8, true)).toBeCloseTo(2.5);
  });

  it("writes group_norm's eps:f32 sitting between u32 counts", () => {
    // {batch,channels,spatial,groups,channels_per_group:u32, eps:f32, rows:u32} = 7 fields -> 32 bytes
    const buf = packTypedParams([
      { u32: 1 },
      { u32: 8 },
      { u32: 16 },
      { u32: 2 },
      { u32: 4 },
      { f32: 1e-5 },
      { u32: 2 },
    ]);
    expect(buf.byteLength).toBe(32);
    const view = new DataView(buf);
    expect(view.getUint32(0, true)).toBe(1);
    expect(view.getUint32(16, true)).toBe(4);
    expect(view.getFloat32(20, true)).toBeCloseTo(1e-5);
    expect(view.getUint32(24, true)).toBe(2);
  });

  it("does not truncate a float field the way a Uint32Array assignment would", () => {
    const buf = packTypedParams([{ f32: 0.5 }]);
    expect(new DataView(buf).getFloat32(0, true)).toBeCloseTo(0.5);
  });

  it("zero-fills the padding tail", () => {
    const buf = packTypedParams([{ u32: 7 }]);
    expect(buf.byteLength).toBe(16);
    expect(Array.from(new Uint32Array(buf))).toEqual([7, 0, 0, 0]);
  });
});
