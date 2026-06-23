/**
 * Pack a flat list of u32 fields (in WGSL struct declaration order, with any vecN<u32>
 * member flattened into N consecutive numbers) into a uniform-buffer-ready ArrayBuffer.
 *
 * Every kernel's Params struct in this project is u32-only and either flat scalars or a
 * vec4<u32> as the first member(s) — in both cases a vec4<u32> occupies exactly 16 bytes
 * with no internal padding, so flattening it into 4 consecutive u32s reproduces the exact
 * WGSL layout. The whole buffer is padded up to the next multiple of 16 bytes, matching
 * std140-style uniform struct alignment (the struct's own alignment is its largest
 * member's alignment, 16 for a vec4<u32>).
 */
export function packParams(fields: readonly number[]): ArrayBuffer {
  const dataBytes = fields.length * 4;
  const paddedBytes = Math.max(16, Math.ceil(dataBytes / 16) * 16);
  const buffer = new ArrayBuffer(paddedBytes);
  const view = new Uint32Array(buffer);
  for (let i = 0; i < fields.length; i++) {
    view[i] = fields[i]!;
  }
  return buffer;
}

export type TypedParamField = { u32: number } | { f32: number };

/**
 * Like packParams, but for Params structs that mix u32 and f32 fields (e.g. clamp's
 * `{n:u32, min_val:f32, max_val:f32}`, group_norm's `eps:f32` sitting between u32
 * counts). Every field here is still exactly 4 bytes wide and laid out sequentially —
 * WGSL doesn't care about the JS-side type, only the raw bytes, so this just needs to
 * write the right bit pattern (Uint32 vs Float32) at each 4-byte offset.
 */
export function packTypedParams(fields: readonly TypedParamField[]): ArrayBuffer {
  const dataBytes = fields.length * 4;
  const paddedBytes = Math.max(16, Math.ceil(dataBytes / 16) * 16);
  const buffer = new ArrayBuffer(paddedBytes);
  const view = new DataView(buffer);
  fields.forEach((field, i) => {
    const offset = i * 4;
    if ("u32" in field) {
      view.setUint32(offset, field.u32, true);
    } else {
      view.setFloat32(offset, field.f32, true);
    }
  });
  return buffer;
}
