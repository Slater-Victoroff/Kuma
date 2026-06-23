// Plain Node has no WebGPU globals. The production code references the real
// browser-global GPUBufferUsage/GPUMapMode enums directly (as WebGPU code normally
// does), so unit tests need these shims to import that code at all. Values match the
// WebGPU spec exactly — see https://www.w3.org/TR/webgpu/#enumdef-gpubufferusageflags.
(globalThis as Record<string, unknown>).GPUBufferUsage = {
  MAP_READ: 0x0001,
  MAP_WRITE: 0x0002,
  COPY_SRC: 0x0004,
  COPY_DST: 0x0008,
  INDEX: 0x0010,
  VERTEX: 0x0020,
  UNIFORM: 0x0040,
  STORAGE: 0x0080,
  INDIRECT: 0x0100,
  QUERY_RESOLVE: 0x0200,
};

(globalThis as Record<string, unknown>).GPUMapMode = {
  READ: 0x0001,
  WRITE: 0x0002,
};
