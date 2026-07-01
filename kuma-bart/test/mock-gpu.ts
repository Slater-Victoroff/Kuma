/** A minimal fake GPUDevice for exercising the scheduler's structural logic (node
 * resolution, dispatch ordering, error paths) without a real GPU. It doesn't compute
 * anything — readback buffers are zero-filled — so it can't verify numeric correctness;
 * that requires a real browser (see kuma-bart's plan notes). Tracks dispatch calls so
 * tests can assert "this op dispatched N kernels" (e.g. 0 for free passthroughs). */
export function createMockDevice() {
  const dispatches: number[] = [];
  // Tracks every buffer .destroy() is called on -- e.g. for asserting a pooled buffer
  // (which must persist across calls, see context.ts's BufferPoolState) is never one
  // of them, the same way `dispatches` lets a test assert dispatch counts.
  const destroyedBuffers = new Set<object>();
  // Every buffer ever created (with its usage flags) -- lets a test measure leaks: a
  // buffer in `createdBuffers` but not `destroyedBuffers` is still live.
  const createdBuffers = new Set<{ usage: number }>();

  const pass = {
    setPipeline: () => {},
    setBindGroup: () => {},
    dispatchWorkgroups: (count: number) => {
      dispatches.push(count);
    },
    end: () => {},
  };

  function fakeBuffer(size: number, usage = 0) {
    const buffer = {
      size,
      usage,
      mapAsync: async () => {},
      getMappedRange: (_offset = 0, length: number = size) => new ArrayBuffer(length),
      unmap: () => {},
      destroy: () => {
        destroyedBuffers.add(buffer);
      },
    };
    createdBuffers.add(buffer);
    return buffer;
  }

  const device = {
    createBuffer: ({ size, usage = 0 }: { size: number; usage?: number }) => fakeBuffer(size, usage),
    createShaderModule: () => ({}),
    createComputePipeline: () => ({ getBindGroupLayout: () => ({}) }),
    // Binding a buffer that's already been .destroy()'d is exactly the real WebGPU
    // "[Buffer] used in submit while destroyed" failure (a use-after-destroy the
    // early-free liveness pass must never produce) -- surface it as a hard error so a
    // test can assert against it.
    createBindGroup: ({ entries }: { entries?: Array<{ resource?: { buffer?: object } }> } = {}) => {
      for (const e of entries ?? []) {
        const buf = e?.resource?.buffer;
        if (buf && destroyedBuffers.has(buf)) {
          throw new Error("mock-gpu: bind group references a destroyed buffer (use-after-destroy)");
        }
      }
      return {};
    },
    createCommandEncoder: () => ({
      beginComputePass: () => pass,
      copyBufferToBuffer: () => {},
      finish: () => ({}),
    }),
    queue: {
      writeBuffer: () => {},
      submit: () => {},
      onSubmittedWorkDone: async () => {},
    },
  };

  return { device: device as unknown as GPUDevice, pass, dispatches, destroyedBuffers, createdBuffers };
}
