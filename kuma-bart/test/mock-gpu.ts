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

  const pass = {
    setPipeline: () => {},
    setBindGroup: () => {},
    dispatchWorkgroups: (count: number) => {
      dispatches.push(count);
    },
    end: () => {},
  };

  function fakeBuffer(size: number) {
    const buffer = {
      size,
      mapAsync: async () => {},
      getMappedRange: (_offset = 0, length: number = size) => new ArrayBuffer(length),
      unmap: () => {},
      destroy: () => {
        destroyedBuffers.add(buffer);
      },
    };
    return buffer;
  }

  const device = {
    createBuffer: ({ size }: { size: number }) => fakeBuffer(size),
    createShaderModule: () => ({}),
    createComputePipeline: () => ({ getBindGroupLayout: () => ({}) }),
    createBindGroup: () => ({}),
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

  return { device: device as unknown as GPUDevice, pass, dispatches, destroyedBuffers };
}
