/** A minimal fake GPUDevice for exercising the scheduler's structural logic (node
 * resolution, dispatch ordering, error paths) without a real GPU. It doesn't compute
 * anything — readback buffers are zero-filled — so it can't verify numeric correctness;
 * that requires a real browser (see kuma-bart's plan notes). Tracks dispatch calls so
 * tests can assert "this op dispatched N kernels" (e.g. 0 for free passthroughs). */
export function createMockDevice() {
  const dispatches: number[] = [];

  const pass = {
    setPipeline: () => {},
    setBindGroup: () => {},
    dispatchWorkgroups: (count: number) => {
      dispatches.push(count);
    },
    end: () => {},
  };

  function fakeBuffer(size: number) {
    return {
      size,
      mapAsync: async () => {},
      getMappedRange: (_offset = 0, length: number = size) => new ArrayBuffer(length),
      unmap: () => {},
      destroy: () => {},
    };
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
    },
  };

  return { device: device as unknown as GPUDevice, pass, dispatches };
}
