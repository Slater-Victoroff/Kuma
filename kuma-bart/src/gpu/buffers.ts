function paddedSize(byteLength: number): number {
  return Math.max(4, Math.ceil(byteLength / 4) * 4);
}

/** Every tensor (weight, input, or intermediate node output) gets its own dedicated
 * buffer — no pooling/reuse/aliasing in v1, mirroring the Python compiler's own
 * "no buffer planning yet" Step 1 scope. STORAGE|COPY_SRC|COPY_DST covers every use: as
 * a kernel's input/output binding, and as the source of a final readback copy. */
export function createStorageBuffer(device: GPUDevice, byteLength: number): GPUBuffer {
  return device.createBuffer({
    size: paddedSize(byteLength),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
}

export function createUniformBuffer(device: GPUDevice, data: ArrayBuffer): GPUBuffer {
  const buffer = device.createBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

export function uploadWeightSlice(
  device: GPUDevice,
  weightsBlob: Uint8Array,
  byteOffset: number,
  byteLength: number,
): GPUBuffer {
  const buffer = createStorageBuffer(device, byteLength);
  // @webgpu/types' GPUAllowSharedBufferSource is still pinned to ArrayBufferView<ArrayBuffer>,
  // while TS 5.7+ made typed arrays generic over ArrayBufferLike (incl. SharedArrayBuffer) by
  // default — these are always concretely ArrayBuffer-backed in practice, so this is safe.
  device.queue.writeBuffer(buffer, 0, weightsBlob as unknown as Uint8Array<ArrayBuffer>, byteOffset, byteLength);
  return buffer;
}

export function uploadFloat32(device: GPUDevice, data: Float32Array): GPUBuffer {
  const buffer = createStorageBuffer(device, data.byteLength);
  device.queue.writeBuffer(buffer, 0, data as unknown as Float32Array<ArrayBuffer>);
  return buffer;
}

/** A freshly created GPUBuffer is zero-initialized per the WebGPU spec — used for
 * bias-less ops (e.g. conv2d / linear with no bias arg) that still need a bias binding. */
export function zeroBuffer(device: GPUDevice, byteLength: number): GPUBuffer {
  return createStorageBuffer(device, byteLength);
}

export async function readBuffers(
  device: GPUDevice,
  reads: { buffer: GPUBuffer; byteLength: number }[],
): Promise<Float32Array[]> {
  const staging = reads.map(({ byteLength }) =>
    device.createBuffer({
      size: paddedSize(byteLength),
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    }),
  );

  const encoder = device.createCommandEncoder();
  reads.forEach(({ buffer, byteLength }, i) => {
    encoder.copyBufferToBuffer(buffer, 0, staging[i]!, 0, paddedSize(byteLength));
  });
  device.queue.submit([encoder.finish()]);

  const results: Float32Array[] = [];
  for (let i = 0; i < reads.length; i++) {
    const stagingBuffer = staging[i]!;
    const byteLength = reads[i]!.byteLength;
    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const copy = new Float32Array(stagingBuffer.getMappedRange(0, byteLength).slice(0));
    stagingBuffer.unmap();
    stagingBuffer.destroy();
    results.push(copy);
  }
  return results;
}
