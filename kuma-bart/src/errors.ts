export class KumaManifestError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "KumaManifestError";
  }
}

export class KumaUnsupportedOpError extends Error {
  constructor(target: string, nodeName: string, detail?: string) {
    super(
      `kuma-bart has no WebGPU kernel for op "${target}" (node "${nodeName}").` +
        (detail ? ` ${detail}` : " This op has no bundled WGSL kernel yet."),
    );
    this.name = "KumaUnsupportedOpError";
  }
}

export class KumaShapeError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "KumaShapeError";
  }
}
