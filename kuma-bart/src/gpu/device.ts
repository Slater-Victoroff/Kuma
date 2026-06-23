export async function requestKumaDevice(): Promise<GPUDevice> {
  if (typeof navigator === "undefined" || !("gpu" in navigator) || !navigator.gpu) {
    throw new Error(
      "WebGPU is not available in this environment (no navigator.gpu). " +
        "kuma-bart requires a browser with WebGPU support (e.g. Chrome/Edge 113+).",
    );
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("WebGPU adapter request failed — no compatible GPU adapter found.");
  }
  return adapter.requestDevice();
}
