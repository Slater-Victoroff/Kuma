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
  // Opportunistic: only enables GPU-side profiling (engine/profile.ts) when the
  // adapter actually supports it -- most desktop Chrome/Edge GPUs do, but this must be
  // requested at device-creation time (features can't be added afterward), so it's
  // requested here unconditionally rather than only when profiling is first used.
  const requiredFeatures: GPUFeatureName[] = adapter.features.has("timestamp-query") ? ["timestamp-query"] : [];
  return adapter.requestDevice({ requiredFeatures });
}
