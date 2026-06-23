import { KumaModel } from "../src/index.js";

const logEl = document.querySelector<HTMLPreElement>("#log")!;
const button = document.querySelector<HTMLButtonElement>("#run")!;
const pathInput = document.querySelector<HTMLInputElement>("#path")!;
const canvas = document.querySelector<HTMLCanvasElement>("#frame")!;

function log(line: string): void {
  logEl.textContent += `\n${line}`;
}

button.addEventListener("click", () => {
  void main().catch((err: unknown) => {
    log(`ERROR: ${err instanceof Error ? err.message : String(err)}`);
    console.error(err);
    button.disabled = false;
  });
});

/** CHW (or NCHW with N=1) float data, values assumed roughly in [0,1] -- clamps and
 * scales to a displayable RGB(A) canvas frame. */
function renderFrame(data: Float32Array, shape: readonly number[]): void {
  const dims = shape.length === 4 ? shape.slice(1) : shape;
  if (dims.length !== 3) return;
  const [channels, height, width] = dims as [number, number, number];
  if (channels !== 1 && channels !== 3) return;

  canvas.width = width;
  canvas.height = height;
  const ctx2d = canvas.getContext("2d")!;
  const imageData = ctx2d.createImageData(width, height);
  const pixels = imageData.data;
  const plane = height * width;

  for (let p = 0; p < plane; p++) {
    const r = data[p]!;
    const g = channels === 3 ? data[plane + p]! : r;
    const b = channels === 3 ? data[2 * plane + p]! : r;
    const o = p * 4;
    pixels[o] = Math.round(Math.min(1, Math.max(0, r)) * 255);
    pixels[o + 1] = Math.round(Math.min(1, Math.max(0, g)) * 255);
    pixels[o + 2] = Math.round(Math.min(1, Math.max(0, b)) * 255);
    pixels[o + 3] = 255;
  }
  ctx2d.putImageData(imageData, 0, 0);
}

async function main(): Promise<void> {
  button.disabled = true;
  const path = pathInput.value.trim();
  logEl.textContent = `loading ${path}...`;
  canvas.width = 0;
  canvas.height = 0;

  const model = await KumaModel.load(path);
  log(`loaded. inputs: ${JSON.stringify(model.inputs)}`);
  log(`outputs: ${JSON.stringify(model.outputs)}`);

  const inputSpec = model.inputs[0]!;
  const n = (inputSpec.shape ?? []).reduce((a, b) => a * b, 1);
  const input = new Float32Array(n);
  if (n === 1) {
    input[0] = 0.5; // a single normalized-time-style input -- middle of the video/range
  } else {
    for (let i = 0; i < n; i++) input[i] = Math.sin(i) * 0.1; // arbitrary deterministic input
  }

  log(`running with input "${inputSpec.name}" = [${Array.from(input).join(", ")}]...`);
  const t0 = performance.now();
  const outputs = await model.run({ [inputSpec.name]: input });
  const elapsed = (performance.now() - t0).toFixed(1);

  for (const spec of model.outputs) {
    const data = outputs[spec.name];
    if (!data) continue;
    let sum = 0;
    let min = Infinity;
    let max = -Infinity;
    for (const v of data) {
      sum += v;
      if (v < min) min = v;
      if (v > max) max = v;
    }
    const mean = sum / data.length;
    log(`output "${spec.name}": n=${data.length} mean=${mean.toFixed(4)} min=${min.toFixed(4)} max=${max.toFixed(4)}`);
    if (spec.shape) renderFrame(data, spec.shape);
  }
  log(`done in ${elapsed}ms.`);
  button.disabled = false;
}
