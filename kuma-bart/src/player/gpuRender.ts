/** Renders a model output buffer directly to a canvas via a WebGPU render
 * pipeline — no CPU round-trip. Reads the compute output storage buffer
 * directly in a fragment shader instead of deinterleaving CHW→RGBA in JS. */

const SHADER_SOURCE = `
struct Params {
  width: u32,
  height: u32,
  channels: u32,
  _pad: u32,
};

@group(0) @binding(0) var<storage, read> frame: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

struct VertexOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

// Fullscreen quad from vertex_index — no vertex buffer needed.
@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOut {
  var positions = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
    vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
  );
  var uvs = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 0.0),
    vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(1.0, 0.0),
  );
  var out: VertexOut;
  out.position = vec4<f32>(positions[idx], 0.0, 1.0);
  out.uv = uvs[idx];
  return out;
}

// CHW (or NCHW with N=1) float data clamped to [0, 1].
@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
  let x = min(u32(in.uv.x * f32(params.width)), params.width - 1u);
  let y = min(u32(in.uv.y * f32(params.height)), params.height - 1u);
  let plane = params.width * params.height;
  let p = y * params.width + x;
  let r = frame[p];
  var g = r;
  var b = r;
  if (params.channels == 3u) {
    g = frame[plane + p];
    b = frame[2u * plane + p];
  }
  return vec4<f32>(clamp(r, 0.0, 1.0), clamp(g, 0.0, 1.0), clamp(b, 0.0, 1.0), 1.0);
}
`;

export class GpuFrameRenderer {
  private readonly context: GPUCanvasContext;
  private readonly format: GPUTextureFormat;
  private readonly pipeline: GPURenderPipeline;
  private readonly paramsBuffer: GPUBuffer;
  private configuredWidth = 0;
  private configuredHeight = 0;

  constructor(
    private readonly device: GPUDevice,
    private readonly canvas: HTMLCanvasElement,
  ) {
    const context = canvas.getContext("webgpu");
    if (!context) {
      throw new Error("Failed to get a WebGPU canvas context.");
    }
    this.context = context;
    this.format = navigator.gpu.getPreferredCanvasFormat();

    const module = device.createShaderModule({ code: SHADER_SOURCE });
    this.pipeline = device.createRenderPipeline({
      layout: "auto",
      vertex: { module, entryPoint: "vs_main" },
      fragment: { module, entryPoint: "fs_main", targets: [{ format: this.format }] },
      primitive: { topology: "triangle-list" },
    });
    this.paramsBuffer = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  /** `shape` is NCHW (N=1) or CHW; channels must be 1 or 3. Returns false for
   * unsupported shapes rather than throwing — this is a best-effort visualizer. */
  render(buffer: GPUBuffer, shape: readonly number[]): boolean {
    const dims = shape.length === 4 ? shape.slice(1) : shape;
    if (dims.length !== 3) return false;
    const [channels, height, width] = dims as [number, number, number];
    if (channels !== 1 && channels !== 3) return false;

    if (width !== this.configuredWidth || height !== this.configuredHeight) {
      this.canvas.width = width;
      this.canvas.height = height;
      this.context.configure({ device: this.device, format: this.format, alphaMode: "opaque" });
      this.configuredWidth = width;
      this.configuredHeight = height;
    }

    this.device.queue.writeBuffer(this.paramsBuffer, 0, new Uint32Array([width, height, channels, 0]));

    const bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer } },
        { binding: 1, resource: { buffer: this.paramsBuffer } },
      ],
    });

    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.context.getCurrentTexture().createView(),
          loadOp: "clear",
          storeOp: "store",
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        },
      ],
    });
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(6);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    return true;
  }
}
