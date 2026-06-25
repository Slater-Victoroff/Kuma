import { describe, expect, it } from "vitest";
import type { GraphNode, KumaManifest } from "../src/types/manifest.js";
import { runGraph } from "../src/engine/scheduler.js";
import { findLinearWeightElisions } from "../src/ops/linear.js";
import { KumaUnsupportedOpError } from "../src/errors.js";
import { BUFFER_POOL_DEPTH, createBufferPoolState, type ResolvedTensor } from "../src/engine/context.js";
import { createMockDevice } from "./mock-gpu.js";

describe("findLinearWeightElisions", () => {
  it("elides a transpose whose sole consumer is addmm and whose input is a weight parameter", () => {
    const nodes: GraphNode[] = [
      { id: 0, name: "w", op: "placeholder", target: "w", args: [], kwargs: {}, meta: {}, kind: "parameter", weight_name: "fc.weight" },
      { id: 1, name: "b", op: "placeholder", target: "b", args: [], kwargs: {}, meta: {}, kind: "parameter", weight_name: "fc.bias" },
      { id: 2, name: "x", op: "placeholder", target: "x", args: [], kwargs: {}, meta: {}, kind: "user_input" },
      { id: 3, name: "t", op: "call_function", target: "aten.t.default", args: [{ node_ref: "w" }], kwargs: {}, meta: {} },
      {
        id: 4,
        name: "addmm",
        op: "call_function",
        target: "aten.addmm.default",
        args: [{ node_ref: "b" }, { node_ref: "x" }, { node_ref: "t" }],
        kwargs: {},
        meta: {},
      },
    ];
    expect(findLinearWeightElisions(nodes).get("t")).toBe("w");
  });

  it("does not elide a transpose with more than one consumer", () => {
    const nodes: GraphNode[] = [
      { id: 0, name: "w", op: "placeholder", target: "w", args: [], kwargs: {}, meta: {}, kind: "parameter", weight_name: "fc.weight" },
      { id: 1, name: "x", op: "placeholder", target: "x", args: [], kwargs: {}, meta: {}, kind: "user_input" },
      { id: 2, name: "t", op: "call_function", target: "aten.t.default", args: [{ node_ref: "w" }], kwargs: {}, meta: {} },
      { id: 3, name: "mm1", op: "call_function", target: "aten.mm.default", args: [{ node_ref: "x" }, { node_ref: "t" }], kwargs: {}, meta: {} },
      { id: 4, name: "mm2", op: "call_function", target: "aten.mm.default", args: [{ node_ref: "x" }, { node_ref: "t" }], kwargs: {}, meta: {} },
    ];
    expect(findLinearWeightElisions(nodes).size).toBe(0);
  });

  it("does not elide a transpose whose input is not a literal weight parameter", () => {
    const nodes: GraphNode[] = [
      { id: 0, name: "x", op: "placeholder", target: "x", args: [], kwargs: {}, meta: {}, kind: "user_input" },
      { id: 1, name: "x2", op: "placeholder", target: "x2", args: [], kwargs: {}, meta: {}, kind: "user_input" },
      { id: 2, name: "t", op: "call_function", target: "aten.t.default", args: [{ node_ref: "x" }], kwargs: {}, meta: {} },
      { id: 3, name: "mm", op: "call_function", target: "aten.mm.default", args: [{ node_ref: "x2" }, { node_ref: "t" }], kwargs: {}, meta: {} },
    ];
    expect(findLinearWeightElisions(nodes).size).toBe(0);
  });
});

function buildReluManifest(): KumaManifest {
  return {
    format: "kuma",
    format_version: 0,
    weight_file: "weights.f32.bin",
    endianness: "little",
    inputs: [{ name: "x", shape: [4] }],
    outputs: [{ name: "relu_out", shape: [4] }],
    weights: [],
    graph: {
      node_count: 3,
      op_counts: { "aten.relu.default": 1 },
      nodes: [
        { id: 0, name: "x", op: "placeholder", target: "x", args: [], kwargs: {}, meta: { shape: [4] }, kind: "user_input" },
        { id: 1, name: "relu_out", op: "call_function", target: "aten.relu.default", args: [{ node_ref: "x" }], kwargs: {}, meta: { shape: [4] } },
        { id: 2, name: "output", op: "output", target: "output", args: [[{ node_ref: "relu_out" }]], kwargs: {}, meta: {} },
      ],
    },
    warnings: [],
    unsupported_ops: [],
  };
}

describe("runGraph (mocked GPUDevice — structural checks only, no real compute)", () => {
  it("resolves a placeholder input through a single op to the output", async () => {
    const { device } = createMockDevice();
    const inputBuffer = device.createBuffer({ size: 16, usage: 0 } as GPUBufferDescriptor);

    const outputs = await runGraph({
      device,
      manifest: buildReluManifest(),
      kernels: new Map([["relu.wgsl", "// fake"]]),
      pipelineCache: new Map(),
      weightBuffers: new Map(),
      inputBuffers: new Map([["x", { buffer: inputBuffer, shape: [4] }]]),
    });

    expect(outputs).toHaveLength(1);
    expect(outputs[0]!.name).toBe("relu_out");
    expect(outputs[0]!.shape).toEqual([4]);
    expect(outputs[0]!.data).toHaveLength(4);
  });

  it("throws KumaUnsupportedOpError for a target with no registered kernel", async () => {
    const { device } = createMockDevice();
    const manifest = buildReluManifest();
    manifest.graph.nodes[1]!.target = "aten.native_layer_norm.default";

    await expect(
      runGraph({
        device,
        manifest,
        kernels: new Map(),
        pipelineCache: new Map(),
        weightBuffers: new Map(),
        inputBuffers: new Map([["x", { buffer: device.createBuffer({ size: 16, usage: 0 } as GPUBufferDescriptor), shape: [4] }]]),
      }),
    ).rejects.toThrow(KumaUnsupportedOpError);
  });

  it("resolves a multi-output node (chunk) through getitem, dispatching one slice per piece", async () => {
    const { device, dispatches } = createMockDevice();
    const manifest: KumaManifest = {
      format: "kuma",
      format_version: 0,
      weight_file: "weights.f32.bin",
      endianness: "little",
      inputs: [{ name: "x", shape: [4, 3] }],
      outputs: [
        { name: "piece0", shape: [2, 3] },
        { name: "piece1", shape: [2, 3] },
      ],
      weights: [],
      graph: {
        node_count: 5,
        op_counts: { "aten.chunk.default": 1, getitem: 2 },
        nodes: [
          { id: 0, name: "x", op: "placeholder", target: "x", args: [], kwargs: {}, meta: { shape: [4, 3] }, kind: "user_input" },
          {
            id: 1,
            name: "chunk",
            op: "call_function",
            target: "aten.chunk.default",
            args: [{ node_ref: "x" }, 2, 0],
            kwargs: {},
            meta: { outputs: [{ shape: [2, 3] }, { shape: [2, 3] }] },
          },
          { id: 2, name: "piece0", op: "call_function", target: "getitem", args: [{ node_ref: "chunk" }, 0], kwargs: {}, meta: { shape: [2, 3] } },
          { id: 3, name: "piece1", op: "call_function", target: "getitem", args: [{ node_ref: "chunk" }, 1], kwargs: {}, meta: { shape: [2, 3] } },
          {
            id: 4,
            name: "output",
            op: "output",
            target: "output",
            args: [[{ node_ref: "piece0" }, { node_ref: "piece1" }]],
            kwargs: {},
            meta: {},
          },
        ],
      },
      warnings: [],
      unsupported_ops: [],
    };

    const outputs = await runGraph({
      device,
      manifest,
      kernels: new Map([["slice.wgsl", "// fake"]]),
      pipelineCache: new Map(),
      weightBuffers: new Map(),
      inputBuffers: new Map([["x", { buffer: device.createBuffer({ size: 48, usage: 0 } as GPUBufferDescriptor), shape: [4, 3] }]]),
    });

    expect(outputs.map((o) => o.name)).toEqual(["piece0", "piece1"]);
    expect(outputs[0]!.shape).toEqual([2, 3]);
    expect(outputs[1]!.shape).toEqual([2, 3]);
    // getitem is a pure lookup (no dispatch) — only chunk's 2 slice.wgsl calls dispatch.
    expect(dispatches).toHaveLength(2);
  });

  it("a free passthrough op (alias) dispatches zero kernels", async () => {
    const { device, dispatches } = createMockDevice();
    const manifest: KumaManifest = {
      format: "kuma",
      format_version: 0,
      weight_file: "weights.f32.bin",
      endianness: "little",
      inputs: [{ name: "x", shape: [4] }],
      outputs: [{ name: "aliased", shape: [4] }],
      weights: [],
      graph: {
        node_count: 3,
        op_counts: { "aten.alias.default": 1 },
        nodes: [
          { id: 0, name: "x", op: "placeholder", target: "x", args: [], kwargs: {}, meta: { shape: [4] }, kind: "user_input" },
          { id: 1, name: "aliased", op: "call_function", target: "aten.alias.default", args: [{ node_ref: "x" }], kwargs: {}, meta: { shape: [4] } },
          { id: 2, name: "output", op: "output", target: "output", args: [[{ node_ref: "aliased" }]], kwargs: {}, meta: {} },
        ],
      },
      warnings: [],
      unsupported_ops: [],
    };

    const outputs = await runGraph({
      device,
      manifest,
      kernels: new Map(),
      pipelineCache: new Map(),
      weightBuffers: new Map(),
      inputBuffers: new Map([["x", { buffer: device.createBuffer({ size: 16, usage: 0 } as GPUBufferDescriptor), shape: [4] }]]),
    });

    expect(outputs[0]!.shape).toEqual([4]);
    expect(dispatches).toHaveLength(0);
  });

  it("complex/real pairing round-trips through aten.complex/aten.real with zero dispatch", async () => {
    const { device, dispatches } = createMockDevice();
    const manifest: KumaManifest = {
      format: "kuma",
      format_version: 0,
      weight_file: "weights.f32.bin",
      endianness: "little",
      inputs: [
        { name: "re", shape: [4] },
        { name: "im", shape: [4] },
      ],
      outputs: [{ name: "back_to_real", shape: [4] }],
      weights: [],
      graph: {
        node_count: 4,
        op_counts: { "aten.complex.default": 1, "aten.real.default": 1 },
        nodes: [
          { id: 0, name: "re", op: "placeholder", target: "re", args: [], kwargs: {}, meta: { shape: [4] }, kind: "user_input" },
          { id: 1, name: "im", op: "placeholder", target: "im", args: [], kwargs: {}, meta: { shape: [4] }, kind: "user_input" },
          {
            id: 2,
            name: "z",
            op: "call_function",
            target: "aten.complex.default",
            args: [{ node_ref: "re" }, { node_ref: "im" }],
            kwargs: {},
            meta: { shape: [4], dtype: "torch.complex64" },
          },
          {
            id: 3,
            name: "back_to_real",
            op: "call_function",
            target: "aten.real.default",
            args: [{ node_ref: "z" }],
            kwargs: {},
            meta: { shape: [4] },
          },
          { id: 4, name: "output", op: "output", target: "output", args: [[{ node_ref: "back_to_real" }]], kwargs: {}, meta: {} },
        ],
      },
      warnings: [],
      unsupported_ops: [],
    };

    const outputs = await runGraph({
      device,
      manifest,
      kernels: new Map(),
      pipelineCache: new Map(),
      weightBuffers: new Map(),
      inputBuffers: new Map([
        ["re", { buffer: device.createBuffer({ size: 16, usage: 0 } as GPUBufferDescriptor), shape: [4] }],
        ["im", { buffer: device.createBuffer({ size: 16, usage: 0 } as GPUBufferDescriptor), shape: [4] }],
      ]),
    });

    expect(outputs[0]!.shape).toEqual([4]);
    expect(dispatches).toHaveLength(0);
  });

  function buildComplexBinaryManifest(opTarget: string, kernelName: string): KumaManifest {
    return {
      format: "kuma",
      format_version: 0,
      weight_file: "weights.f32.bin",
      endianness: "little",
      inputs: [
        { name: "re", shape: [4] },
        { name: "im", shape: [4] },
        { name: "r", shape: [4] },
      ],
      outputs: [{ name: "out_imag", shape: [4] }],
      weights: [],
      graph: {
        node_count: 5,
        op_counts: {},
        nodes: [
          { id: 0, name: "re", op: "placeholder", target: "re", args: [], kwargs: {}, meta: { shape: [4] }, kind: "user_input" },
          { id: 1, name: "im", op: "placeholder", target: "im", args: [], kwargs: {}, meta: { shape: [4] }, kind: "user_input" },
          { id: 2, name: "r", op: "placeholder", target: "r", args: [], kwargs: {}, meta: { shape: [4] }, kind: "user_input" },
          { id: 3, name: "z", op: "call_function", target: "aten.complex.default", args: [{ node_ref: "re" }, { node_ref: "im" }], kwargs: {}, meta: { shape: [4] } },
          { id: 4, name: "combined", op: "call_function", target: opTarget, args: [{ node_ref: "z" }, { node_ref: "r" }], kwargs: {}, meta: { shape: [4] } },
          { id: 5, name: "out_imag", op: "call_function", target: "aten.imag.default", args: [{ node_ref: "combined" }], kwargs: {}, meta: { shape: [4] } },
          { id: 6, name: "output", op: "output", target: "output", args: [[{ node_ref: "out_imag" }]], kwargs: {}, meta: {} },
        ],
      },
      warnings: [],
      unsupported_ops: [],
    };
  }

  function complexInputBuffers(device: GPUDevice): Map<string, ResolvedTensor> {
    return new Map([
      ["re", { buffer: device.createBuffer({ size: 16, usage: 0 } as GPUBufferDescriptor), shape: [4] }],
      ["im", { buffer: device.createBuffer({ size: 16, usage: 0 } as GPUBufferDescriptor), shape: [4] }],
      ["r", { buffer: device.createBuffer({ size: 16, usage: 0 } as GPUBufferDescriptor), shape: [4] }],
    ]);
  }

  it("aten.mul.Tensor(complex, real) propagates the imaginary part (2 mul.wgsl dispatches)", async () => {
    const { device, dispatches } = createMockDevice();
    const outputs = await runGraph({
      device,
      manifest: buildComplexBinaryManifest("aten.mul.Tensor", "mul.wgsl"),
      kernels: new Map([["mul.wgsl", "// fake"]]),
      pipelineCache: new Map(),
      weightBuffers: new Map(),
      inputBuffers: complexInputBuffers(device),
    });
    expect(outputs[0]!.shape).toEqual([4]);
    expect(dispatches).toHaveLength(2); // real mul + imag mul
  });

  it("aten.add.Tensor(complex, real) propagates the imaginary part with zero extra dispatch", async () => {
    const { device, dispatches } = createMockDevice();
    const outputs = await runGraph({
      device,
      manifest: buildComplexBinaryManifest("aten.add.Tensor", "add.wgsl"),
      kernels: new Map([["add.wgsl", "// fake"]]),
      pipelineCache: new Map(),
      weightBuffers: new Map(),
      inputBuffers: complexInputBuffers(device),
    });
    expect(outputs[0]!.shape).toEqual([4]);
    // 1 dispatch for the real part; out_im = a_im + 0 reuses a's own imag buffer directly.
    expect(dispatches).toHaveLength(1);
  });

  it("aten.div.Tensor(complex, real) propagates the imaginary part (1 extra div.wgsl)", async () => {
    const { device, dispatches } = createMockDevice();
    const outputs = await runGraph({
      device,
      manifest: buildComplexBinaryManifest("aten.div.Tensor", "div.wgsl"),
      kernels: new Map([["div.wgsl", "// fake"]]),
      pipelineCache: new Map(),
      weightBuffers: new Map(),
      inputBuffers: complexInputBuffers(device),
    });
    expect(outputs[0]!.shape).toEqual([4]);
    expect(dispatches).toHaveLength(2); // real div + imag div
  });

  it("aten.div.Tensor with a complex denominator fails loudly instead of silently dropping it", async () => {
    const { device } = createMockDevice();
    const manifest: KumaManifest = {
      format: "kuma",
      format_version: 0,
      weight_file: "weights.f32.bin",
      endianness: "little",
      inputs: [
        { name: "re", shape: [4] },
        { name: "im", shape: [4] },
      ],
      outputs: [{ name: "out", shape: [4] }],
      weights: [],
      graph: {
        node_count: 4,
        op_counts: {},
        nodes: [
          { id: 0, name: "re", op: "placeholder", target: "re", args: [], kwargs: {}, meta: { shape: [4] }, kind: "user_input" },
          { id: 1, name: "im", op: "placeholder", target: "im", args: [], kwargs: {}, meta: { shape: [4] }, kind: "user_input" },
          { id: 2, name: "z", op: "call_function", target: "aten.complex.default", args: [{ node_ref: "re" }, { node_ref: "im" }], kwargs: {}, meta: { shape: [4] } },
          { id: 3, name: "out", op: "call_function", target: "aten.div.Tensor", args: [{ node_ref: "z" }, { node_ref: "z" }], kwargs: {}, meta: { shape: [4] } },
          { id: 4, name: "output", op: "output", target: "output", args: [[{ node_ref: "out" }]], kwargs: {}, meta: {} },
        ],
      },
      warnings: [],
      unsupported_ops: [],
    };

    await expect(
      runGraph({
        device,
        manifest,
        kernels: new Map([["div.wgsl", "// fake"]]),
        pipelineCache: new Map(),
        weightBuffers: new Map(),
        inputBuffers: new Map([
          ["re", { buffer: device.createBuffer({ size: 16, usage: 0 } as GPUBufferDescriptor), shape: [4] }],
          ["im", { buffer: device.createBuffer({ size: 16, usage: 0 } as GPUBufferDescriptor), shape: [4] }],
        ]),
      }),
    ).rejects.toThrow(KumaUnsupportedOpError);
  });

  it("aten.rsub.Scalar negates the imaginary part of a complex input", async () => {
    const { device, dispatches } = createMockDevice();
    const manifest: KumaManifest = {
      format: "kuma",
      format_version: 0,
      weight_file: "weights.f32.bin",
      endianness: "little",
      inputs: [
        { name: "re", shape: [4] },
        { name: "im", shape: [4] },
      ],
      outputs: [{ name: "out_imag", shape: [4] }],
      weights: [],
      graph: {
        node_count: 5,
        op_counts: {},
        nodes: [
          { id: 0, name: "re", op: "placeholder", target: "re", args: [], kwargs: {}, meta: { shape: [4] }, kind: "user_input" },
          { id: 1, name: "im", op: "placeholder", target: "im", args: [], kwargs: {}, meta: { shape: [4] }, kind: "user_input" },
          { id: 2, name: "z", op: "call_function", target: "aten.complex.default", args: [{ node_ref: "re" }, { node_ref: "im" }], kwargs: {}, meta: { shape: [4] } },
          { id: 3, name: "rs", op: "call_function", target: "aten.rsub.Scalar", args: [{ node_ref: "z" }, 1.0], kwargs: {}, meta: { shape: [4] } },
          { id: 4, name: "out_imag", op: "call_function", target: "aten.imag.default", args: [{ node_ref: "rs" }], kwargs: {}, meta: { shape: [4] } },
          { id: 5, name: "output", op: "output", target: "output", args: [[{ node_ref: "out_imag" }]], kwargs: {}, meta: {} },
        ],
      },
      warnings: [],
      unsupported_ops: [],
    };

    const outputs = await runGraph({
      device,
      manifest,
      kernels: new Map([
        ["sub.wgsl", "// fake"],
        ["neg.wgsl", "// fake"],
      ]),
      pipelineCache: new Map(),
      weightBuffers: new Map(),
      inputBuffers: new Map([
        ["re", { buffer: device.createBuffer({ size: 16, usage: 0 } as GPUBufferDescriptor), shape: [4] }],
        ["im", { buffer: device.createBuffer({ size: 16, usage: 0 } as GPUBufferDescriptor), shape: [4] }],
      ]),
    });
    expect(outputs[0]!.shape).toEqual([4]);
    expect(dispatches).toHaveLength(2); // sub (real part) + neg (imag part)
  });

  it("stack of 3 tensors dispatches 2 concat.wgsl calls (N-1 chain, via free unsqueeze)", async () => {
    const { device, dispatches } = createMockDevice();
    const manifest: KumaManifest = {
      format: "kuma",
      format_version: 0,
      weight_file: "weights.f32.bin",
      endianness: "little",
      inputs: [
        { name: "a", shape: [2, 3] },
        { name: "b", shape: [2, 3] },
        { name: "c", shape: [2, 3] },
      ],
      outputs: [{ name: "stacked", shape: [3, 2, 3] }],
      weights: [],
      graph: {
        node_count: 5,
        op_counts: { "aten.stack.default": 1 },
        nodes: [
          { id: 0, name: "a", op: "placeholder", target: "a", args: [], kwargs: {}, meta: { shape: [2, 3] }, kind: "user_input" },
          { id: 1, name: "b", op: "placeholder", target: "b", args: [], kwargs: {}, meta: { shape: [2, 3] }, kind: "user_input" },
          { id: 2, name: "c", op: "placeholder", target: "c", args: [], kwargs: {}, meta: { shape: [2, 3] }, kind: "user_input" },
          {
            id: 3,
            name: "stacked",
            op: "call_function",
            target: "aten.stack.default",
            args: [[{ node_ref: "a" }, { node_ref: "b" }, { node_ref: "c" }], 0],
            kwargs: {},
            meta: { shape: [3, 2, 3] },
          },
          { id: 4, name: "output", op: "output", target: "output", args: [[{ node_ref: "stacked" }]], kwargs: {}, meta: {} },
        ],
      },
      warnings: [],
      unsupported_ops: [],
    };

    const outputs = await runGraph({
      device,
      manifest,
      kernels: new Map([["concat.wgsl", "// fake"]]),
      pipelineCache: new Map(),
      weightBuffers: new Map(),
      inputBuffers: new Map([
        ["a", { buffer: device.createBuffer({ size: 24, usage: 0 } as GPUBufferDescriptor), shape: [2, 3] }],
        ["b", { buffer: device.createBuffer({ size: 24, usage: 0 } as GPUBufferDescriptor), shape: [2, 3] }],
        ["c", { buffer: device.createBuffer({ size: 24, usage: 0 } as GPUBufferDescriptor), shape: [2, 3] }],
      ]),
    });

    expect(outputs[0]!.shape).toEqual([3, 2, 3]);
    expect(dispatches).toHaveLength(2);
  });

  function buildDivModeManifest(mode: string): KumaManifest {
    return {
      format: "kuma",
      format_version: 0,
      weight_file: "weights.f32.bin",
      endianness: "little",
      inputs: [
        { name: "a", shape: [4] },
        { name: "b", shape: [4] },
      ],
      outputs: [{ name: "divided", shape: [4] }],
      weights: [],
      graph: {
        node_count: 3,
        op_counts: { "aten.div.Tensor_mode": 1 },
        nodes: [
          { id: 0, name: "a", op: "placeholder", target: "a", args: [], kwargs: {}, meta: { shape: [4] }, kind: "user_input" },
          { id: 1, name: "b", op: "placeholder", target: "b", args: [], kwargs: {}, meta: { shape: [4] }, kind: "user_input" },
          {
            id: 2,
            name: "divided",
            op: "call_function",
            target: "aten.div.Tensor_mode",
            args: [{ node_ref: "a" }, { node_ref: "b" }, mode],
            kwargs: {},
            meta: { shape: [4] },
          },
          { id: 3, name: "output", op: "output", target: "output", args: [[{ node_ref: "divided" }]], kwargs: {}, meta: {} },
        ],
      },
      warnings: [],
      unsupported_ops: [],
    };
  }

  it('div_mode="floor" dispatches div.wgsl then floor.wgsl', async () => {
    const { device, dispatches } = createMockDevice();
    const inputBuffers = new Map([
      ["a", { buffer: device.createBuffer({ size: 16, usage: 0 } as GPUBufferDescriptor), shape: [4] }],
      ["b", { buffer: device.createBuffer({ size: 16, usage: 0 } as GPUBufferDescriptor), shape: [4] }],
    ]);

    const outputs = await runGraph({
      device,
      manifest: buildDivModeManifest("floor"),
      kernels: new Map([
        ["div.wgsl", "// fake"],
        ["floor.wgsl", "// fake"],
      ]),
      pipelineCache: new Map(),
      weightBuffers: new Map(),
      inputBuffers,
    });

    expect(outputs[0]!.shape).toEqual([4]);
    expect(dispatches).toHaveLength(2);
  });

  it('div_mode="trunc" (unsupported) fails loudly instead of guessing', async () => {
    const { device } = createMockDevice();
    const inputBuffers = new Map([
      ["a", { buffer: device.createBuffer({ size: 16, usage: 0 } as GPUBufferDescriptor), shape: [4] }],
      ["b", { buffer: device.createBuffer({ size: 16, usage: 0 } as GPUBufferDescriptor), shape: [4] }],
    ]);

    await expect(
      runGraph({
        device,
        manifest: buildDivModeManifest("trunc"),
        kernels: new Map(),
        pipelineCache: new Map(),
        weightBuffers: new Map(),
        inputBuffers,
      }),
    ).rejects.toThrow(KumaUnsupportedOpError);
  });

  function buildSwitchManifest(): KumaManifest {
    return {
      format: "kuma",
      format_version: 0,
      weight_file: "weights.f32.bin",
      endianness: "little",
      inputs: [{ name: "norm_t", shape: [1] }],
      outputs: [{ name: "result", shape: [1] }],
      weights: [],
      graph: {
        node_count: 6,
        op_counts: {},
        nodes: [
          { id: 0, name: "norm_t", op: "placeholder", target: "norm_t", args: [], kwargs: {}, meta: { shape: [1] }, kind: "user_input" },
          {
            id: 1,
            name: "route",
            op: "js_snippet",
            target: "route.js",
            args: [{ node_ref: "norm_t" }],
            kwargs: {},
            meta: { outputs: [{ shape: [1] }, { shape: [1] }] },
          },
          { id: 2, name: "seg_id", op: "call_function", target: "getitem", args: [{ node_ref: "route" }, 0], kwargs: {}, meta: {} },
          { id: 3, name: "local_val", op: "call_function", target: "getitem", args: [{ node_ref: "route" }, 1], kwargs: {}, meta: {} },
          {
            id: 4,
            name: "switched",
            op: "switch",
            target: "switch",
            args: [{ node_ref: "local_val" }],
            kwargs: {},
            meta: { shape: [1] },
            selector: { node_ref: "seg_id" },
            branches: [
              {
                inputs: [{ node_ref: "b0_input" }],
                output: { node_ref: "b0_out" },
                nodes: [
                  { id: 0, name: "b0_input", op: "placeholder", target: "b0_input", args: [], kwargs: {}, meta: { shape: [1] }, kind: "user_input" },
                  { id: 1, name: "b0_out", op: "call_function", target: "aten.relu.default", args: [{ node_ref: "b0_input" }], kwargs: {}, meta: { shape: [1] } },
                ],
              },
              {
                inputs: [{ node_ref: "b1_input" }],
                output: { node_ref: "b1_out" },
                nodes: [
                  { id: 0, name: "b1_input", op: "placeholder", target: "b1_input", args: [], kwargs: {}, meta: { shape: [1] }, kind: "user_input" },
                  { id: 1, name: "b1_out", op: "call_function", target: "aten.gelu.default", args: [{ node_ref: "b1_input" }], kwargs: {}, meta: { shape: [1] } },
                ],
              },
            ],
          },
          { id: 5, name: "output", op: "output", target: "output", args: [[{ node_ref: "switched" }]], kwargs: {}, meta: {} },
        ],
      },
      warnings: [],
      unsupported_ops: [],
    };
  }

  const ROUTE_SNIPPET = "function main(inputs) { const normT = inputs[0]; const segId = normT[0] < 0.5 ? 0 : 1; return [[segId], normT]; }";

  it("js_snippet -> getitem -> switch picks exactly one branch (segment 0), dispatching only its kernel", async () => {
    const { device, dispatches } = createMockDevice();
    const normT = new Float32Array([0.2]); // routes to branch 0 per ROUTE_SNIPPET

    const outputs = await runGraph({
      device,
      manifest: buildSwitchManifest(),
      kernels: new Map([
        ["relu.wgsl", "// fake"],
        ["gelu.wgsl", "// fake"],
      ]),
      pipelineCache: new Map(),
      weightBuffers: new Map(),
      inputBuffers: new Map([["norm_t", { buffer: device.createBuffer({ size: 4, usage: 0 } as GPUBufferDescriptor), shape: [1] }]]),
      rawInputs: new Map([["norm_t", normT]]),
      snippets: new Map([["route.js", ROUTE_SNIPPET]]),
      snippetCache: new Map(),
    });

    expect(outputs[0]!.shape).toEqual([1]);
    // Exactly one kernel ran (branch 0's relu) -- branch 1's gelu never dispatched at all.
    expect(dispatches).toHaveLength(1);
  });

  it("...and the other branch (segment 1) when the snippet routes differently", async () => {
    const { device, dispatches } = createMockDevice();
    const normT = new Float32Array([0.8]); // routes to branch 1 per ROUTE_SNIPPET

    await runGraph({
      device,
      manifest: buildSwitchManifest(),
      kernels: new Map([
        ["relu.wgsl", "// fake"],
        ["gelu.wgsl", "// fake"],
      ]),
      pipelineCache: new Map(),
      weightBuffers: new Map(),
      inputBuffers: new Map([["norm_t", { buffer: device.createBuffer({ size: 4, usage: 0 } as GPUBufferDescriptor), shape: [1] }]]),
      rawInputs: new Map([["norm_t", normT]]),
      snippets: new Map([["route.js", ROUTE_SNIPPET]]),
      snippetCache: new Map(),
    });

    expect(dispatches).toHaveLength(1);
  });

  it("aten.mul.Tensor tolerates a literal scalar second operand (not just tensor refs)", async () => {
    const { device, dispatches } = createMockDevice();
    const manifest: KumaManifest = {
      format: "kuma",
      format_version: 0,
      weight_file: "weights.f32.bin",
      endianness: "little",
      inputs: [{ name: "x", shape: [4] }],
      outputs: [{ name: "scaled", shape: [4] }],
      weights: [],
      graph: {
        node_count: 3,
        op_counts: { "aten.mul.Tensor": 1 },
        nodes: [
          { id: 0, name: "x", op: "placeholder", target: "x", args: [], kwargs: {}, meta: { shape: [4] }, kind: "user_input" },
          { id: 1, name: "scaled", op: "call_function", target: "aten.mul.Tensor", args: [{ node_ref: "x" }, 32], kwargs: {}, meta: { shape: [4] } },
          { id: 2, name: "output", op: "output", target: "output", args: [[{ node_ref: "scaled" }]], kwargs: {}, meta: {} },
        ],
      },
      warnings: [],
      unsupported_ops: [],
    };

    const outputs = await runGraph({
      device,
      manifest,
      kernels: new Map([["mul.wgsl", "// fake"]]),
      pipelineCache: new Map(),
      weightBuffers: new Map(),
      inputBuffers: new Map([["x", { buffer: device.createBuffer({ size: 16, usage: 0 } as GPUBufferDescriptor), shape: [4] }]]),
    });

    expect(outputs[0]!.shape).toEqual([4]);
    expect(dispatches).toHaveLength(1);
  });

  it("aten.rsub.Scalar (scalar - tensor) dispatches sub.wgsl with a broadcast constant", async () => {
    const { device, dispatches } = createMockDevice();
    const manifest: KumaManifest = {
      format: "kuma",
      format_version: 0,
      weight_file: "weights.f32.bin",
      endianness: "little",
      inputs: [{ name: "x", shape: [4] }],
      outputs: [{ name: "result", shape: [4] }],
      weights: [],
      graph: {
        node_count: 3,
        op_counts: { "aten.rsub.Scalar": 1 },
        nodes: [
          { id: 0, name: "x", op: "placeholder", target: "x", args: [], kwargs: {}, meta: { shape: [4] }, kind: "user_input" },
          { id: 1, name: "result", op: "call_function", target: "aten.rsub.Scalar", args: [{ node_ref: "x" }, 1.0], kwargs: {}, meta: { shape: [4] } },
          { id: 2, name: "output", op: "output", target: "output", args: [[{ node_ref: "result" }]], kwargs: {}, meta: {} },
        ],
      },
      warnings: [],
      unsupported_ops: [],
    };

    const outputs = await runGraph({
      device,
      manifest,
      kernels: new Map([["sub.wgsl", "// fake"]]),
      pipelineCache: new Map(),
      weightBuffers: new Map(),
      inputBuffers: new Map([["x", { buffer: device.createBuffer({ size: 16, usage: 0 } as GPUBufferDescriptor), shape: [4] }]]),
    });

    expect(outputs[0]!.shape).toEqual([4]);
    expect(dispatches).toHaveLength(1);
  });

  it("aten.rsub.Scalar with alpha != 1 fails loudly instead of computing the wrong thing", async () => {
    const { device } = createMockDevice();
    const manifest: KumaManifest = {
      format: "kuma",
      format_version: 0,
      weight_file: "weights.f32.bin",
      endianness: "little",
      inputs: [{ name: "x", shape: [4] }],
      outputs: [{ name: "result", shape: [4] }],
      weights: [],
      graph: {
        node_count: 3,
        op_counts: {},
        nodes: [
          { id: 0, name: "x", op: "placeholder", target: "x", args: [], kwargs: {}, meta: { shape: [4] }, kind: "user_input" },
          { id: 1, name: "result", op: "call_function", target: "aten.rsub.Scalar", args: [{ node_ref: "x" }, 1.0, 2.0], kwargs: {}, meta: { shape: [4] } },
          { id: 2, name: "output", op: "output", target: "output", args: [[{ node_ref: "result" }]], kwargs: {}, meta: {} },
        ],
      },
      warnings: [],
      unsupported_ops: [],
    };

    await expect(
      runGraph({
        device,
        manifest,
        kernels: new Map(),
        pipelineCache: new Map(),
        weightBuffers: new Map(),
        inputBuffers: new Map([["x", { buffer: device.createBuffer({ size: 16, usage: 0 } as GPUBufferDescriptor), shape: [4] }]]),
      }),
    ).rejects.toThrow(KumaUnsupportedOpError);
  });

  it("aten.conv2d.default with the bare 3-arg form (stride/padding/dilation/groups omitted at their defaults)", async () => {
    const { device, dispatches } = createMockDevice();
    const manifest: KumaManifest = {
      format: "kuma",
      format_version: 0,
      weight_file: "weights.f32.bin",
      endianness: "little",
      inputs: [{ name: "x", shape: [1, 3, 4, 4] }],
      outputs: [{ name: "y", shape: [1, 2, 4, 4] }],
      weights: [
        { name: "w", shape: [2, 3, 3, 3], dtype: "float32", n_elements: 54, byte_offset: 0, byte_length: 216 },
        { name: "b", shape: [2], dtype: "float32", n_elements: 2, byte_offset: 216, byte_length: 8 },
      ],
      graph: {
        node_count: 5,
        op_counts: {},
        nodes: [
          { id: 0, name: "x", op: "placeholder", target: "x", args: [], kwargs: {}, meta: { shape: [1, 3, 4, 4] }, kind: "user_input" },
          { id: 1, name: "w", op: "placeholder", target: "w", args: [], kwargs: {}, meta: {}, kind: "parameter", weight_name: "w" },
          { id: 2, name: "b", op: "placeholder", target: "b", args: [], kwargs: {}, meta: {}, kind: "parameter", weight_name: "b" },
          {
            id: 3,
            name: "y",
            op: "call_function",
            target: "aten.conv2d.default",
            // bare 3-arg form, exactly as observed in the real exported model
            args: [{ node_ref: "x" }, { node_ref: "w" }, { node_ref: "b" }],
            kwargs: {},
            meta: { shape: [1, 2, 4, 4] },
          },
          { id: 4, name: "output", op: "output", target: "output", args: [[{ node_ref: "y" }]], kwargs: {}, meta: {} },
        ],
      },
      warnings: [],
      unsupported_ops: [],
    };

    const outputs = await runGraph({
      device,
      manifest,
      kernels: new Map([["conv2d.wgsl", "// fake"]]),
      pipelineCache: new Map(),
      weightBuffers: new Map([
        ["w", { buffer: device.createBuffer({ size: 216, usage: 0 } as GPUBufferDescriptor), shape: [2, 3, 3, 3] }],
        ["b", { buffer: device.createBuffer({ size: 8, usage: 0 } as GPUBufferDescriptor), shape: [2] }],
      ]),
      inputBuffers: new Map([["x", { buffer: device.createBuffer({ size: 192, usage: 0 } as GPUBufferDescriptor), shape: [1, 3, 4, 4] }]]),
    });

    expect(outputs[0]!.shape).toEqual([1, 2, 4, 4]);
    expect(dispatches).toHaveLength(1);
  });

  it("aten.group_norm.default with the 4-arg form (eps omitted at its default)", async () => {
    const { device, dispatches } = createMockDevice();
    const manifest: KumaManifest = {
      format: "kuma",
      format_version: 0,
      weight_file: "weights.f32.bin",
      endianness: "little",
      inputs: [{ name: "x", shape: [1, 4, 2, 2] }],
      outputs: [{ name: "y", shape: [1, 4, 2, 2] }],
      weights: [
        { name: "w", shape: [4], dtype: "float32", n_elements: 4, byte_offset: 0, byte_length: 16 },
        { name: "b", shape: [4], dtype: "float32", n_elements: 4, byte_offset: 16, byte_length: 16 },
      ],
      graph: {
        node_count: 5,
        op_counts: {},
        nodes: [
          { id: 0, name: "x", op: "placeholder", target: "x", args: [], kwargs: {}, meta: { shape: [1, 4, 2, 2] }, kind: "user_input" },
          { id: 1, name: "w", op: "placeholder", target: "w", args: [], kwargs: {}, meta: {}, kind: "parameter", weight_name: "w" },
          { id: 2, name: "b", op: "placeholder", target: "b", args: [], kwargs: {}, meta: {}, kind: "parameter", weight_name: "b" },
          {
            id: 3,
            name: "y",
            op: "call_function",
            target: "aten.group_norm.default",
            // 4-arg form, exactly as observed in the real exported model (eps omitted)
            args: [{ node_ref: "x" }, 2, { node_ref: "w" }, { node_ref: "b" }],
            kwargs: {},
            meta: { shape: [1, 4, 2, 2] },
          },
          { id: 4, name: "output", op: "output", target: "output", args: [[{ node_ref: "y" }]], kwargs: {}, meta: {} },
        ],
      },
      warnings: [],
      unsupported_ops: [],
    };

    const outputs = await runGraph({
      device,
      manifest,
      kernels: new Map([["groupnorm.wgsl", "// fake"]]),
      pipelineCache: new Map(),
      weightBuffers: new Map([
        ["w", { buffer: device.createBuffer({ size: 16, usage: 0 } as GPUBufferDescriptor), shape: [4] }],
        ["b", { buffer: device.createBuffer({ size: 16, usage: 0 } as GPUBufferDescriptor), shape: [4] }],
      ]),
      inputBuffers: new Map([["x", { buffer: device.createBuffer({ size: 64, usage: 0 } as GPUBufferDescriptor), shape: [1, 4, 2, 2] }]]),
    });

    expect(outputs[0]!.shape).toEqual([1, 4, 2, 2]);
    expect(dispatches).toHaveLength(1);
  });

  it("aten.linear.default applied channel-last to a 4D input (e.g. nn.Linear over [B,H,W,C])", async () => {
    const { device, dispatches } = createMockDevice();
    const manifest: KumaManifest = {
      format: "kuma",
      format_version: 0,
      weight_file: "weights.f32.bin",
      endianness: "little",
      inputs: [{ name: "x", shape: [1, 3, 2, 4] }],
      outputs: [{ name: "y", shape: [1, 3, 2, 5] }],
      weights: [
        { name: "w", shape: [5, 4], dtype: "float32", n_elements: 20, byte_offset: 0, byte_length: 80 },
        { name: "b", shape: [5], dtype: "float32", n_elements: 5, byte_offset: 80, byte_length: 20 },
      ],
      graph: {
        node_count: 5,
        op_counts: {},
        nodes: [
          { id: 0, name: "x", op: "placeholder", target: "x", args: [], kwargs: {}, meta: { shape: [1, 3, 2, 4] }, kind: "user_input" },
          { id: 1, name: "w", op: "placeholder", target: "w", args: [], kwargs: {}, meta: {}, kind: "parameter", weight_name: "w" },
          { id: 2, name: "b", op: "placeholder", target: "b", args: [], kwargs: {}, meta: {}, kind: "parameter", weight_name: "b" },
          {
            id: 3,
            name: "y",
            op: "call_function",
            target: "aten.linear.default",
            // x is 4D (channel-last); only the last dim (4) contracts against weight's K.
            args: [{ node_ref: "x" }, { node_ref: "w" }, { node_ref: "b" }],
            kwargs: {},
            meta: { shape: [1, 3, 2, 5] },
          },
          { id: 4, name: "output", op: "output", target: "output", args: [[{ node_ref: "y" }]], kwargs: {}, meta: {} },
        ],
      },
      warnings: [],
      unsupported_ops: [],
    };

    const outputs = await runGraph({
      device,
      manifest,
      kernels: new Map([["linear.wgsl", "// fake"]]),
      pipelineCache: new Map(),
      weightBuffers: new Map([
        ["w", { buffer: device.createBuffer({ size: 80, usage: 0 } as GPUBufferDescriptor), shape: [5, 4] }],
        ["b", { buffer: device.createBuffer({ size: 20, usage: 0 } as GPUBufferDescriptor), shape: [5] }],
      ]),
      inputBuffers: new Map([["x", { buffer: device.createBuffer({ size: 96, usage: 0 } as GPUBufferDescriptor), shape: [1, 3, 2, 4] }]]),
    });

    // leading dims [1,3,2] preserved, only the last dim (4 -> 5) changes.
    expect(outputs[0]!.shape).toEqual([1, 3, 2, 5]);
    expect(dispatches).toHaveLength(1);
  });
});

describe("runGraph buffer pooling (BufferPoolState — see engine/context.ts)", () => {
  function inputBuffersFor(device: GPUDevice): Map<string, ResolvedTensor> {
    return new Map([["x", { buffer: device.createBuffer({ size: 16, usage: 0 } as GPUBufferDescriptor), shape: [4] }]]);
  }

  it("without a shared bufferPool (the default), repeated calls never alias each other's buffers", async () => {
    const { device } = createMockDevice();
    const manifest = buildReluManifest();
    const params = {
      device,
      manifest,
      kernels: new Map([["relu.wgsl", "// fake"]]),
      pipelineCache: new Map(),
      weightBuffers: new Map(),
      inputBuffers: inputBuffersFor(device),
    };

    const first = await runGraph(params);
    const second = await runGraph(params);

    expect(first[0]!.buffer).not.toBe(second[0]!.buffer);
  });

  it("with a shared bufferPool, a node's output buffer is reused exactly every BUFFER_POOL_DEPTH calls", async () => {
    const { device } = createMockDevice();
    const manifest = buildReluManifest();
    const bufferPool = createBufferPoolState();
    const params = {
      device,
      manifest,
      kernels: new Map([["relu.wgsl", "// fake"]]),
      pipelineCache: new Map(),
      weightBuffers: new Map(),
      inputBuffers: inputBuffersFor(device),
      bufferPool,
    };

    // Sequential (awaited one at a time) calls, BUFFER_POOL_DEPTH + 2 of them, so the
    // ring wraps around at least once.
    const buffers: GPUBuffer[] = [];
    for (let i = 0; i < BUFFER_POOL_DEPTH + 2; i++) {
      const outputs = await runGraph(params);
      buffers.push(outputs[0]!.buffer);
    }

    // Generation N and generation N + BUFFER_POOL_DEPTH land on the same ring slot.
    expect(buffers[BUFFER_POOL_DEPTH]).toBe(buffers[0]);
    expect(buffers[BUFFER_POOL_DEPTH + 1]).toBe(buffers[1]);
    // Every other pair within one rotation is a genuinely distinct buffer.
    for (let i = 0; i < BUFFER_POOL_DEPTH; i++) {
      for (let j = i + 1; j < BUFFER_POOL_DEPTH; j++) {
        expect(buffers[i]).not.toBe(buffers[j]);
      }
    }
  });

  it("pooled buffers are never destroyed by runGraph's own end-of-call cleanup", async () => {
    const { device, destroyedBuffers } = createMockDevice();
    const manifest = buildReluManifest();
    const bufferPool = createBufferPoolState();
    const params = {
      device,
      manifest,
      kernels: new Map([["relu.wgsl", "// fake"]]),
      pipelineCache: new Map(),
      weightBuffers: new Map(),
      inputBuffers: inputBuffersFor(device),
      bufferPool,
    };

    const seenBuffers = new Set<GPUBuffer>();
    for (let i = 0; i < BUFFER_POOL_DEPTH + 2; i++) {
      const outputs = await runGraph(params);
      seenBuffers.add(outputs[0]!.buffer);
    }

    for (const buffer of seenBuffers) {
      expect(destroyedBuffers.has(buffer as unknown as object)).toBe(false);
    }
  });

  it("concurrent (not individually-awaited) calls sharing a pool never lose or double-assign a generation", async () => {
    // Mirrors how demo/main.ts actually drives this: multiple runToGpu() calls fired
    // without awaiting each one first, relying on acquireGenerationSlot/
    // releaseGenerationSlot (engine/context.ts) to keep concurrent calls safe. This is
    // the scenario where a subtle off-by-one or a missed atomicity guarantee in the
    // generation-counter scheme would actually show up -- a purely sequential,
    // one-call-at-a-time test (the test above) can't exercise this at all, since by
    // the time call N starts, call N-1's pool slot has always long since been
    // confirmed free.
    //
    // Deliberately *not* assuming array index == assigned generation order below: when
    // every call is fired in the same synchronous loop before any of them has reached
    // its own await, which one's continuation actually runs first is a microtask-
    // scheduling detail, not something this test can or should pin down. The
    // invariants checked here hold regardless of that order.
    const { device } = createMockDevice();
    const manifest = buildReluManifest();
    const bufferPool = createBufferPoolState();
    const callCount = BUFFER_POOL_DEPTH + 2;

    const results = await Promise.all(
      Array.from({ length: callCount }, () =>
        runGraph({
          device,
          manifest,
          kernels: new Map([["relu.wgsl", "// fake"]]),
          pipelineCache: new Map(),
          weightBuffers: new Map(),
          inputBuffers: inputBuffersFor(device),
          bufferPool,
        }),
      ),
    );

    // Every call must have advanced callGeneration exactly once -- a lost update
    // (two concurrent calls both reading the same stale value) would leave this short
    // of callCount; a double-increment would overshoot it.
    expect(bufferPool.callGeneration).toBe(callCount);

    // The pool must have rotated through exactly BUFFER_POOL_DEPTH distinct buffers
    // across all callCount calls -- not fewer (which would mean two different
    // generations incorrectly collided on one slot) and not more (which would mean
    // the ring didn't actually wrap around / pool to begin with).
    const distinctBuffers = new Set(results.map((outputs) => outputs[0]!.buffer));
    expect(distinctBuffers.size).toBe(Math.min(callCount, BUFFER_POOL_DEPTH));
  });
});
