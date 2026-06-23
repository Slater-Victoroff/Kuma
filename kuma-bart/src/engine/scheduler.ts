import { isNodeRef, type ArgValue, type GraphNode, type KumaManifest, type NodeRef } from "../types/manifest.js";
import { OpContext, type ResolvedTensor } from "./context.js";
import { numElements } from "./shape.js";
import { opRegistry, findLinearWeightElisions } from "../ops/index.js";
import { KumaManifestError, KumaUnsupportedOpError } from "../errors.js";
import { readBuffers, uploadFloat32 } from "../gpu/buffers.js";
import { getSnippetFn, type SnippetFn } from "./snippets.js";

export interface RunGraphParams {
  device: GPUDevice;
  manifest: KumaManifest;
  kernels: ReadonlyMap<string, string>;
  pipelineCache: Map<string, GPUComputePipeline>;
  weightBuffers: ReadonlyMap<string, ResolvedTensor>;
  inputBuffers: ReadonlyMap<string, ResolvedTensor>;
  /** Raw (pre-upload) input arrays, keyed the same way as `inputBuffers` — what
   * `js_snippet` nodes evaluate against, since they only ever depend on top-level
   * model inputs (never a GPU-computed intermediate; see engine/snippets.ts). Optional:
   * only needed for manifests that actually contain js_snippet/switch nodes. */
  rawInputs?: ReadonlyMap<string, Float32Array>;
  snippets?: ReadonlyMap<string, string>;
  snippetCache?: Map<string, SnippetFn>;
}

export interface RunGraphOutput {
  name: string;
  shape: number[];
  data: Float32Array;
}

interface SwitchResolution {
  /** The flattened, GPU-loop-ready node list: js_snippet nodes and JS-routed getitem
   * nodes removed entirely (nothing for the GPU to do), switch nodes replaced by
   * their one chosen branch's nodes plus a synthetic alias so later refs to the
   * switch's own name still resolve. */
  nodes: GraphNode[];
  /** Branch-input bindings, uploaded from JS values — merge into `resolved` before
   * the main loop runs. */
  seeded: Map<string, ResolvedTensor>;
}

/**
 * Synchronous pre-pass: evaluates every `js_snippet` node against plain JS values
 * (model inputs, or another snippet's output — never a GPU buffer, per the v1 scope
 * restriction), resolves every `switch` node's selector, and splices the chosen
 * branch's nodes into the effective execution list. Runs entirely before any GPU
 * command encoder exists. See kuma-bart's plan notes for why this is scoped to
 * input-only snippet dependencies (sidesteps mid-graph GPU readback entirely).
 */
function resolveSwitches(
  device: GPUDevice,
  nodes: readonly GraphNode[],
  rawInputs: ReadonlyMap<string, Float32Array>,
  snippetSources: ReadonlyMap<string, string>,
  snippetCache: Map<string, SnippetFn>,
): SwitchResolution {
  const flat: GraphNode[] = [];
  const seeded = new Map<string, ResolvedTensor>();
  // Plain JS-side values only — a js_snippet's (possibly multi-) output, or a getitem
  // extracting one of them. Keyed "<node>::<index>" for multi-output, plus the
  // getitem's own name once extracted (so further refs by either name work).
  const jsOutputs = new Map<string, number[]>();
  const jsNodeNames = new Set<string>();

  function evalJsValue(ref: ArgValue): number[] {
    if (!isNodeRef(ref)) {
      throw new KumaManifestError(`js_snippet expected a value reference, got ${JSON.stringify(ref)}.`);
    }
    const raw = rawInputs.get(ref.node_ref);
    if (raw) return Array.from(raw);
    const fromJs = jsOutputs.get(ref.node_ref);
    if (fromJs) return fromJs;
    throw new KumaManifestError(
      `js_snippet input "${ref.node_ref}" is neither a model input nor a prior JS-snippet value.`,
    );
  }

  for (const node of nodes) {
    if (node.op === "js_snippet") {
      const fn = getSnippetFn(snippetCache, snippetSources, node.target);
      const inputs = node.args.map((arg) => evalJsValue(arg));
      const outputs = fn(inputs);
      outputs.forEach((arr, i) => jsOutputs.set(`${node.name}::${i}`, arr));
      jsNodeNames.add(node.name);
      continue;
    }

    if (node.op === "call_function" && node.target === "getitem") {
      const [sourceRef, indexArg] = node.args as [NodeRef, number];
      if (jsNodeNames.has(sourceRef.node_ref)) {
        const arr = jsOutputs.get(`${sourceRef.node_ref}::${indexArg}`);
        if (!arr) {
          throw new KumaManifestError(
            `getitem "${node.name}" references "${sourceRef.node_ref}"[${indexArg}], which produced no JS-snippet output.`,
          );
        }
        jsOutputs.set(node.name, arr);
        jsNodeNames.add(node.name);
        continue;
      }
      // Not JS-routed (e.g. a normal aten.chunk.default getitem) — flows through
      // to the main loop unchanged, exactly as before this feature existed.
      flat.push(node);
      continue;
    }

    if (node.op === "switch") {
      if (!node.selector) {
        throw new KumaManifestError(`switch "${node.name}" is missing its selector.`);
      }
      const selectorValues = jsOutputs.get(node.selector.node_ref);
      if (!selectorValues || selectorValues.length !== 1) {
        throw new KumaManifestError(
          `switch "${node.name}"'s selector ("${node.selector.node_ref}") must resolve to exactly one JS-side value.`,
        );
      }
      const selectorValue = Math.trunc(selectorValues[0]!);
      const branches = node.branches ?? [];
      const branch = branches[selectorValue];
      if (!branch) {
        throw new KumaManifestError(
          `switch "${node.name}": no branch for selector value ${selectorValue} (have ${branches.length}).`,
        );
      }

      const outerArgs = node.args;
      const branchInputs = branch.inputs;
      if (branchInputs.length !== outerArgs.length) {
        throw new KumaManifestError(
          `switch "${node.name}": branch declares ${branchInputs.length} input(s), but ${outerArgs.length} were provided.`,
        );
      }
      for (let i = 0; i < branchInputs.length; i++) {
        const arr = evalJsValue(outerArgs[i]!);
        const data = new Float32Array(arr);
        seeded.set(branchInputs[i]!.node_ref, { buffer: uploadFloat32(device, data), shape: [data.length] });
      }

      flat.push(...branch.nodes);
      // Synthetic free passthrough so later refs to the switch's own name (e.g. the
      // top-level output node) resolve to whichever branch actually ran.
      flat.push({
        id: -1,
        name: node.name,
        op: "call_function",
        target: "aten.alias.default",
        args: [branch.output],
        kwargs: {},
        meta: node.meta,
      });
      continue;
    }

    flat.push(node);
  }

  return { nodes: flat, seeded };
}

/** Walks `manifest.graph.nodes` in order (already topologically sorted by the Python
 * exporter), resolving each node to a GPUBuffer and dispatching its op, then reads back
 * the tensors the `output` node points at. */
export async function runGraph(params: RunGraphParams): Promise<RunGraphOutput[]> {
  const { device, manifest, kernels, pipelineCache, weightBuffers, inputBuffers } = params;
  const rawInputs = params.rawInputs ?? new Map<string, Float32Array>();
  const snippets = params.snippets ?? new Map<string, string>();
  const snippetCache = params.snippetCache ?? new Map<string, SnippetFn>();

  const { nodes, seeded } = resolveSwitches(device, manifest.graph.nodes, rawInputs, snippets, snippetCache);
  const elisions = findLinearWeightElisions(nodes);
  const resolved = new Map<string, ResolvedTensor>(seeded);

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();

  let outputNode: GraphNode | undefined;

  for (const node of nodes) {
    if (node.op === "placeholder") {
      if (node.kind === "parameter" || node.kind === "buffer") {
        const tensor = weightBuffers.get(node.weight_name!);
        if (!tensor) {
          throw new KumaManifestError(
            `Manifest references weight "${node.weight_name}" with no matching entry in weights[].`,
          );
        }
        resolved.set(node.name, tensor);
      } else {
        const tensor = resolved.get(node.name) ?? inputBuffers.get(node.name);
        if (!tensor) {
          throw new KumaManifestError(`Missing input for placeholder "${node.name}" — pass it to KumaModel.run().`);
        }
        resolved.set(node.name, tensor);
      }
      continue;
    }

    if (node.op === "output") {
      outputNode = node;
      continue;
    }

    // call_function
    const aliasTarget = elisions.get(node.name);
    if (aliasTarget !== undefined) {
      const tensor = resolved.get(aliasTarget);
      if (!tensor) {
        throw new KumaManifestError(
          `Internal: elided transpose "${node.name}" aliases "${aliasTarget}" before it was resolved.`,
        );
      }
      resolved.set(node.name, tensor);
      continue;
    }

    const handler = opRegistry.get(node.target);
    if (!handler) {
      throw new KumaUnsupportedOpError(node.target, node.name);
    }
    const ctx = new OpContext(device, kernels, pipelineCache, pass, resolved, node);
    handler(ctx);
    // Multi-output nodes (e.g. aten.chunk.default) register per-index results via
    // setIndexedOutput instead of a single setOutput — nothing to check here for them.
    const isMultiOutput = node.meta.outputs !== undefined;
    if (!isMultiOutput && !resolved.has(node.name)) {
      throw new KumaManifestError(
        `Internal: op handler for "${node.target}" did not register an output for node "${node.name}".`,
      );
    }
  }

  pass.end();
  device.queue.submit([encoder.finish()]);

  if (!outputNode) {
    throw new KumaManifestError('Manifest graph has no "output" node.');
  }

  const outputRefs = (outputNode.args[0] as NodeRef[] | undefined) ?? [];
  const reads = outputRefs.map((ref) => {
    const tensor = resolved.get(ref.node_ref);
    if (!tensor) {
      throw new KumaManifestError(`Output references "${ref.node_ref}" which was never computed.`);
    }
    return { buffer: tensor.buffer, byteLength: numElements(tensor.shape) * 4, shape: tensor.shape };
  });

  const datas = await readBuffers(device, reads);

  return manifest.outputs.map((spec, i) => ({
    name: spec.name,
    shape: reads[i]!.shape,
    data: datas[i]!,
  }));
}
