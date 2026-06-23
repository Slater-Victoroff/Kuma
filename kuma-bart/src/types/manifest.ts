export interface NodeRef {
  node_ref: string;
}

export type ArgValue = NodeRef | number | boolean | string | null | ArgValue[];

export interface IOSpec {
  name: string;
  shape?: number[];
  dtype?: string;
  kind?: string;
}

export interface WeightEntry {
  name: string;
  shape: number[];
  dtype: string;
  n_elements: number;
  byte_offset: number;
  byte_length: number;
}

export interface NodeMeta {
  shape?: number[];
  dtype?: string;
  /** Multi-output nodes (e.g. aten.chunk.default) have no top-level `shape` — instead
   * one entry per output, picked apart by subsequent `getitem(node, i)` nodes. */
  outputs?: { shape: number[]; dtype?: string }[];
}

export type GraphNodeOp = "placeholder" | "call_function" | "output" | "js_snippet" | "switch";
export type PlaceholderKind = "parameter" | "buffer" | "user_input";

/** One possible branch of a `switch` node — its own independently-namespaced node list
 * (a "little graph", names guaranteed unique against the outer scope and other
 * branches by whatever assembled the manifest), plus which of its own placeholder
 * nodes need binding from the switch's outer-scope `args` (positionally matched), and
 * which of its own nodes is the branch's result. */
export interface SwitchBranch {
  nodes: GraphNode[];
  inputs: NodeRef[];
  output: NodeRef;
}

export interface GraphNode {
  id: number;
  name: string;
  op: GraphNodeOp;
  target: string;
  /** For `op: "switch"`: outer-scope JS-side values (e.g. a js_snippet's output, via
   * getitem), positionally matched to the chosen branch's own `inputs`. */
  args: ArgValue[];
  kwargs: Record<string, ArgValue>;
  meta: NodeMeta;
  kind?: PlaceholderKind;
  weight_name?: string;
  /** `op: "switch"` only — which (js_snippet-derived) value selects the branch. */
  selector?: NodeRef;
  /** `op: "switch"` only — branches[selector's resolved integer value] executes;
   * the rest are never dispatched. */
  branches?: SwitchBranch[];
}

export interface KumaManifest {
  format: string;
  format_version: number;
  weight_file: string;
  endianness: "little" | "big";
  inputs: IOSpec[];
  outputs: IOSpec[];
  weights: WeightEntry[];
  graph: {
    node_count: number;
    op_counts: Record<string, number>;
    nodes: GraphNode[];
  };
  warnings: string[];
  unsupported_ops: string[];
}

export function isNodeRef(v: ArgValue): v is NodeRef {
  return typeof v === "object" && v !== null && !Array.isArray(v) && "node_ref" in v;
}
