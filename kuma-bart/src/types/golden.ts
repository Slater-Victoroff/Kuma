/** golden.json's shape -- a companion to manifest.json, optionally bundled in the same
 * .iph package. Captured from one real eager-PyTorch run per branch (see
 * src/kuma/golden.py on the Python side): a handful of summary stats per graph node,
 * never full activations, for verifying a runtime computes the *same* values a real
 * PyTorch run would, not just NaN-free ones. */

export interface GoldenTensorStats {
  shape: number[];
  n: number;
  finite: number;
  mean: number;
  min: number;
  max: number;
  first: number[];
  /** Flat indices into the tensor, deterministically chosen at capture time and spread
   * across its full extent (not just the start) -- paired 1:1 with `spread`. A
   * verifier samples its own computed tensor at these same indices rather than needing
   * to reproduce the Python side's index-picking scheme itself. */
  spread_indices: number[];
  spread: number[];
}

export interface GoldenNodeStats {
  /** Real part always present; "im" only for a node whose eager value was complex --
   * mirrors kuma-bart's ResolvedTensor.imag pairing (never one interleaved buffer). */
  re: GoldenTensorStats;
  im?: GoldenTensorStats;
}

export interface GoldenBranch {
  /** Keyed by the (already branch-namespaced, where applicable) placeholder node name
   * -- exactly the same name the manifest's own graph uses for that placeholder. */
  inputs: Record<string, number[]>;
  /** Keyed by node name, same namespacing as the manifest's graph nodes. */
  nodes: Record<string, GoldenNodeStats>;
}

export interface GoldenData {
  format_version: number;
  /** One entry per branch (matching the manifest's switch node's `branches` order,
   * index-for-index) for a multi-segment model; exactly one entry, un-namespaced, for a
   * non-branching model -- see kuma.compiler.compile vs kuma.branching.compile_branching
   * on the Python side. */
  branches: GoldenBranch[];
}
