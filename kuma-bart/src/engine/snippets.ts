import { KumaManifestError } from "../errors.js";

export type SnippetFn = (inputs: number[][]) => number[][];

/**
 * Compiles a `js_snippet` node's bundled source. Calling convention is deliberately
 * minimal: the snippet defines a top-level `function main(inputs)` — `inputs` is one
 * plain `number[]` per declared input (in the node's `args` order), and it returns one
 * `number[]` per declared output (in the node's `meta.outputs` order). No module
 * system, no imports — this is for tiny scalar/control-flow logic (e.g. picking which
 * branch of a `switch` node runs), not general-purpose script execution.
 */
export function compileSnippet(name: string, source: string): SnippetFn {
  let fn: unknown;
  try {
    const factory = new Function(`${source}\nreturn main;`);
    fn = factory();
  } catch (err) {
    throw new KumaManifestError(`Failed to compile snippet "${name}": ${(err as Error).message}`);
  }
  if (typeof fn !== "function") {
    throw new KumaManifestError(`Snippet "${name}" must define a top-level function named "main".`);
  }
  return fn as SnippetFn;
}

/** Looks up `name` in `cache`, compiling (and caching) from `sources` on first use —
 * mirrors OpContext's pipeline caching, but standalone since this runs in the
 * scheduler's js_snippet/switch pre-pass, before any OpContext exists. */
export function getSnippetFn(
  cache: Map<string, SnippetFn>,
  sources: ReadonlyMap<string, string>,
  name: string,
): SnippetFn {
  let fn = cache.get(name);
  if (!fn) {
    const source = sources.get(name);
    if (!source) {
      throw new KumaManifestError(`.iph package is missing snippet source "snippets/${name}".`);
    }
    fn = compileSnippet(name, source);
    cache.set(name, fn);
  }
  return fn;
}
