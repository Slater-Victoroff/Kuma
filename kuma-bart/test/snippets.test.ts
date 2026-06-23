import { describe, expect, it } from "vitest";
import { compileSnippet, getSnippetFn } from "../src/engine/snippets.js";
import { KumaManifestError } from "../src/errors.js";

describe("compileSnippet", () => {
  it("compiles a snippet defining main(inputs) and calls it", () => {
    const fn = compileSnippet("double.js", "function main(inputs) { return [inputs[0].map((x) => x * 2)]; }");
    expect(fn([[1, 2, 3]])).toEqual([[2, 4, 6]]);
  });

  it("supports multiple inputs and multiple outputs", () => {
    const fn = compileSnippet(
      "swap.js",
      "function main(inputs) { return [inputs[1], inputs[0]]; }",
    );
    expect(fn([[1, 2], [3, 4]])).toEqual([[3, 4], [1, 2]]);
  });

  it("throws KumaManifestError if the snippet defines no top-level main", () => {
    expect(() => compileSnippet("bad.js", "const x = 1;")).toThrow(KumaManifestError);
  });

  it("throws KumaManifestError on a syntax error in the snippet source", () => {
    expect(() => compileSnippet("broken.js", "function main(inputs) { return [")).toThrow(KumaManifestError);
  });
});

describe("getSnippetFn", () => {
  it("compiles once and caches on subsequent lookups", () => {
    const cache = new Map();
    const sources = new Map([["counted.js", "function main(inputs) { return [[42]]; }"]]);

    const first = getSnippetFn(cache, sources, "counted.js");
    const second = getSnippetFn(cache, sources, "counted.js");
    expect(first).toBe(second);
    expect(cache.size).toBe(1);
  });

  it("throws KumaManifestError when the snippet source is missing from the package", () => {
    const cache = new Map();
    const sources = new Map();
    expect(() => getSnippetFn(cache, sources, "missing.js")).toThrow(KumaManifestError);
  });
});
