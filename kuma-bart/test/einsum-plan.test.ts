import { describe, expect, it } from "vitest";
import { planEinsum } from "../src/engine/einsum-plan.js";

interface NDArray {
  data: number[];
  shape: number[];
}

function contiguousStrides(shape: readonly number[]): number[] {
  const strides = new Array<number>(shape.length);
  let acc = 1;
  for (let d = shape.length - 1; d >= 0; d--) {
    strides[d] = acc;
    acc *= shape[d]!;
  }
  return strides;
}

function numElements(shape: readonly number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

/** Independent reference: the literal definition of einsum -- sum over every index
 * letter not in the output, of the product of every operand's corresponding entry. */
function bruteForceEinsum(equation: string, operands: NDArray[]): NDArray {
  const arrowIdx = equation.indexOf("->");
  const inputSubs = equation
    .slice(0, arrowIdx)
    .split(",")
    .map((s) => s.trim());
  const outputSub = equation.slice(arrowIdx + 2).trim();

  const extent = new Map<string, number>();
  inputSubs.forEach((sub, i) => {
    for (let d = 0; d < sub.length; d++) extent.set(sub[d]!, operands[i]!.shape[d]!);
  });

  const allLetters = [...extent.keys()];
  const letterExtents = allLetters.map((l) => extent.get(l)!);
  const totalCombos = numElements(letterExtents);

  const outShape = [...outputSub].map((l) => extent.get(l)!);
  const outStrides = contiguousStrides(outShape);
  const opStrides = operands.map((op) => contiguousStrides(op.shape));
  const out = new Array<number>(numElements(outShape)).fill(0);

  for (let combo = 0; combo < totalCombos; combo++) {
    let rem = combo;
    const letterVal = new Map<string, number>();
    for (let li = allLetters.length - 1; li >= 0; li--) {
      const ext = letterExtents[li]!;
      letterVal.set(allLetters[li]!, rem % ext);
      rem = Math.floor(rem / ext);
    }

    let product = 1;
    for (let i = 0; i < operands.length; i++) {
      const sub = inputSubs[i]!;
      let offset = 0;
      for (let d = 0; d < sub.length; d++) offset += letterVal.get(sub[d]!)! * opStrides[i]![d]!;
      product *= operands[i]!.data[offset]!;
    }

    let outOffset = 0;
    for (let d = 0; d < outputSub.length; d++) outOffset += letterVal.get(outputSub[d]!)! * outStrides[d]!;
    out[outOffset] = out[outOffset]! + product;
  }

  return { data: out, shape: outShape };
}

/** Mirrors exactly what ops/einsum.ts dispatches for real (generalized permute +
 * reshape + batched matmul), but on plain JS arrays -- so planEinsum's classification
 * logic gets checked against the brute-force definition above before any GPU code
 * touches it. */
function permuteND(arr: NDArray, dims: readonly number[]): NDArray {
  const inStrides = contiguousStrides(arr.shape);
  const outShape = dims.map((d) => arr.shape[d]!);
  const outStrides = contiguousStrides(outShape);
  const size = numElements(outShape);
  const out = new Array<number>(size);
  for (let i = 0; i < size; i++) {
    let inOffset = 0;
    for (let d = 0; d < dims.length; d++) {
      const coord = Math.floor(i / outStrides[d]!) % outShape[d]!;
      inOffset += coord * inStrides[dims[d]!]!;
    }
    out[i] = arr.data[inOffset]!;
  }
  return { data: out, shape: outShape };
}

function bmmFlat(a: number[], b: number[], batch: number, m: number, k: number, n: number): number[] {
  const out = new Array<number>(batch * m * n).fill(0);
  for (let bi = 0; bi < batch; bi++) {
    for (let mi = 0; mi < m; mi++) {
      for (let ni = 0; ni < n; ni++) {
        let acc = 0;
        for (let ki = 0; ki < k; ki++) {
          acc += a[(bi * m + mi) * k + ki]! * b[(bi * k + ki) * n + ni]!;
        }
        out[(bi * m + mi) * n + ni] = acc;
      }
    }
  }
  return out;
}

function executeEinsumPlan(equation: string, operands: NDArray[]): NDArray {
  const plan = planEinsum(equation, operands.map((o) => o.shape));
  let acc = operands[0]!;
  for (const step of plan.steps) {
    const permutedAcc = permuteND(acc, step.accPermDims);
    const permutedOp = permuteND(operands[step.operandIndex]!, step.opPermDims);
    const result = bmmFlat(permutedAcc.data, permutedOp.data, step.batchSize, step.m, step.k, step.n);
    acc = { data: result, shape: step.resultShape };
  }
  const final = permuteND(acc, plan.finalPermDims);
  return final;
}

function randomArray(shape: number[], seed: number): NDArray {
  let s = seed;
  const next = () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return (s % 1000) / 1000 - 0.5;
  };
  const n = numElements(shape);
  const data = Array.from({ length: n }, next);
  return { data, shape };
}

describe("planEinsum + pairwise-bmm execution vs. brute-force reference", () => {
  it("plain matrix multiply: ij,jk->ik", () => {
    const a = randomArray([3, 4], 1);
    const b = randomArray([4, 5], 2);
    const expected = bruteForceEinsum("ij,jk->ik", [a, b]);
    const actual = executeEinsumPlan("ij,jk->ik", [a, b]);
    expect(actual.shape).toEqual(expected.shape);
    actual.data.forEach((v, i) => expect(v).toBeCloseTo(expected.data[i]!, 6));
  });

  it("batched matmul: bij,bjk->bik", () => {
    const a = randomArray([2, 3, 4], 3);
    const b = randomArray([2, 4, 5], 4);
    const expected = bruteForceEinsum("bij,bjk->bik", [a, b]);
    const actual = executeEinsumPlan("bij,bjk->bik", [a, b]);
    expect(actual.shape).toEqual(expected.shape);
    actual.data.forEach((v, i) => expect(v).toBeCloseTo(expected.data[i]!, 6));
  });

  it("original (unbatched) Tucker reconstruction: ijkl,ti,cj,hk,wl->tchw", () => {
    const core = randomArray([2, 3, 4, 5], 5);
    const ut = randomArray([6, 2], 6);
    const uc = randomArray([7, 3], 7);
    const uh = randomArray([8, 4], 8);
    const uw = randomArray([9, 5], 9);
    const eq = "ijkl,ti,cj,hk,wl->tchw";
    const expected = bruteForceEinsum(eq, [core, ut, uc, uh, uw]);
    const actual = executeEinsumPlan(eq, [core, ut, uc, uh, uw]);
    expect(actual.shape).toEqual(expected.shape);
    actual.data.forEach((v, i) => expect(v).toBeCloseTo(expected.data[i]!, 5));
  });

  it("modulated (batched-over-t) Tucker reconstruction: ijkl,ti,tcj,thk,twl->tchw", () => {
    const T = 3;
    const core = randomArray([2, 3, 4, 5], 10);
    const ut = randomArray([T, 2], 11);
    const uc = randomArray([T, 7, 3], 12);
    const uh = randomArray([T, 8, 4], 13);
    const uw = randomArray([T, 9, 5], 14);
    const eq = "ijkl,ti,tcj,thk,twl->tchw";
    const expected = bruteForceEinsum(eq, [core, ut, uc, uh, uw]);
    const actual = executeEinsumPlan(eq, [core, ut, uc, uh, uw]);
    expect(actual.shape).toEqual(expected.shape);
    actual.data.forEach((v, i) => expect(v).toBeCloseTo(expected.data[i]!, 5));
  });

  it("outer product (no shared/contracted indices): i,j->ij", () => {
    const a = randomArray([3], 15);
    const b = randomArray([4], 16);
    const expected = bruteForceEinsum("i,j->ij", [a, b]);
    const actual = executeEinsumPlan("i,j->ij", [a, b]);
    expect(actual.shape).toEqual(expected.shape);
    actual.data.forEach((v, i) => expect(v).toBeCloseTo(expected.data[i]!, 6));
  });

  it("three-operand chain with output order different from accumulation order", () => {
    const a = randomArray([3, 4], 17); // ij
    const b = randomArray([4, 5], 18); // jk
    const c = randomArray([5, 6], 19); // kl
    const eq = "ij,jk,kl->li"; // output order reversed relative to natural accumulation
    const expected = bruteForceEinsum(eq, [a, b, c]);
    const actual = executeEinsumPlan(eq, [a, b, c]);
    expect(actual.shape).toEqual(expected.shape);
    actual.data.forEach((v, i) => expect(v).toBeCloseTo(expected.data[i]!, 5));
  });

  it("rejects implicit-output equations", () => {
    expect(() => planEinsum("ij,jk", [[3, 4], [4, 5]])).toThrow(/explicit/);
  });

  it("rejects a single-operand equation", () => {
    expect(() => planEinsum("ij->ji", [[3, 4]])).toThrow(/2\+ operand/);
  });

  it("rejects inconsistent extents for the same index letter", () => {
    expect(() => planEinsum("ij,jk->ik", [[3, 4], [5, 6]])).toThrow(/inconsistent extents/);
  });
});

interface ComplexNDArray {
  re: number[];
  im?: number[];
  shape: number[];
}

/** Independent complex reference: same brute-force definition as bruteForceEinsum
 * above, but accumulating complex products (a+bi)(c+di) = (ac-bd)+(ad+bc)i. */
function bruteForceEinsumComplex(equation: string, operands: ComplexNDArray[]): ComplexNDArray {
  const arrowIdx = equation.indexOf("->");
  const inputSubs = equation
    .slice(0, arrowIdx)
    .split(",")
    .map((s) => s.trim());
  const outputSub = equation.slice(arrowIdx + 2).trim();

  const extent = new Map<string, number>();
  inputSubs.forEach((sub, i) => {
    for (let d = 0; d < sub.length; d++) extent.set(sub[d]!, operands[i]!.shape[d]!);
  });

  const allLetters = [...extent.keys()];
  const letterExtents = allLetters.map((l) => extent.get(l)!);
  const totalCombos = numElements(letterExtents);

  const outShape = [...outputSub].map((l) => extent.get(l)!);
  const outStrides = contiguousStrides(outShape);
  const opStrides = operands.map((op) => contiguousStrides(op.shape));
  const outRe = new Array<number>(numElements(outShape)).fill(0);
  const outIm = new Array<number>(numElements(outShape)).fill(0);

  for (let combo = 0; combo < totalCombos; combo++) {
    let rem = combo;
    const letterVal = new Map<string, number>();
    for (let li = allLetters.length - 1; li >= 0; li--) {
      const ext = letterExtents[li]!;
      letterVal.set(allLetters[li]!, rem % ext);
      rem = Math.floor(rem / ext);
    }

    let prodRe = 1;
    let prodIm = 0;
    for (let i = 0; i < operands.length; i++) {
      const sub = inputSubs[i]!;
      let offset = 0;
      for (let d = 0; d < sub.length; d++) offset += letterVal.get(sub[d]!)! * opStrides[i]![d]!;
      const vRe = operands[i]!.re[offset]!;
      const vIm = operands[i]!.im ? operands[i]!.im![offset]! : 0;
      const newRe = prodRe * vRe - prodIm * vIm;
      const newIm = prodRe * vIm + prodIm * vRe;
      prodRe = newRe;
      prodIm = newIm;
    }

    let outOffset = 0;
    for (let d = 0; d < outputSub.length; d++) outOffset += letterVal.get(outputSub[d]!)! * outStrides[d]!;
    outRe[outOffset] = outRe[outOffset]! + prodRe;
    outIm[outOffset] = outIm[outOffset]! + prodIm;
  }

  return { re: outRe, im: outIm, shape: outShape };
}

/** Mirrors ops/einsum.ts's permuteMaybeComplex + dispatchComplexBmm exactly, but on
 * plain JS arrays. */
function permuteMaybeComplexJS(arr: ComplexNDArray, dims: readonly number[]): ComplexNDArray {
  const re = permuteND({ data: arr.re, shape: arr.shape }, dims).data;
  const shape = dims.map((d) => arr.shape[d]!);
  if (!arr.im) return { re, shape };
  const im = permuteND({ data: arr.im, shape: arr.shape }, dims).data;
  return { re, im, shape };
}

function complexBmmJS(a: ComplexNDArray, b: ComplexNDArray, batch: number, m: number, k: number, n: number): ComplexNDArray {
  const reAB = bmmFlat(a.re, b.re, batch, m, k, n);
  const shape = [batch, m, n];
  if (!a.im && !b.im) return { re: reAB, shape };
  if (a.im && !b.im) return { re: reAB, im: bmmFlat(a.im, b.re, batch, m, k, n), shape };
  if (!a.im && b.im) return { re: reAB, im: bmmFlat(a.re, b.im, batch, m, k, n), shape };
  const imIm = bmmFlat(a.im!, b.im!, batch, m, k, n);
  const reIm = bmmFlat(a.re, b.im!, batch, m, k, n);
  const imRe = bmmFlat(a.im!, b.re, batch, m, k, n);
  return { re: reAB.map((v, i) => v - imIm[i]!), im: reIm.map((v, i) => v + imRe[i]!), shape };
}

function executeEinsumPlanComplex(equation: string, operands: ComplexNDArray[]): ComplexNDArray {
  const plan = planEinsum(equation, operands.map((o) => o.shape));
  let acc = operands[0]!;
  for (const step of plan.steps) {
    const permutedAcc = permuteMaybeComplexJS(acc, step.accPermDims);
    const permutedOp = permuteMaybeComplexJS(operands[step.operandIndex]!, step.opPermDims);
    const aFor3d: ComplexNDArray = { re: permutedAcc.re, im: permutedAcc.im, shape: [step.batchSize, step.m, step.k] };
    const bFor3d: ComplexNDArray = { re: permutedOp.re, im: permutedOp.im, shape: [step.batchSize, step.k, step.n] };
    const result = complexBmmJS(aFor3d, bFor3d, step.batchSize, step.m, step.k, step.n);
    acc = { re: result.re, im: result.im, shape: step.resultShape };
  }
  return permuteMaybeComplexJS(acc, plan.finalPermDims);
}

function randomComplex(shape: number[], seed: number, complex: boolean): ComplexNDArray {
  const re = randomArray(shape, seed).data;
  if (!complex) return { re, shape };
  return { re, im: randomArray(shape, seed + 1000).data, shape };
}

describe("complex-aware einsum execution vs. a complex brute-force reference", () => {
  it("complex core, 1 real factor, 3 complex factors (mirrors the real 3mbunny graph)", () => {
    const eq = "ijkl,ti,cj,hk,wl->tchw";
    const operands = [
      randomComplex([2, 3, 4, 5], 1, true),
      randomComplex([6, 2], 3, false),
      randomComplex([7, 3], 4, true),
      randomComplex([8, 4], 6, true),
      randomComplex([9, 5], 8, true),
    ];
    const expected = bruteForceEinsumComplex(eq, operands);
    const actual = executeEinsumPlanComplex(eq, operands);
    expect(actual.shape).toEqual(expected.shape);
    actual.re.forEach((v, i) => expect(v).toBeCloseTo(expected.re[i]!, 5));
    actual.im!.forEach((v, i) => expect(v).toBeCloseTo(expected.im![i]!, 5));
  });

  it("all-real operands produce no imaginary part", () => {
    const eq = "ij,jk->ik";
    const operands = [randomComplex([3, 4], 10, false), randomComplex([4, 5], 11, false)];
    const expected = bruteForceEinsumComplex(eq, operands);
    const actual = executeEinsumPlanComplex(eq, operands);
    expect(actual.im).toBeUndefined();
    actual.re.forEach((v, i) => expect(v).toBeCloseTo(expected.re[i]!, 6));
  });

  it("both operands complex", () => {
    const eq = "ij,jk->ik";
    const operands = [randomComplex([3, 4], 12, true), randomComplex([4, 5], 14, true)];
    const expected = bruteForceEinsumComplex(eq, operands);
    const actual = executeEinsumPlanComplex(eq, operands);
    actual.re.forEach((v, i) => expect(v).toBeCloseTo(expected.re[i]!, 6));
    actual.im!.forEach((v, i) => expect(v).toBeCloseTo(expected.im![i]!, 6));
  });

  it("modulated (batched-over-t) Tucker with a complex core and mixed factors", () => {
    const eq = "ijkl,ti,tcj,thk,twl->tchw";
    const T = 3;
    const operands = [
      randomComplex([2, 3, 4, 5], 19, true),
      randomComplex([T, 2], 21, false),
      randomComplex([T, 7, 3], 22, true),
      randomComplex([T, 8, 4], 24, true),
      randomComplex([T, 9, 5], 26, true),
    ];
    const expected = bruteForceEinsumComplex(eq, operands);
    const actual = executeEinsumPlanComplex(eq, operands);
    expect(actual.shape).toEqual(expected.shape);
    actual.re.forEach((v, i) => expect(v).toBeCloseTo(expected.re[i]!, 5));
    actual.im!.forEach((v, i) => expect(v).toBeCloseTo(expected.im![i]!, 5));
  });
});
