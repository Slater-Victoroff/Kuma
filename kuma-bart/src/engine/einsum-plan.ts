/**
 * General einsum support, planned as a sequence of pairwise contractions executed via
 * permute + (batched) matmul — the same strategy numpy/opt_einsum use internally. Pure
 * planning logic, no GPU/JS-array dependency, so it's directly unit-testable (see
 * test/einsum-plan.test.ts) independent of how a caller actually executes the plan.
 *
 * Only explicit "->output" equations are supported (no implicit-output inference), and
 * only 2+ operand contractions (a single-operand einsum — transpose/diagonal/sum-only —
 * isn't needed by anything today and isn't covered).
 *
 * Algorithm: start with operand 0 as the running accumulator. For each subsequent
 * operand, classify every index letter relative to {accumulator, this operand}:
 *   - in both, and needed later (by a later operand or the output)  -> batch
 *   - in both, not needed later                                     -> contracted (summed)
 *   - only in the accumulator                                       -> accOnly (kept)
 *   - only in this operand                                          -> opOnly (kept)
 * Contract via a batched matmul: permute the accumulator to
 * [batch, accOnly, contracted] and the operand to [batch, contracted, opOnly], reshape
 * both to 3D, bmm, reshape the (batch, accOnly-product, opOnly-product) result back out
 * to the per-letter shape [batchDims..., accOnlyDims..., opOnlyDims...]. That becomes
 * the new accumulator, with subscript batchLetters+accOnlyLetters+opOnlyLetters. After
 * the last operand, a final permute reorders the accumulated axes to match the
 * requested output subscript exactly.
 */

export interface EinsumStep {
  operandIndex: number;
  accPermDims: number[];
  accPermutedShape: number[];
  opPermDims: number[];
  opPermutedShape: number[];
  batchSize: number;
  m: number;
  k: number;
  n: number;
  resultShape: number[];
  resultSub: string;
}

export interface EinsumPlan {
  steps: EinsumStep[];
  finalPermDims: number[];
  outputShape: number[];
}

function parseEquation(equation: string): { inputSubs: string[]; outputSub: string } {
  const arrowIdx = equation.indexOf("->");
  if (arrowIdx < 0) {
    throw new Error(`einsum equation "${equation}" must have an explicit "->output" (implicit output isn't supported).`);
  }
  const inputSubs = equation
    .slice(0, arrowIdx)
    .split(",")
    .map((s) => s.trim());
  const outputSub = equation.slice(arrowIdx + 2).trim();
  return { inputSubs, outputSub };
}

export function planEinsum(equation: string, operandShapes: readonly (readonly number[])[]): EinsumPlan {
  const { inputSubs, outputSub } = parseEquation(equation);

  if (inputSubs.length !== operandShapes.length) {
    throw new Error(`einsum "${equation}" expects ${inputSubs.length} operand(s), got ${operandShapes.length}.`);
  }
  if (inputSubs.length < 2) {
    throw new Error(`einsum "${equation}": only contractions of 2+ operands are supported, got ${inputSubs.length}.`);
  }
  inputSubs.forEach((sub, i) => {
    if (sub.length !== operandShapes[i]!.length) {
      throw new Error(
        `einsum "${equation}": operand ${i} has rank ${operandShapes[i]!.length} but subscript "${sub}" has ${sub.length} letter(s).`,
      );
    }
  });

  const extent = new Map<string, number>();
  inputSubs.forEach((sub, i) => {
    for (let d = 0; d < sub.length; d++) {
      const letter = sub[d]!;
      const e = operandShapes[i]![d]!;
      const existing = extent.get(letter);
      if (existing !== undefined && existing !== e) {
        throw new Error(`einsum "${equation}": index "${letter}" has inconsistent extents (${existing} vs ${e}).`);
      }
      extent.set(letter, e);
    }
  });

  const steps: EinsumStep[] = [];
  let accSub = inputSubs[0]!;
  let accShape = operandShapes[0]!.slice();

  for (let i = 1; i < inputSubs.length; i++) {
    const opSub = inputSubs[i]!;
    const opShape = operandShapes[i]!.slice();

    const neededLater = new Set<string>(outputSub);
    for (let j = i + 1; j < inputSubs.length; j++) {
      for (const ch of inputSubs[j]!) neededLater.add(ch);
    }

    const accLetters = [...accSub];
    const opLetters = [...opSub];
    const shared = accLetters.filter((l) => opLetters.includes(l));

    const batchLetters = shared.filter((l) => neededLater.has(l));
    const contractedLetters = shared.filter((l) => !neededLater.has(l));
    const accOnlyLetters = accLetters.filter((l) => !opLetters.includes(l));
    const opOnlyLetters = opLetters.filter((l) => !accLetters.includes(l));

    const accPermDims = [...batchLetters, ...accOnlyLetters, ...contractedLetters].map((l) => accSub.indexOf(l));
    const opPermDims = [...batchLetters, ...contractedLetters, ...opOnlyLetters].map((l) => opSub.indexOf(l));
    const accPermutedShape = accPermDims.map((d) => accShape[d]!);
    const opPermutedShape = opPermDims.map((d) => opShape[d]!);

    const batchSize = batchLetters.reduce((p, l) => p * extent.get(l)!, 1);
    const m = accOnlyLetters.reduce((p, l) => p * extent.get(l)!, 1);
    const k = contractedLetters.reduce((p, l) => p * extent.get(l)!, 1);
    const n = opOnlyLetters.reduce((p, l) => p * extent.get(l)!, 1);

    const resultSub = [...batchLetters, ...accOnlyLetters, ...opOnlyLetters].join("");
    const resultShape = [
      ...batchLetters.map((l) => extent.get(l)!),
      ...accOnlyLetters.map((l) => extent.get(l)!),
      ...opOnlyLetters.map((l) => extent.get(l)!),
    ];

    steps.push({
      operandIndex: i,
      accPermDims,
      accPermutedShape,
      opPermDims,
      opPermutedShape,
      batchSize,
      m,
      k,
      n,
      resultShape,
      resultSub,
    });

    accSub = resultSub;
    accShape = resultShape;
  }

  const finalPermDims = [...outputSub].map((l) => accSub.indexOf(l));
  const outputShape = [...outputSub].map((l) => extent.get(l)!);

  return { steps, finalPermDims, outputShape };
}
