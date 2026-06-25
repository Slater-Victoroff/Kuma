/** Used by the golden-value verifier (engine/verify.ts) to summarize a computed
 * tensor's data for comparison against a real eager-PyTorch run's recorded values. */
export interface BufferSummary {
  finite: number;
  nan: number;
  posInf: number;
  negInf: number;
  mean: number;
  min: number;
  max: number;
  first: number[];
}

/** Comparisons against NaN are always false, so naive min/max tracking alone would
 * misreport min=Infinity, max=-Infinity for an all-NaN buffer instead of surfacing the
 * NaN — this counts explicitly instead. */
export function summarize(data: Float32Array, firstCount = 8): BufferSummary {
  let finite = 0;
  let nan = 0;
  let posInf = 0;
  let negInf = 0;
  let sum = 0;
  let min = Infinity;
  let max = -Infinity;

  for (const v of data) {
    if (Number.isNaN(v)) {
      nan++;
    } else if (v === Infinity) {
      posInf++;
    } else if (v === -Infinity) {
      negInf++;
    } else {
      finite++;
      sum += v;
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }

  return {
    finite,
    nan,
    posInf,
    negInf,
    mean: finite ? sum / finite : NaN,
    min: finite ? min : NaN,
    max: finite ? max : NaN,
    first: Array.from(data.slice(0, firstCount)),
  };
}
