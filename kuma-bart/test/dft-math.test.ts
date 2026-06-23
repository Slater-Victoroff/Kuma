import { describe, expect, it } from "vitest";
import { complexIfftBasis, irfftBasis } from "../src/engine/dft.js";

function matVec(mat: Float32Array, rows: number, cols: number, vec: readonly number[]): number[] {
  const out = new Array(rows).fill(0);
  for (let r = 0; r < rows; r++) {
    let sum = 0;
    for (let c = 0; c < cols; c++) sum += mat[r * cols + c]! * vec[c]!;
    out[r] = sum;
  }
  return out;
}

function closeArrays(a: readonly number[], b: readonly number[], tol = 1e-4): void {
  expect(a.length).toBe(b.length);
  for (let i = 0; i < a.length; i++) {
    expect(Math.abs(a[i]! - b[i]!)).toBeLessThan(tol);
  }
}

/** Independent reference: direct double-sum complex IDFT, 'ortho' norm — same formula
 * as complexIfftBasis but computed without ever building a matrix, to catch
 * implementation bugs (transposed indices, off-by-one) in the basis construction. */
function referenceComplexIdft(xr: readonly number[], xi: readonly number[]): { re: number[]; im: number[] } {
  const n = xr.length;
  const re = new Array(n).fill(0);
  const im = new Array(n).fill(0);
  for (let t = 0; t < n; t++) {
    let sumRe = 0;
    let sumIm = 0;
    for (let k = 0; k < n; k++) {
      const theta = (2 * Math.PI * k * t) / n;
      const c = Math.cos(theta);
      const s = Math.sin(theta);
      sumRe += xr[k]! * c - xi[k]! * s;
      sumIm += xr[k]! * s + xi[k]! * c;
    }
    re[t] = sumRe / Math.sqrt(n);
    im[t] = sumIm / Math.sqrt(n);
  }
  return { re, im };
}

/** Independent reference for irfft: reconstruct the full W-point spectrum via Hermitian
 * mirroring, then a direct (non-matrix) complex IDFT, keeping only the real part. This
 * is a genuinely different derivation path than irfftBasis's "double the middle bins"
 * algebra, so agreement is a real check on that algebra, not just on indexing. */
function referenceIrfft(real: readonly number[], imag: readonly number[], w: number): number[] {
  const whalf = real.length;
  const fullRe = new Array(w);
  const fullIm = new Array(w);
  for (let k = 0; k < whalf; k++) {
    fullRe[k] = real[k]!;
    fullIm[k] = imag[k]!;
  }
  for (let k = whalf; k < w; k++) {
    const mirror = w - k;
    fullRe[k] = real[mirror]!;
    fullIm[k] = -imag[mirror]!;
  }
  const out = new Array(w).fill(0);
  for (let n = 0; n < w; n++) {
    let sumRe = 0;
    for (let k = 0; k < w; k++) {
      const theta = (2 * Math.PI * k * n) / w;
      sumRe += fullRe[k] * Math.cos(theta) - fullIm[k] * Math.sin(theta);
    }
    out[n] = sumRe / Math.sqrt(w);
  }
  return out;
}

function randomArray(n: number, seed: number): number[] {
  // Tiny deterministic PRNG (mulberry32) so failures are reproducible.
  let s = seed;
  return Array.from({ length: n }, () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296 - 0.5;
  });
}

function addArrays(a: readonly number[], b: readonly number[]): number[] {
  return a.map((v, i) => v + b[i]!);
}

function subArrays(a: readonly number[], b: readonly number[]): number[] {
  return a.map((v, i) => v - b[i]!);
}

describe("complexIfftBasis", () => {
  it("matches a direct-summation IDFT reference on random data", () => {
    const n = 12;
    const xr = randomArray(n, 1);
    const xi = randomArray(n, 2);
    const { cos, sin } = complexIfftBasis(n);

    const cosXr = matVec(cos, n, n, xr);
    const cosXi = matVec(cos, n, n, xi);
    const sinXr = matVec(sin, n, n, xr);
    const sinXi = matVec(sin, n, n, xi);
    const re = subArrays(cosXr, sinXi);
    const im = addArrays(sinXr, cosXi);

    const ref = referenceComplexIdft(xr, xi);
    closeArrays(re, ref.re);
    closeArrays(im, ref.im);
  });

  it("matches the closed-form single-frequency impulse", () => {
    // X[k] = delta(k,1): ifft (ortho) is x[n] = exp(i*2*pi*n/N)/sqrt(N).
    const n = 8;
    const xr = new Array(n).fill(0);
    xr[1] = 1;
    const xi = new Array(n).fill(0);
    const { cos, sin } = complexIfftBasis(n);

    const re = subArrays(matVec(cos, n, n, xr), matVec(sin, n, n, xi));
    const im = addArrays(matVec(sin, n, n, xr), matVec(cos, n, n, xi));

    const expectedRe = Array.from({ length: n }, (_, i) => Math.cos((2 * Math.PI * i) / n) / Math.sqrt(n));
    const expectedIm = Array.from({ length: n }, (_, i) => Math.sin((2 * Math.PI * i) / n) / Math.sqrt(n));
    closeArrays(re, expectedRe);
    closeArrays(im, expectedIm);
  });
});

describe("irfftBasis", () => {
  it("matches a Hermitian-mirror + direct-IDFT reference on random data", () => {
    const w = 16;
    const whalf = w / 2 + 1;
    const real = randomArray(whalf, 3);
    const imag = randomArray(whalf, 4);
    imag[0] = 0; // a physically-real spectrum has Im(X[0])=0; the formula ignores it
    imag[whalf - 1] = 0; // ...and Im(X[Nyquist])=0 — but irfftBasis must be robust either way

    const { a, b } = irfftBasis(w);
    const out = addArrays(matVec(a, w, whalf, real), matVec(b, w, whalf, imag));

    const ref = referenceIrfft(real, imag, w);
    closeArrays(out, ref);
  });

  it("ignores the imaginary part of the k=0 and Nyquist bins (matches reference even when they're nonzero)", () => {
    const w = 16;
    const whalf = w / 2 + 1;
    const real = randomArray(whalf, 5);
    const imag = randomArray(whalf, 6); // deliberately nonzero at the edges this time

    const { a, b } = irfftBasis(w);
    const out = addArrays(matVec(a, w, whalf, real), matVec(b, w, whalf, imag));

    const ref = referenceIrfft(real, imag, w);
    closeArrays(out, ref);
  });

  it("throws for an odd output length", () => {
    expect(() => irfftBasis(15)).toThrow();
  });
});
