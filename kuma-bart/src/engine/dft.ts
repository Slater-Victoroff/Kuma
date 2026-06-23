/**
 * Pure-math basis matrices for computing DFT/IDFT via matrix multiplication, so
 * fft_irfft2 can reuse linear.wgsl instead of needing a dedicated FFT kernel. Both
 * transform lengths in the one model that needs this (180, 320) are small and not
 * powers of two anyway, so direct O(N^2) matmul is the simplest correct approach, not
 * a missed optimization opportunity.
 *
 * Row-major, indexed [n][k] with n = output row, k = input column — i.e. already in
 * linear.wgsl's expected weight:(N,K) layout (out = x @ weight^T), no transpose needed.
 */

/**
 * Basis for a length-N complex IFFT, 'ortho' norm:
 *   Re(ifft(X)) = cos @ Xr - sin @ Xi
 *   Im(ifft(X)) = sin @ Xr + cos @ Xi
 * where cos[n,k] = cos(2*pi*k*n/N)/sqrt(N), sin[n,k] = sin(2*pi*k*n/N)/sqrt(N).
 * Both matrices are symmetric (cos[n,k] === cos[k,n]), so orientation is unambiguous.
 */
export function complexIfftBasis(n: number): { cos: Float32Array; sin: Float32Array } {
  const cosMat = new Float32Array(n * n);
  const sinMat = new Float32Array(n * n);
  const scale = 1 / Math.sqrt(n);
  for (let row = 0; row < n; row++) {
    for (let k = 0; k < n; k++) {
      const theta = (2 * Math.PI * k * row) / n;
      cosMat[row * n + k] = Math.cos(theta) * scale;
      sinMat[row * n + k] = Math.sin(theta) * scale;
    }
  }
  return { cos: cosMat, sin: sinMat };
}

/**
 * Basis for irfft: W real outputs from Whalf = W/2+1 complex bins ('ortho' norm):
 *   out = a @ Xr + b @ Xi   (a, b shaped (W, Whalf))
 *
 * Derivation: reconstructing the full W-point spectrum via Hermitian symmetry
 * (X_full[W-k] = conj(X[k])) and taking the real part of the standard inverse DFT, the
 * k=0 and Nyquist (k=W/2) bins each appear once (contributing only their real part —
 * Re is linear, so each term's imaginary part simply drops out), while every other bin
 * contributes its own-plus-mirror pair, i.e. doubled:
 *   out[n] = (1/sqrt(W)) * [ Xr[0] + Xr[W/2]*(-1)^n + sum_{k=1}^{W/2-1} 2*(Xr[k]*cos(th)
 *            - Xi[k]*sin(th)) ],  th = 2*pi*k*n/W
 */
export function irfftBasis(w: number): { a: Float32Array; b: Float32Array } {
  const whalf = w / 2 + 1;
  if (!Number.isInteger(whalf)) {
    throw new Error(`irfftBasis: w=${w} must be even (got w/2+1=${whalf}).`);
  }
  const nyquist = w / 2;
  const scale = 1 / Math.sqrt(w);
  const a = new Float32Array(w * whalf);
  const b = new Float32Array(w * whalf);

  for (let n = 0; n < w; n++) {
    for (let k = 0; k < whalf; k++) {
      let coeffA: number;
      let coeffB: number;
      if (k === 0) {
        coeffA = 1;
        coeffB = 0;
      } else if (k === nyquist) {
        coeffA = n % 2 === 0 ? 1 : -1;
        coeffB = 0;
      } else {
        const theta = (2 * Math.PI * k * n) / w;
        coeffA = 2 * Math.cos(theta);
        coeffB = -2 * Math.sin(theta);
      }
      a[n * whalf + k] = coeffA * scale;
      b[n * whalf + k] = coeffB * scale;
    }
  }
  return { a, b };
}
