export const KERNEL_WORKGROUP_SIZE = 64;
export const MAX_WORKGROUPS_PER_DIMENSION = 65535;

/**
 * Picks (x, y) workgroup-dispatch counts for `totalWorkgroups`, maxing out x first.
 * Every kernel in this project's set computes its linear element index as
 * `gid.x + gid.y * (num_workgroups.x * 64)` (see kuma's kernels/__init__.py — every
 * kernel was updated to this scheme together), so any (x, y) with `x*y >= totalWorkgroups`
 * is correct: invocations beyond the real element count are no-ops via each kernel's own
 * `if (i >= params.n) { return; }` bounds check. Returns null if even a full
 * 65535x65535 grid isn't enough (≈4.3 billion workgroups, ≈274 billion elements —
 * not expected to happen in practice, but checked rather than assumed).
 */
export function computeDispatchGrid(totalWorkgroups: number): { x: number; y: number } | null {
  if (totalWorkgroups <= MAX_WORKGROUPS_PER_DIMENSION) {
    return { x: totalWorkgroups, y: 1 };
  }
  const x = MAX_WORKGROUPS_PER_DIMENSION;
  const y = Math.ceil(totalWorkgroups / x);
  return y <= MAX_WORKGROUPS_PER_DIMENSION ? { x, y } : null;
}
