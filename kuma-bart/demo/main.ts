import { KumaModel, type VerifyReport, type ProfileReport } from "../src/index.js";
import { GpuFrameRenderer } from "./gpuRender.js";

const logEl = document.querySelector<HTMLPreElement>("#log")!;
const loadButton = document.querySelector<HTMLButtonElement>("#load")!;
const verifyButton = document.querySelector<HTMLButtonElement>("#verify")!;
const profileButton = document.querySelector<HTMLButtonElement>("#profile")!;
const playButton = document.querySelector<HTMLButtonElement>("#play")!;
const pathInput = document.querySelector<HTMLInputElement>("#path")!;
const tSlider = document.querySelector<HTMLInputElement>("#t")!;
const tValueEl = document.querySelector<HTMLSpanElement>("#tValue")!;
const frameStatusEl = document.querySelector<HTMLDivElement>("#frameStatus")!;
const canvas = document.querySelector<HTMLCanvasElement>("#frame")!;

function log(line: string): void {
  logEl.textContent += `\n${line}`;
}

let model: KumaModel | undefined;
let gpuRenderer: GpuFrameRenderer | undefined; // created once load() has a device to render on
let timeInputName: string | undefined; // set once load() confirms a single-scalar time input
let inFlight = 0; // frames submitted but not yet confirmed complete via onSubmittedWorkDone()
let pendingT: number | undefined; // latest-wins: superseded mid-flight requests land here
let playing = false;
let playStartMs = 0;
let capacityHits = 0; // how many requested frames hit the MAX_IN_FLIGHT cap this Play session
let requestedFrames = 0; // how many frames Play actually tried to request this session
let lastTickMs = -Infinity; // for detecting gaps between rAF callbacks wider than expected

// 0% capacity hits rules out the in-flight pool as the cause of a reported pause, which
// points at either (a) the JS main thread itself being briefly blocked (e.g. a GC pause
// -- every frame allocates real JS objects with no pooling, at up to 60fps) or (b)
// runToGpu()'s own CPU-side encode+submit step occasionally spiking for some other
// reason. Neither shows up in MAX_IN_FLIGHT/capacityHits at all, since render() is
// gated only by runToGpu()'s promise resolving, never by onSubmittedWorkDone() -- a
// gap between rAF ticks much wider than the ~16.67ms a 60fps cap should produce is (a);
// an enqueue time in frameStatus much higher than the usual ~2-7ms is (b).
const STALL_THRESHOLD_MS = 50;

// Measured: real GPU compute for one frame is ~7ms, but whatever mechanism this
// browser uses to notify the CPU that submitted GPU work has completed (whether via
// onSubmittedWorkDone() or mapAsync()) has a ~30ms latency *independent of how much
// work or data is involved* -- confirmed by a GPU-side timestamp around the shared
// pass landing at ~7ms (matching profile()'s per-op breakdown) while a 16-byte debug
// readback alone took ~30ms. Waiting for that notification after every single frame
// (the old one-frame-at-a-time backpressure) means every frame pays that ~30ms tax in
// full, which is exactly the "stuck at 30fps" ceiling -- not a kernel bottleneck.
// Allowing a small, bounded number of frames in flight simultaneously (standard
// double/triple buffering) amortizes that fixed latency across multiple frames instead
// of serializing it per-frame, while still bounding the GPU memory backlog (nothing
// pools buffers -- see runGraph's cleanup) far below what caused the earlier OOM crash.
//
// 6, not 3: that ~30ms notification latency has real jitter (we've measured it
// swinging well outside a tight band -- OS scheduling, GC pauses, GPU power-state
// dips all plausibly contribute), and at a 60fps submission rate (16.67ms/frame),
// ceil(30ms / 16.67ms) = 2 is already razor-thin -- 3 had almost no margin for that
// jitter, which is exactly what an irregular, run-to-run-inconsistent stagger looks
// like: fine until jitter happens to exceed the available slots, then one frame's
// request gets queued instead of submitted. 6 covers jitter up to ~100ms. Higher
// would absorb even more jitter, but increases the GPU memory backlog this is
// trading against (each in-flight frame holds real intermediate buffers until the
// driver can reclaim them) -- raise further if staggering persists, but move
// incrementally rather than jumping straight to a much larger number with no
// measurement of this model's actual per-frame footprint.
const MAX_IN_FLIGHT = 6;

// runToGpu()'s own promise resolves on CPU-side submission (~2-7ms), not GPU
// completion -- it doesn't wait for anything GPU-side at all. With multiple frames in
// flight, that means two overlapping requests' .then() callbacks can fire in a
// different order than the frames were requested in (ordinary JS scheduling
// variance, nothing GPU-related), and since each render() call fully overwrites the
// canvas, whichever one calls it *last* wins the screen -- regardless of which `t` is
// actually newer. That's a visible "rewind" stutter, the standard hazard of any
// pipelined renderer with multiple frames in flight and no completion-order
// guarantee. Fixed the standard way: tag every submitted frame with a monotonically
// increasing sequence number, and only let a frame reach the canvas if no newer
// sequence has already claimed it.
let frameSequence = 0;
let latestRenderedSequence = -1;

// Caps how often Play's rAF loop *submits* new frames, independent of the display's
// own refresh rate (which requestAnimationFrame is tied to, and which can be
// 120/144Hz+ on plenty of real hardware) -- with no cap, a high-refresh display would
// submit far more frames than anything downstream can use, for no visual benefit,
// and increase how many frames are simultaneously in flight competing for the same
// MAX_IN_FLIGHT budget (more contention, not more smoothness).
const PLAY_TARGET_FPS = 60;
const PLAY_MIN_FRAME_INTERVAL_MS = 1000 / PLAY_TARGET_FPS;
let lastPlayFrameMs = -Infinity;

const PLAY_SWEEP_MS = 6000; // wall-clock duration of one full t: 0 -> 1 sweep

loadButton.addEventListener("click", () => {
  void load().catch((err: unknown) => {
    log(`ERROR: ${err instanceof Error ? err.message : String(err)}`);
    console.error(err);
    loadButton.disabled = false;
  });
});

verifyButton.addEventListener("click", () => {
  void runVerify().catch((err: unknown) => {
    log(`ERROR: ${err instanceof Error ? err.message : String(err)}`);
    console.error(err);
  });
});

profileButton.addEventListener("click", () => {
  void runProfile().catch((err: unknown) => {
    log(`ERROR: ${err instanceof Error ? err.message : String(err)}`);
    console.error(err);
  });
});

tSlider.addEventListener("input", () => {
  const t = Number(tSlider.value);
  tValueEl.textContent = t.toFixed(3);
  void runAt(t);
});

playButton.addEventListener("click", () => {
  playing = !playing;
  playButton.textContent = playing ? "Pause" : "Play";
  if (playing) {
    playStartMs = performance.now();
    capacityHits = 0;
    requestedFrames = 0;
    requestAnimationFrame(playTick);
  } else if (requestedFrames > 0) {
    // Direct evidence for whether MAX_IN_FLIGHT is actually enough, rather than
    // judging by eye alone -- a capacityHits rate near 0% means the in-flight pool
    // absorbed the GPU-completion-notification jitter without ever blocking a
    // frame's submission; a non-trivial rate means it's still happening and
    // MAX_IN_FLIGHT should go higher.
    log(`Play session: ${capacityHits}/${requestedFrames} requested frames (${((100 * capacityHits) / requestedFrames).toFixed(1)}%) hit the MAX_IN_FLIGHT=${MAX_IN_FLIGHT} cap and had to wait for capacity.`);
  }
});

async function load(): Promise<void> {
  loadButton.disabled = true;
  verifyButton.disabled = true;
  profileButton.disabled = true;
  tSlider.disabled = true;
  playButton.disabled = true;
  const path = pathInput.value.trim();
  logEl.textContent = `loading ${path}...`;
  frameStatusEl.textContent = "";
  canvas.width = 0;
  canvas.height = 0;

  model = await KumaModel.load(path);
  gpuRenderer = new GpuFrameRenderer(model.gpuDevice, canvas);
  log(`loaded. inputs: ${JSON.stringify(model.inputs)}`);
  log(`outputs: ${JSON.stringify(model.outputs)}`);
  verifyButton.disabled = false;

  // Runs every branch once (a no-op if this package has no golden.json) so shader
  // compilation for every kernel every branch could use happens now, during the
  // already-expected load wait, rather than mid-playback the first time scrubbing
  // crosses into a branch that's never been rendered yet -- that was landing as a
  // real, user-visible stutter right at segment boundaries.
  log("warming up every branch's pipelines...");
  const warmUpT0 = performance.now();
  await model.warmUp();
  log(`warm-up done in ${(performance.now() - warmUpT0).toFixed(0)}ms.`);

  const inputSpec = model.inputs[0]!;
  const n = (inputSpec.shape ?? []).reduce((a, b) => a * b, 1);
  if (n === 1) {
    timeInputName = inputSpec.name;
    tSlider.disabled = false;
    playButton.disabled = false;
    profileButton.disabled = false;
    log(`scrubbing input "${inputSpec.name}" over t in [0, 1] via the slider above.`);
    runAt(Number(tSlider.value));
  } else {
    timeInputName = undefined;
    log(`input "${inputSpec.name}" has ${n} elements (not a scalar time input) -- scrubber stays disabled.`);
  }
  loadButton.disabled = false;
}

function formatReport(report: VerifyReport): string {
  const lines: string[] = [`verify: ${report.ok ? "PASS" : "FAIL"} (${report.branches.length} branch(es))`];
  for (const b of report.branches) {
    lines.push(`  branch ${b.branch}: checked ${b.nodesChecked} node(s), ${b.mismatches.length} mismatch(es), ${b.nodesMissing.length} missing`);
    if (b.nodesMissing.length > 0) {
      lines.push(`    missing: ${b.nodesMissing.join(", ")}`);
    }
    for (const m of b.mismatches) {
      lines.push(`    ${m.node} [${m.part}]:`);
      for (const d of m.diffs) {
        lines.push(`      ${d.field}: golden=${d.golden} actual=${d.actual}`);
      }
    }
  }
  return lines.join("\n");
}

async function runVerify(): Promise<void> {
  if (!model) return;
  verifyButton.disabled = true;
  log("running verify()...");
  const t0 = performance.now();
  const report = await model.verify();
  const elapsed = performance.now() - t0;
  log(`verify done in ${elapsed.toFixed(0)}ms:\n${formatReport(report)}`);
  console.log("[kuma] verify report", report);
  verifyButton.disabled = false;
}

function formatProfileReport(report: ProfileReport & { realMilliseconds: number; baselineMilliseconds: number }): string {
  const lines: string[] = [
    `profile: real frame time = ${report.realMilliseconds.toFixed(1)}ms (trust this one -- measured the same way the interactive scrubber is)`,
    `  onSubmittedWorkDone() baseline (nothing newly submitted) = ${report.baselineMilliseconds.toFixed(1)}ms -- if this is`,
    `  a big chunk of real frame time, that's fixed sync-call overhead, not GPU compute.`,
    `  per-op breakdown total = ${(report.totalMicroseconds / 1000).toFixed(2)}ms, summed across ${report.perNode.length} dispatched nodes --`,
    `  ${(report.totalMicroseconds / 1000 / report.realMilliseconds).toFixed(1)}x of real frame time (each node needs its own GPU pass to get individual`,
    `  timing, a WebGPU constraint, not a bug -- use byTarget's percentages below, not its ms values).`,
  ];
  lines.push(`by op class (top 15, sorted by % of per-op breakdown total):`);
  for (const t of report.byTarget.slice(0, 15)) {
    const pct = (100 * t.totalMicroseconds) / report.totalMicroseconds;
    lines.push(
      `  ${t.target.padEnd(32)} ${pct.toFixed(1)}% ` +
        `count=${t.count} avg=${t.avgMicroseconds.toFixed(1)}us`,
    );
  }
  return lines.join("\n");
}

// Isolated profile() clicks have human-paced gaps between them -- long enough for many
// GPUs (especially integrated/laptop ones) to idle down to a low power state in
// between, so a single click can spend much of its time paying to ramp the clock back
// up rather than measuring steady-state performance. Running it several times back to
// back (no human-paced gap between them) keeps the GPU busy throughout, and the later
// runs converge on the same steady-state number actual sustained interactive use (e.g.
// continuous scrubbing/Play) sees -- which is the number that actually matters.
const PROFILE_WARMUP_RUNS = 5;

async function runProfile(): Promise<void> {
  const m = model;
  const name = timeInputName;
  if (!m || !name) return;
  profileButton.disabled = true;
  log(`running profile() x${PROFILE_WARMUP_RUNS} (back to back, to ride out GPU power-state ramp-up between runs)...`);
  const input = new Float32Array([Number(tSlider.value)]);
  let report: (ProfileReport & { realMilliseconds: number; baselineMilliseconds: number }) | undefined;
  const realMilliseconds: number[] = [];
  const baselineMilliseconds: number[] = [];
  for (let i = 0; i < PROFILE_WARMUP_RUNS; i++) {
    report = await m.profile({ [name]: input });
    realMilliseconds.push(report.realMilliseconds);
    baselineMilliseconds.push(report.baselineMilliseconds);
  }
  log(`realMilliseconds across ${PROFILE_WARMUP_RUNS} back-to-back runs: ${realMilliseconds.map((ms) => ms.toFixed(1)).join(", ")}`);
  log(`baselineMilliseconds (onSubmittedWorkDone with nothing pending) across the same runs: ${baselineMilliseconds.map((ms) => ms.toFixed(1)).join(", ")}`);
  log(formatProfileReport(report!));
  console.log("[kuma] profile report (last run)", report);
  profileButton.disabled = false;
}

/** Runs the model at time `t` and renders the result. Up to MAX_IN_FLIGHT frames can be
 * submitted without waiting for any of them to finish (see MAX_IN_FLIGHT's comment for
 * why) -- calls beyond that capacity (fast slider drags, or Play's rAF loop outrunning
 * the GPU) don't queue up, only the most recently requested `t` gets picked up next
 * once capacity frees, same as a video player dropping frames instead of falling
 * behind. */
function runAt(t: number): void {
  const m = model;
  const name = timeInputName;
  if (!m || !name) return;
  requestedFrames++;
  if (inFlight >= MAX_IN_FLIGHT) {
    capacityHits++;
    pendingT = t;
    return;
  }
  pendingT = undefined;
  inFlight++;
  const sequence = frameSequence++;

  const input = new Float32Array([t]);
  const t0 = performance.now();
  // runToGpu (not run/runRaw) -- skips the GPU->CPU readback for the output entirely.
  // The renderer reads pixels straight from the GPU buffer. Use Profile/Verify for
  // value-level stats (finite/bad/mean/min/max); this path only has timing.
  void m
    .runToGpu({ [name]: input })
    .then((outputs) => {
      // This frame's own promise resolving doesn't mean it's still the newest one --
      // see frameSequence's comment. Drop it rather than rewind the display.
      if (sequence > latestRenderedSequence) {
        latestRenderedSequence = sequence;
        for (const out of outputs) {
          gpuRenderer?.render(out.buffer, out.shape);
        }
        const elapsed = performance.now() - t0;
        if (elapsed > STALL_THRESHOLD_MS) {
          console.warn(`[kuma] runToGpu() spike: ${elapsed.toFixed(1)}ms enqueue (usually ~2-7ms) for t=${t.toFixed(3)}`);
        }
        if (outputs[0]) {
          frameStatusEl.textContent =
            `t=${t.toFixed(3)} (${elapsed.toFixed(1)}ms enqueue, ${inFlight} in flight) output "${outputs[0].name}": ` +
            `shape=${JSON.stringify(outputs[0].shape)} (GPU-direct render; use Profile/Verify for value stats)`;
        }
      }
      // Not awaited inline -- that would put this frame right back to fully
      // serialized, one-at-a-time submission, the exact ~30ms-per-frame tax this
      // in-flight scheme exists to amortize. Tracked asynchronously purely to know
      // when capacity frees up for whatever's waiting in pendingT.
      return m.gpuDevice.queue.onSubmittedWorkDone();
    })
    .catch((err: unknown) => {
      log(`ERROR at t=${t.toFixed(3)}: ${err instanceof Error ? err.message : String(err)}`);
      console.error(err);
      playing = false;
      playButton.textContent = "Play";
    })
    .finally(() => {
      inFlight--;
      if (pendingT !== undefined) {
        const next = pendingT;
        pendingT = undefined;
        runAt(next);
      }
    });
}

function playTick(nowMs: number): void {
  if (!playing) return;
  // requestAnimationFrame itself can't fire while the main thread is busy with
  // something else (a GC pause, or anything else blocking synchronously) -- a gap much
  // wider than a normal frame interval here means the *entire* main thread stalled,
  // not just this rendering path specifically.
  if (Number.isFinite(lastTickMs) && nowMs - lastTickMs > STALL_THRESHOLD_MS) {
    console.warn(`[kuma] main-thread stall: ${(nowMs - lastTickMs).toFixed(1)}ms between rAF callbacks`);
  }
  lastTickMs = nowMs;
  // Gates the slider position update too, not just runAt -- keeps the displayed t and
  // the actually-rendered frame in sync, rather than a slider that glides smoothly
  // while the image it's labeling updates at a slower, capped rate.
  if (nowMs - lastPlayFrameMs >= PLAY_MIN_FRAME_INTERVAL_MS) {
    lastPlayFrameMs = nowMs;
    const t = ((nowMs - playStartMs) % PLAY_SWEEP_MS) / PLAY_SWEEP_MS;
    tSlider.value = String(t);
    tValueEl.textContent = t.toFixed(3);
    runAt(t);
  }
  requestAnimationFrame(playTick);
}
