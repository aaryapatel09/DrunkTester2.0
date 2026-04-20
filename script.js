// DrunkTester 2.0 — browser-only impairment screener.
//
// Three tests, all client-side. Nothing uploaded.
//   1. Reaction time — 5 trials of click-when-green
//   2. Gaze stability — MediaPipe FaceLandmarker iris tracking, 10 s
//   3. Speech smoothness — Web Speech API, WER vs prompt + read duration
//
// A composite score is computed against a baseline the user records while sober.
// Without a baseline, raw numbers are still shown but the composite stays hidden.

import {
  FilesetResolver,
  FaceLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const PROMPT = document.getElementById("sp-prompt").textContent.trim();
const BASELINE_KEY = "dt2.baseline.v1";

const state = {
  rt: null,   // { mean, sd }
  gaze: null, // { jitter }
  speech: null, // { wer, duration, words }
  baseline: JSON.parse(localStorage.getItem(BASELINE_KEY) || "null"),
};

function saveBaseline(b) {
  localStorage.setItem(BASELINE_KEY, JSON.stringify(b));
  state.baseline = b;
  renderBaselineBanner();
  renderComposite();
}

function renderBaselineBanner() {
  const el = document.getElementById("baselineBanner");
  if (!state.baseline) {
    el.classList.remove("hidden");
    return;
  }
  el.classList.add("hidden");
  // Once a baseline exists, show when it was recorded next to the Save button
  // so users notice stale baselines (e.g. a 3-month-old reading is suspect).
  const host = document.getElementById("saveBaseline");
  if (host && state.baseline.at) {
    const days = Math.floor((Date.now() - state.baseline.at) / 86_400_000);
    const when = days === 0 ? "today" : `${days} day${days === 1 ? "" : "s"} ago`;
    host.textContent = `Save current results as baseline (current: recorded ${when})`;
  }
}

renderBaselineBanner();

// ============================================================================
// Test 1 — reaction time
// ============================================================================
const rtStage = document.getElementById("rt-stage");
const rtMsg = document.getElementById("rt-msg");
const rtMetrics = document.getElementById("rt-metrics");

const RT_TRIALS = 5;
let rtRun = { trials: [], idx: 0, waiting: false, goTime: 0, timer: null };

function startRt() {
  rtRun = { trials: [], idx: 0, waiting: false, goTime: 0, timer: null };
  rtMetrics.innerHTML = "";
  nextTrial();
}

function nextTrial() {
  if (rtRun.idx >= RT_TRIALS) {
    finishRt();
    return;
  }
  rtRun.idx++;
  rtStage.className = "rt-stage waiting";
  rtMsg.textContent = `Trial ${rtRun.idx}/${RT_TRIALS} — wait for green…`;
  rtRun.waiting = true;
  const delay = 1000 + Math.random() * 2500;
  rtRun.timer = setTimeout(() => {
    rtStage.className = "rt-stage go";
    rtMsg.textContent = "GO — click!";
    rtRun.goTime = performance.now();
    rtRun.waiting = false;
  }, delay);
}

rtStage.addEventListener("click", () => {
  if (rtRun.timer === null && rtRun.idx === 0) {
    startRt();
    return;
  }
  if (rtRun.waiting) {
    clearTimeout(rtRun.timer);
    rtStage.className = "rt-stage too-early";
    rtMsg.textContent = "Too early — restarting trial.";
    rtRun.waiting = false;
    rtRun.idx--;
    setTimeout(nextTrial, 900);
    return;
  }
  if (rtRun.goTime) {
    const rt = performance.now() - rtRun.goTime;
    rtRun.trials.push(rt);
    rtRun.goTime = 0;
    rtStage.className = "rt-stage";
    rtMsg.textContent = `${rt.toFixed(0)} ms — next trial…`;
    setTimeout(nextTrial, 700);
  }
});

function finishRt() {
  const ts = rtRun.trials;
  const mean = ts.reduce((a, b) => a + b, 0) / ts.length;
  const sd = Math.sqrt(ts.reduce((a, b) => a + (b - mean) ** 2, 0) / ts.length);
  state.rt = { mean, sd };
  rtStage.className = "rt-stage";
  rtMsg.textContent = "Done. Click to redo.";
  rtRun.timer = null;
  rtRun.idx = 0;
  renderRtMetrics();
  renderComposite();
}

function renderRtMetrics() {
  if (!state.rt) return;
  const { mean, sd } = state.rt;
  const base = state.baseline?.rt;
  rtMetrics.innerHTML = `
    ${metricHtml("Mean RT", mean.toFixed(0) + " ms", base && delta((mean - base.mean) / base.mean, "up"))}
    ${metricHtml("SD", sd.toFixed(0) + " ms", base && delta((sd - base.sd) / Math.max(1, base.sd), "up"))}
    ${metricHtml("Trials", state.rt ? RT_TRIALS : 0)}
  `;
}

// ============================================================================
// Test 2 — gaze stability via MediaPipe FaceLandmarker
// ============================================================================
const gzVideo = document.getElementById("gz-video");
const gzCanvas = document.getElementById("gz-overlay");
const gzCtx = gzCanvas.getContext("2d");
const gzMsg = document.getElementById("gz-msg");
const gzStartBtn = document.getElementById("gz-start");
const gzStopBtn = document.getElementById("gz-stop");
const gzMetrics = document.getElementById("gz-metrics");

let faceLm = null;
let gzStream = null;
let gzRaf = null;
let gzSamples = [];
let gzT0 = 0;
const GAZE_DURATION_MS = 10_000;

async function initFaceLm() {
  gzMsg.textContent = "Loading model…";
  const fileset = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  faceLm = await FaceLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate: "GPU",
    },
    outputFaceBlendshapes: false,
    runningMode: "VIDEO",
    numFaces: 1,
  });
}

async function startGaze() {
  gzStartBtn.disabled = true;
  try {
    if (!faceLm) await initFaceLm();
    gzStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: "user" }, audio: false });
    gzVideo.srcObject = gzStream;
    await new Promise((r) => (gzVideo.onloadedmetadata = r));
    gzCanvas.width = gzVideo.videoWidth;
    gzCanvas.height = gzVideo.videoHeight;
    gzSamples = [];
    gzT0 = performance.now();
    gzStopBtn.disabled = false;
    gzMsg.textContent = "Stare at the centre dot…";
    gazeLoop();
  } catch (e) {
    gzMsg.textContent = "Error: " + e.message;
    gzStartBtn.disabled = false;
  }
}

function stopGaze(finished = false) {
  if (gzRaf) cancelAnimationFrame(gzRaf);
  gzRaf = null;
  if (gzStream) gzStream.getTracks().forEach((t) => t.stop());
  gzStream = null;
  gzVideo.srcObject = null;
  gzCtx.clearRect(0, 0, gzCanvas.width, gzCanvas.height);
  gzStartBtn.disabled = false;
  gzStopBtn.disabled = true;
  gzMsg.textContent = finished ? "Done." : "Stopped";
  if (finished) finishGaze();
}

function gazeLoop() {
  const t = performance.now();
  if (t - gzT0 > GAZE_DURATION_MS) {
    stopGaze(true);
    return;
  }
  if (gzVideo.readyState >= 2) {
    const res = faceLm.detectForVideo(gzVideo, t);
    if (res.faceLandmarks && res.faceLandmarks.length) {
      const lm = res.faceLandmarks[0];
      // MediaPipe iris centers: left 468, right 473 (when output_face_blendshapes
      // is off the default 478-point mesh still includes iris points).
      const left = lm[468] || lm[159];
      const right = lm[473] || lm[386];
      if (left && right) {
        const cx = (left.x + right.x) / 2;
        const cy = (left.y + right.y) / 2;
        const eyeDist = Math.hypot(right.x - left.x, right.y - left.y);
        gzSamples.push({ x: cx, y: cy, eyeDist });
        drawGazeMarkers(left, right);
      }
    }
    gzMsg.textContent = `Stare at centre… ${((GAZE_DURATION_MS - (t - gzT0)) / 1000).toFixed(1)} s`;
  }
  gzRaf = requestAnimationFrame(gazeLoop);
}

function drawGazeMarkers(l, r) {
  gzCtx.clearRect(0, 0, gzCanvas.width, gzCanvas.height);
  gzCtx.fillStyle = "#6aa8ff";
  for (const p of [l, r]) {
    gzCtx.beginPath();
    gzCtx.arc(p.x * gzCanvas.width, p.y * gzCanvas.height, 4, 0, Math.PI * 2);
    gzCtx.fill();
  }
}

function finishGaze() {
  if (gzSamples.length < 10) {
    gzMetrics.innerHTML = `<div class="metric"><div class="k">No face detected</div></div>`;
    return;
  }
  // Normalise jitter by the inter-ocular distance so head distance from the
  // camera cancels out. The previous version computed raw radial SD in
  // normalised image coords — the comment claimed "mean-eye-distance units"
  // but the code didn't actually divide.
  const mx = gzSamples.reduce((a, s) => a + s.x, 0) / gzSamples.length;
  const my = gzSamples.reduce((a, s) => a + s.y, 0) / gzSamples.length;
  const meanEyeDist = gzSamples.reduce((a, s) => a + s.eyeDist, 0) / gzSamples.length;
  const rs = gzSamples.map((s) => Math.hypot(s.x - mx, s.y - my));
  const rawJitter = Math.sqrt(rs.reduce((a, r) => a + r * r, 0) / rs.length);
  const jitter = rawJitter / Math.max(1e-6, meanEyeDist);

  // Also flag head drift: if the user moved their head during the test the
  // inflated jitter is noise, not nystagmus. eyeDist standard deviation is a
  // decent proxy for depth wobble; mean-of-samples drift catches pan/tilt.
  const eyeDistSd = Math.sqrt(
    gzSamples.reduce((a, s) => a + (s.eyeDist - meanEyeDist) ** 2, 0) / gzSamples.length
  );
  const depthWobble = eyeDistSd / Math.max(1e-6, meanEyeDist);

  state.gaze = { jitter, depthWobble, samples: gzSamples.length };
  renderGazeMetrics();
  renderComposite();
}

function renderGazeMetrics() {
  if (!state.gaze) return;
  const { jitter, depthWobble, samples } = state.gaze;
  const base = state.baseline?.gaze;
  const headWarn = depthWobble > 0.05
    ? `<div class="metric" style="grid-column: 1 / -1"><div class="k">Heads up</div><div class="v" style="font-size:13px">Head moved during test; rerun while holding still for a cleaner reading.</div></div>`
    : "";
  gzMetrics.innerHTML = `
    ${metricHtml("Iris jitter", jitter.toFixed(4), base && delta((jitter - base.jitter) / Math.max(1e-6, base.jitter), "up"))}
    ${metricHtml("Head motion", (depthWobble * 100).toFixed(1) + "%")}
    ${metricHtml("Samples", samples)}
    ${headWarn}
  `;
}

gzStartBtn.addEventListener("click", startGaze);
gzStopBtn.addEventListener("click", () => stopGaze(false));

// ============================================================================
// Test 3 — speech smoothness via Web Speech API
// ============================================================================
const spStart = document.getElementById("sp-start");
const spStop = document.getElementById("sp-stop");
const spLive = document.getElementById("sp-live");
const spMetrics = document.getElementById("sp-metrics");

let recog = null;
let spT0 = 0;

function supportsSpeech() {
  return "webkitSpeechRecognition" in window || "SpeechRecognition" in window;
}

spStart.addEventListener("click", () => {
  if (!supportsSpeech()) {
    spLive.textContent = "Web Speech API not supported in this browser (try Chrome or Edge).";
    return;
  }
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  recog = new SR();
  recog.lang = "en-US";
  recog.interimResults = true;
  recog.continuous = true;
  let finalText = "";
  recog.onresult = (e) => {
    let interim = "";
    for (let i = e.resultIndex; i < e.results.length; i++) {
      const t = e.results[i][0].transcript;
      if (e.results[i].isFinal) finalText += t;
      else interim += t;
    }
    spLive.textContent = (finalText + " " + interim).trim();
  };
  recog.onerror = (e) => {
    const human = {
      "not-allowed": "Microphone permission denied. Allow mic access in your browser and try again.",
      "no-speech": "No speech detected — try closer to the mic and re-run.",
      "audio-capture": "No microphone found.",
      "network": "Speech recognition needs internet access and it failed to reach the service.",
      "aborted": "Recording aborted.",
    }[e.error] || `Speech error: ${e.error}`;
    spLive.textContent = human;
  };
  recog.onend = () => {
    spStart.disabled = false;
    spStop.disabled = true;
    if (!finalText.trim()) return;
    const duration = (performance.now() - spT0) / 1000;
    const wer = computeWer(PROMPT, finalText);
    state.speech = { wer, duration, words: finalText.split(/\s+/).filter(Boolean).length };
    renderSpeechMetrics();
    renderComposite();
  };
  spT0 = performance.now();
  spLive.textContent = "Listening… read the sentence now.";
  spStart.disabled = true;
  spStop.disabled = false;
  recog.start();
});

spStop.addEventListener("click", () => recog && recog.stop());

function renderSpeechMetrics() {
  if (!state.speech) return;
  const { wer, duration, words } = state.speech;
  const base = state.baseline?.speech;
  spMetrics.innerHTML = `
    ${metricHtml("Word error rate", (wer * 100).toFixed(1) + "%", base && delta(wer - base.wer, "up", { abs: true }))}
    ${metricHtml("Read duration", duration.toFixed(2) + " s", base && delta((duration - base.duration) / Math.max(1, base.duration), "up"))}
    ${metricHtml("Words heard", words)}
  `;
}

function computeWer(expected, actual) {
  const exp = normalizeWords(expected);
  const act = normalizeWords(actual);
  if (exp.length === 0) return 0;
  const dist = levenshtein(exp, act);
  return dist / exp.length;
}

function normalizeWords(s) {
  return s.toLowerCase().replace(/[^a-z0-9\s']/g, "").split(/\s+/).filter(Boolean);
}

function levenshtein(a, b) {
  const m = a.length, n = b.length;
  const dp = Array.from({ length: m + 1 }, () => new Array(n + 1));
  for (let i = 0; i <= m; i++) dp[i][0] = i;
  for (let j = 0; j <= n; j++) dp[0][j] = j;
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      dp[i][j] = a[i - 1] === b[j - 1]
        ? dp[i - 1][j - 1]
        : 1 + Math.min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]);
    }
  }
  return dp[m][n];
}

// ============================================================================
// Composite + baseline management
// ============================================================================
function renderComposite() {
  renderRtMetrics();
  renderGazeMetrics();
  renderSpeechMetrics();
  const el = document.getElementById("comp-num");
  const band = document.getElementById("comp-band");
  if (!state.baseline || !state.rt || !state.gaze || !state.speech) {
    el.textContent = "—";
    band.textContent = state.baseline ? "Run all three tests" : "Needs a baseline";
    band.className = "composite-band none";
    return;
  }
  const rtΔ = rel(state.rt.mean, state.baseline.rt.mean);
  const gzΔ = rel(state.gaze.jitter, state.baseline.gaze.jitter);
  const spΔ = Math.max(0, state.speech.wer - state.baseline.speech.wer);
  // Weights: RT slowdown 0.4, gaze jitter 0.3, speech WER increase 0.3
  // Cap each at +200% to avoid one noisy dimension dominating.
  const cap = (x) => Math.min(Math.max(x, 0), 2);
  const score = 100 * (0.4 * cap(rtΔ) + 0.3 * cap(gzΔ) + 0.3 * cap(spΔ * 4));
  const rounded = Math.round(score);
  el.textContent = rounded;
  let bandName;
  if (score < 15) { bandName = "low"; band.textContent = "close to baseline"; }
  else if (score < 40) { bandName = "mod"; band.textContent = "elevated"; }
  else { bandName = "high"; band.textContent = "high — do not drive"; }
  band.className = "composite-band " + bandName;

  // Persist this run to history — but only once per test set so a simple
  // rerender doesn't add duplicates. The dedupe key is the tuple of the raw
  // metric values; saving a baseline or editing inputs restarts the test
  // anyway, so this is naturally unique per completed session.
  recordSession({
    at: Date.now(),
    score: rounded,
    band: bandName,
    rt_mean: state.rt.mean,
    rt_sd: state.rt.sd,
    gaze_jitter: state.gaze.jitter,
    speech_wer: state.speech.wer,
    baseline_at: state.baseline.at ?? null,
  });
}

function rel(now, base) { return (now - base) / Math.max(1e-6, base); }

document.getElementById("saveBaseline").addEventListener("click", () => {
  if (!state.rt || !state.gaze || !state.speech) {
    alert("Run all three tests first.");
    return;
  }
  saveBaseline({ rt: state.rt, gaze: state.gaze, speech: state.speech, at: Date.now() });
  alert("Baseline saved.");
});

document.getElementById("resetTests").addEventListener("click", () => {
  state.rt = null; state.gaze = null; state.speech = null;
  document.getElementById("rt-metrics").innerHTML = "";
  document.getElementById("gz-metrics").innerHTML = "";
  document.getElementById("sp-metrics").innerHTML = "";
  document.getElementById("sp-live").textContent = "";
  rtMsg.textContent = "Click to start";
  gzMsg.textContent = "Click to start camera";
  renderComposite();
});

document.getElementById("baselineClear").addEventListener("click", () => {
  localStorage.removeItem(BASELINE_KEY);
  state.baseline = null;
  renderBaselineBanner();
  renderComposite();
});

// ============================================================================
// Session history
// ============================================================================
const HISTORY_KEY = "dt2.history.v1";
const HISTORY_MAX = 50;
const historyCard = document.getElementById("historyCard");
const historyList = document.getElementById("historyList");
const historyChart = document.getElementById("historyChart");
let lastSessionKey = null;

function loadHistory() {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveHistory(list) {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(list.slice(-HISTORY_MAX)));
}

function recordSession(s) {
  // Dedupe: the same test data shouldn't get recorded twice just because a
  // render fires (e.g. the user changes the auto-append checkbox). The
  // signature is the tuple of raw metric values; any real new test run
  // produces different numbers.
  const key = [s.rt_mean.toFixed(3), s.gaze_jitter.toFixed(6), s.speech_wer.toFixed(4)].join("|");
  if (key === lastSessionKey) return;
  lastSessionKey = key;

  const list = loadHistory();
  list.push(s);
  saveHistory(list);
  renderHistory();
}

function renderHistory() {
  const list = loadHistory();
  if (list.length === 0) {
    historyCard.style.display = "none";
    return;
  }
  historyCard.style.display = "block";

  // Chart — plot score over time as a polyline. X axis is session index
  // (evenly spaced) rather than real time, so a 4-session-in-one-night run
  // reads cleanly; real times are in the tooltip/list below.
  const W = 600, H = 160, PAD = 8;
  const plotW = W - 2 * PAD, plotH = H - 2 * PAD;
  const max = Math.max(50, ...list.map((s) => s.score));
  const points = list.map((s, i) => {
    const x = list.length === 1 ? W / 2 : PAD + (i / (list.length - 1)) * plotW;
    const y = PAD + (1 - s.score / max) * plotH;
    return [x, y, s];
  });
  const pathD = points.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ");
  const dots = points.map(([x, y, s]) => {
    const color = s.band === "high" ? "#e96b6b" : s.band === "mod" ? "#e5a23b" : "#46c08a";
    return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="3.5" fill="${color}"><title>${s.score} (${s.band}) — ${new Date(s.at).toLocaleString()}</title></circle>`;
  }).join("");
  // Faint guide line at the "elevated" threshold (40)
  const guideY = PAD + (1 - 40 / max) * plotH;
  historyChart.innerHTML = `
    <line x1="${PAD}" x2="${W - PAD}" y1="${guideY.toFixed(1)}" y2="${guideY.toFixed(1)}" stroke="#232834" stroke-dasharray="3,3"/>
    <path d="${pathD}" fill="none" stroke="#6aa8ff" stroke-width="1.5"/>
    ${dots}
  `;

  // List — newest first
  historyList.innerHTML = list.slice().reverse().map((s) => {
    const when = new Date(s.at);
    const whenStr = when.toLocaleString(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
    return `<li>
      <span class="when">${escapeHtml(whenStr)}</span>
      <span class="band composite-band ${escapeHtml(s.band)}">${escapeHtml(s.band)}</span>
      <span class="score">${s.score}</span>
    </li>`;
  }).join("");
}

document.getElementById("clearHistory").addEventListener("click", () => {
  if (!confirm("Delete all stored sessions? This can't be undone.")) return;
  localStorage.removeItem(HISTORY_KEY);
  lastSessionKey = null;
  renderHistory();
});

renderHistory();

// ============================================================================
// Tiny render helpers
// ============================================================================
function metricHtml(k, v, d = "") {
  return `<div class="metric"><div class="k">${escapeHtml(k)}</div><div class="v">${escapeHtml(String(v))}</div>${d ? `<div class="d ${d.cls || ""}">${d.text}</div>` : ""}</div>`;
}

function delta(ratio, dirBad, opts = {}) {
  if (!isFinite(ratio)) return null;
  const pct = opts.abs ? ratio * 100 : ratio * 100;
  const sign = pct >= 0 ? "+" : "";
  const worse = (dirBad === "up" && pct > 0) || (dirBad === "down" && pct < 0);
  return { text: `${sign}${pct.toFixed(1)}% vs baseline`, cls: worse ? "up" : "down" };
}

function escapeHtml(s) { return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]); }

renderComposite();
