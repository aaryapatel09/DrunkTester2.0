# DrunkTester 2.0

Browser-only impairment screener. Three short tests that measure real signals correlated with intoxication, compared against a sober baseline you record yourself.

> **This is not a breathalyzer.** It cannot measure blood alcohol. It measures proxies. Do not use it to decide whether to drive.

## How it works

Three tests — all client-side, no uploads:

1. **Reaction time** — 5 trials of click-when-green. Reports mean RT and SD.
   Alcohol reliably increases both ([Maylor & Rabbitt, 1993](https://pubmed.ncbi.nlm.nih.gov/8234536/); many since).
2. **Gaze stability** — MediaPipe FaceLandmarker tracks your iris position for 10 seconds while you stare at a fixed dot. Jitter is reported as the radial standard deviation of iris centroid position in image coordinates. Intoxication degrades smooth pursuit and saccadic control.
3. **Speech smoothness** — You read a fixed sentence aloud. The browser's Web Speech API transcribes you; we compute the word error rate against the prompt plus total read duration. Slurred or hesitant speech pushes both up.

A **composite score** combines the three as weighted percentage deltas from your baseline:
- RT mean (40%)
- gaze jitter (30%)
- speech WER increase (30%, weighted 4× because WER saturates low)

Each dimension is capped at +200% so one noisy signal can't dominate.

## Run it

You need a browser that supports:
- `navigator.mediaDevices.getUserMedia` (any modern browser)
- MediaPipe Tasks Vision WASM (Chrome, Edge, Firefox, Safari 16.4+)
- `SpeechRecognition` (Chrome / Edge — Firefox and Safari lack it, so the speech test will show a warning there)

Serve the files over `http://localhost` or HTTPS (required for getUserMedia):

```bash
python3 -m http.server 8000
# open http://localhost:8000
```

## Calibration workflow

1. **Record a sober baseline.** Run all three tests while sober, click **Save current results as baseline.** Baseline is stored in `localStorage`, never leaves the device.
2. **Later, retake the tests.** Each metric shows its raw value plus `±X% vs baseline`, and the composite score summarises.

A single sober run is noisy — for a real baseline, run it 3–5 times and average mentally, or extend the script to store multiple baselines.

## Honest limits

- Reaction-time test measures **visuomotor RT**, which is affected by fatigue, caffeine, screen lag, and mouse/trackpad latency — not just alcohol.
- Gaze-jitter metric depends on head stability; if you move your head during the test, jitter rises regardless of intoxication.
- Web Speech API transcription quality varies by accent and microphone. A high WER might mean your mic is bad, not that you're drunk.
- No peer-reviewed threshold maps these proxies to BAC. The "elevated / high" bands are heuristic, not clinical.

If you want to harden this, the right next step is a supervised model trained on (proxy metrics → actual BAC measurements) pairs — which requires data this project does not have.
