# SignBridge Backlog

Open feature work for SignBridge. Items are captured verbatim from the
working-session list before prioritization. Sub-bullets unpack each item
into the smaller pieces it likely decomposes into; mark them off as they
ship.

Status legend: `[ ]` = open, `[~]` = in progress, `[x]` = done.

---

## 1. Three modes under the Live tab: Kiosk, Lanyard, Conf Call

Mode switcher inside the Live tab. **Hard rule for all modes:** zero
changes to current working pipeline -- camera/sensor/zoom/inference
settings, conversation flow, session lifecycle, send/regenerate/edit,
CTQI badge, etc. are preserved exactly as tested. Each mode is a thin
wrapper that toggles a few UI knobs around the unchanged pipeline.

### 1a. Kiosk mode (agreed spec)

- Devices: 1x iPad (built-in camera, mic, speaker; or headphones)
- Roles: deaf customer signs to the iPad; hearing staff speaks to the
  same iPad
- Video: iPad built-in camera
- Audio in: iPad mic or headphone mic
- Audio out (TTS): per-device Settings toggle ("Enable TTS playback");
  off = staff reads from screen, on = synth voice through iPad /
  headphone speaker
- I Sign / I Speak toggle: hidden in this mode
- Screen, conversation flow, session lifecycle: unchanged

### 1b. Lanyard mode (agreed spec)

Equivalent to Kiosk but with the WiFi cam as the video source.

- Devices: 1x shared iPad + 1x ESP32 WiFi cam on wearer's lanyard +
  USB power bank in wearer's pocket
- Roles: same as Kiosk; signer in front of wearer, both share the
  iPad for transcript + audio
- Video: WiFi cam stream (existing camera-proxy path)
- Audio in: iPad mic or headphone mic (NOT the lanyard mic; that is
  separate item 2)
- Audio out (TTS): per-device Settings toggle (same as Kiosk)
- I Sign / I Speak toggle: hidden
- Screen, conversation flow, session lifecycle: unchanged
- **Selecting Lanyard auto-opens the WiFi camera URL prompt** if not
  already connected (decided in mode-definition review)
- Deferred for V1: on-board lanyard mic (item 2), signer-side alert
  UX (item 5)

### 1c. Conf Call mode -- DEFERRED to P5

Definition postponed until P0 ships. Conf Call involves remote
participants, existing video-call apps (Zoom/Meet/Teams?), and more
moving pieces than the other two modes. Better to define after the
shared mode infrastructure exists.

### Implementation checklist (P0: Kiosk + Lanyard only) — **SHIPPED**

- [x] Mode selector UI inside Live tab (Kiosk / Lanyard; Conf Call
      placeholder disabled-until-built) — `app.js:1157-1228`,
      `templates/index.html:72-75`
- [x] Selected mode persisted in localStorage as `signbridge.mode`;
      returning user lands in their last-used mode — `app.js:1162`
- [x] Kiosk mode behavior: hide I Sign / I Speak, default camera =
      built-in — `app.js:1199, 1204-1208`
- [x] Lanyard mode behavior: hide I Sign / I Speak, default camera =
      WiFi Camera (IP), auto-open URL prompt on mode select if not
      already connected — `app.js:1209-1220`
- [x] "Enable TTS playback" toggle in Settings (per-device, defaults:
      off for Lanyard, off for Kiosk -- staff reads, opt-in for TTS) —
      `templates/index.html:396-405`, `app.js:1966`

## 2. Get the microphone working on the WiFi cam

Use the Knowles PDM mic on the XIAO ESP32-S3 Sense expansion board so
the lanyard wearer's speech can be captured from the device on their
chest rather than the laptop / phone microphone.

- [ ] Firmware: PDM driver, capture loop, encode (PCM or Opus)
- [ ] Firmware: new `/audio` endpoint (WebSocket, streaming chunks)
- [ ] Flask: consume `/audio` from the board, push frames into the
      same STT pipeline currently fed by Web Speech API
- [ ] Decide STT engine for server-side path (Whisper local? Vosk?
      cloud Whisper API?). Web Speech API only works in-browser.
- [ ] SignBridge UX: mic source selector ("device mic" vs "WiFi cam
      mic") in settings; auto-pick WiFi cam mic when in Lanyard mode

## 3. Sample videos for each sign in Sign Bank, accessible without the external drive

Sign-bank videos currently live on an external drive that won't be
attached during real use. Need a storage solution that works for the
demo and beyond.

**Shipped:**
- [x] Decide storage: **Cloudflare R2** (S3-compatible, ~$0/mo at our
      scale, public-bucket reads, signer-12 preferred per migration
      script). Local-bundle fallback kept for dev.
- [x] Migration: WLASL clips moved to R2 via
      `project-utilities/migrate_wlasl_to_r2.py`. R2 base URL:
      `pub-356ccbb34efd4ada8ba91c03fe330713.r2.dev`.
- [x] App: sign-bank fetch path resolves `PREFER_LOCAL_SIGN_BANK` first,
      then R2 (`_gloss_to_r2_url`) — `app.py:74, 1764-1773, 2046-2051`.
- [x] App: `/sign-bank/<filename>` route serves local files
      (`app.py:3666`); R2 URLs returned directly to client otherwise.

**Close-out (P0 — small remaining tasks):**
- [ ] Coverage audit: list glosses present in each active domain's
      active-class set vs glosses present in R2. Output a CSV of
      gloss × {local, r2, missing}. Decide whether to backfill missing
      ones or accept "no sample" UI for those.
- [ ] Graceful UI when a sign-bank video 404s in the client (sign-bank
      panel currently assumes the video loads). One-line `onerror`
      handler that shows "no sample available" placeholder.
- [ ] Document R2 cost / latency / offline tradeoff in a short
      `docs/sign-bank-storage.md` (3 paragraphs, decision rationale +
      cutover steps if we need to move).

## 4. Per-mode setup wizards (audio-in, audio-out, video-in, layout, devices)

For each of the three Live modes, work out:
- Audio input source(s) (device mic / WiFi-cam mic / phone mic)
- Audio output target (laptop speaker / phone / external)
- Video input source (device webcam / WiFi cam / multiple)
- Preview display layout (kiosk full-screen / lanyard transcript-only
  / conf-call PiP grid)
- Number of devices physically in the setup (1 vs N)
- Whether "I Sign" / "I Speak" mode toggle is still meaningful per mode

**Largely shipped via 1a/1b:** mode selector persists the active mode,
each mode defaults its own camera, hides I Sign / I Speak, and surfaces
the relevant TTS toggle. No separate "wizard" is needed — defaults
land users in a usable state on first entry.

**Close-out (P0 — small remaining tasks):**
- [ ] **Settings page filters by active mode**: hide the speaker-side
      knobs (TTS, wake word, chime) when the active mode doesn't use
      them, and hide the WiFi-cam config when mode != lanyard. Audit
      `templates/index.html` Settings sections.
- [ ] **Pre-flight validation**: when entering Lanyard mode, if the
      WiFi cam URL is empty / unreachable, show an inline banner
      ("Lanyard mode needs the WiFi camera connected — [Set URL]")
      instead of letting Start Conversation fail silently.
- [ ] **Per-mode TTS default**: confirm Lanyard defaults to TTS-on
      (signer can't see screen so audio is required) and Kiosk defaults
      to TTS-off (staff reads screen). Currently both default to off
      per `app.js:31` — needs a mode-aware default in `_loadTtsToggle`.

**Decided out of scope:**
- Per-mode separately-keyed localStorage for every setting (current
  global-per-device storage is sufficient — users don't switch modes
  often enough to warrant divergent prefs)
- Separate "I Sign" / "I Speak" toggle per mode (both modes hide it;
  decision baked into 1a/1b)
- First-run walkthrough wizard (defaults are good enough)

## 5. Signer alert when CTQI is low (regenerate / edit needed) -- visible from the lanyard

The CTQI score's value proposition is acted on by the signer when it's
low. In Lanyard mode the signer can't see the laptop screen, so the
alert needs to surface where the signer can perceive it.

**Partially shipped:** the hearing-person side already gets an audible
chime (app.js:4719-4749) + pulsing red border (app.js:4850) when CTQI
drops below the threshold. The signer-facing channel below is still open.

- [ ] Decide signer-facing alert channel: phone vibrate? phone screen?
      a smartwatch ping? an audible cue the hearing person hears and
      relays? a small LED / haptic add-on on the lanyard?
- [ ] Implement the alert path (whatever channel wins above)
- [x] Threshold + cooldown — already in place for the chime
- [ ] Make the regenerate / edit action reachable from the signer's
      side, not just the hearing person's UI
- [ ] Optional: a one-tap "I said that wrong, try again" gesture for
      the signer to manually trigger regenerate

## 6. Graceful failure when predictions are wrong

Connected to 5. The system must degrade well, not silently mislead.

**Mostly shipped:**
- [x] Visible confidence indicator on every emitted caption — CTQI
      badge always rendered (app.js:4845-4850)
- [x] When confidence is below the hard floor, show "[unclear]"
      instead of guessing — CTQI_HARD_FLOOR default 40, render at
      app.js:4850
- [x] Always-available "this was wrong" affordance for the hearing
      person to correct in-place (text edit) — present across live and
      upload tabs (templates/index.html:212, upload.js paragraph edit)

**Still open (P2):**
- [ ] Log failure cases (with consent) for offline analysis
- [ ] In Kiosk mode specifically: a "tap to repeat" prompt back to the
      signer when the caption is rejected or edited

## 7. A clear, unified Settings tab

The current settings popup exists but is buried inside the Live tab
cog. Promote to a top-level destination with all knobs in one place.

- [x] New top-level Settings tab in the main nav
- [x] Sections (V1 shipped): Audio, Advanced. Camera + signing-zone
      stay in the in-session cog popup since they're typically tuned
      mid-session, not as one-time preferences.
- [ ] (Deferred) Promote camera-source + signing-zone into a Camera
      section on the Settings page too
- [ ] (Deferred) Per-mode config grouping (item 4)
- [ ] (Deferred) Reset to defaults per section
- [x] Visual design pass: matches the dark theme
- [x] User-settings exposed (V1):
  - **CTQI alert threshold** -- editable Settings -> Advanced
    (default 80, persisted in localStorage). Below this: pulsing red
    banner + optional chime.
  - **CTQI hard floor** -- editable Settings -> Advanced (default 40).
    Below this: caption text replaced with "[unclear]" (the model is
    essentially guessing -- don't present a confidently-wrong sentence).
  - **TTS toggle** -- Settings -> Audio AND cog popup (bidirectional)
  - **Low-CTQI chime** -- Settings -> Audio AND cog popup
  - **Speaker wake word** -- Settings -> Audio AND cog popup
- [ ] (Deferred) **Gloss-confidence floor** -- still server-side env
      var `SIGNBRIDGE_MIN_CONFIDENCE` (default 0.35). NOT exposed in
      Settings UI -- this is a per-sign gate, distinct from CTQI which
      is sentence-level. Making it runtime-mutable would require a
      server-side session-level refactor; not user-requested.

## 7b. Wake-word speaker activation -- RESOLVED (Chrome) / NOT-FIXED (Edge)

Wake word ("Signer" by default, configurable in settings) triggers
SpeechRecognition for the speaker's reply, stripping the wake word
from the transcript. **Works in Chrome.** Does not work reliably in
Edge -- documented as a known browser limitation.

### Working design (Chrome)

Single-shot SpeechRecognition sessions, each started inside a
user-gesture click handler:
- **Start Conversation** click arms session 1 for the first speaker turn
- **Send Message** click (after each signed utterance) arms a fresh
  session for the next speaker turn -- natural conversational rhythm
- **New Conversation** click arms a session after full reset

No auto-restart on `onend` (setTimeout-triggered start has no user
gesture and Chromium aborts it). When a session ends (silence finalize
or Chromium ~60s timeout), it stays idle until the next click.

Tap to Speak button hidden in Kiosk/Lanyard (the wake-word flow makes
it redundant and confused turn-taking UX). Code retained as dormant
fallback.

### Edge attempts (all failed)

- Attempt 1: auto-start on live mode entry -> `network` error instantly.
  No user-gesture for SpeechRecognition.start().
- Attempt 2: start from button-click + auto-restart on onend -> `aborted`
  loop. The setTimeout-driven restarts had no gesture, but ALSO Edge
  seems to handle continuous-mode sessions less robustly than Chrome.
- Attempt 3: drop `audio:true` from getUserMedia + click-triggered start
  -> wake word still failed AND Tap to Speak broke (Edge's
  SpeechRecognition relies on getUserMedia's implicit mic grant).
- Attempt 4: single-shot from Start/Send/New-Conversation clicks (same
  as the Chrome working design) -> `network` error after a few seconds.
  Edge's STT cloud appears to close idle SR connections more
  aggressively than Google's, plus background-audio interference (a
  movie playing nearby in a non-English language) was confusing
  recognition further.

### Recommendation

- For Edge users: stick to Tap to Speak (works) until either
  (a) server-side Whisper is implemented, or (b) a future Edge update
  improves SpeechRecognition reliability.
- Long-term proper fix: server-side Whisper STT in Flask. No browser
  dependency, no cloud STT, works offline. Estimated ~1 day to
  implement. Promotes to its own backlog item if/when picked up.

## 8. Language translation on send — MVP scope (P0)

After ASL is recognized and the LLM constructs an English sentence,
clicking "Send sentence" should translate the final text to the
device user's preferred language before display / TTS. Open question
whether the signer and the speaker need separate language preferences
(both consuming the conversation in their own preferred language) or
a single device-wide preference is enough.

For now: signer-side defaults to American English, since the only
recognition model loaded is ASL and a user who chose ASL expects to
read English output. Translation kicks in only for the speaker side
or when explicitly changed.

### MVP scope (P0 — what we build now)

Smallest end-to-end thing that's actually useful for a demo:

- [ ] **Engine**: reuse the existing LLM (OpenAI client already wired
      for sentence construction) as the translation backend — one new
      `/api/translate` route, prompt = "translate this English to
      {target_lang}; reply with only the translation". No new vendor,
      no new API key, no new failure mode.
- [ ] **Language scope**: ship with Spanish, French, Mandarin, Hindi
      as the picker options. Easy to extend later by adding to the
      enum; LLM handles them all.
- [ ] **Setting**: single device-wide `speaker_lang` pref in Settings
      → Audio (new "Speaker language" select, default English). One
      pref per device, NOT split signer/speaker — that's a later
      refinement once we have real bilingual users to ask.
- [ ] **Trigger**: translate ONLY on Send, never on rolling caption
      updates. Keeps cost + latency bounded.
- [ ] **Display**: replace the caption text with the translation; keep
      a small "EN: <original>" line beneath in a muted color so the
      signer can verify the English the model produced. (Cheap, helps
      debugging when translation is wrong.)
- [ ] **TTS in target language**: pass `lang` attribute on the
      `SpeechSynthesisUtterance` so the browser picks a voice that
      matches. Web Speech API ships voices for all four MVP languages
      on modern Chrome/Edge.
- [ ] **Graceful failure**: if the LLM translate call fails or returns
      empty, fall back to the English caption + a small "⚠ translation
      unavailable" indicator. Don't block Send.

### Out of MVP scope (revisit in P2+ once MVP is in users' hands)

- Dedicated translation vendor (DeepL / Google Translate / NLLB).
  LLM-as-translator is fine for the MVP and stays good enough as
  long as the LLM keeps improving.
- Split `signer_lang` vs `speaker_lang` — only worth it if a real
  user asks for it.
- STT in non-English (hearing-person-side mic in Spanish, etc.)
- Translating every interim rolling caption (cost + flicker)

## 9. Per-mode translate pipeline: streaming vs batch (record -> Translate)

Two ways to drive recognition, selectable **per Live mode** via the
`TRANSLATE_MODE` map in app.js:

- **stream** -- continuous WS pipeline: frames stream live, motion-gated
  segmentation + per-sign inference run in real time, caption built
  incrementally (original production flow).
- **batch** -- record the whole signing clip locally with NO inference;
  "Start Signing" begins an open-ended recording, "Translate" sends the
  entire clip to `/api/process-signing-clip` (segment + gloss inference in
  one pass) then one LLM sentence build. Send / Regenerate / Edit unchanged.

Motivation: signers prefer signing a full sentence then translating, rather
than the system reacting word-by-word. Batch also matches the isolated-sign
(WLASL) training distribution better when the signer pauses between words
-> cleaner segmentation + higher accuracy.

Current state (experiment, behind config -- default Kiosk=batch, Lanyard=stream):
- [~] Per-mode config `TRANSLATE_MODE = { kiosk: 'batch', lanyard: 'stream' }`;
      `_isBatchTranslate()` reads `SIGNBRIDGE_MODE` live. Any combination valid.
- [x] Batch flow wired: Start Signing -> open-ended MediaRecorder; Translate
      -> process-signing-clip -> construct-sentence-live; Send/Regenerate/Edit
      reuse the existing handlers (no segmenter; reads `liveCollectedGlosses`).
- [x] Tail-motion cleanup: drop glosses starting in the last
      `BATCH_TRAILING_TRIM_SEC` (default 0.6s) of the clip + dedup consecutive
      identical glosses (hands lowering / reaching for the mouse were creating
      phantom signs like a doubled BATHROOM + a spurious BREAKDOWN).
- [x] Restart / New Conversation / stopLiveMode have batch-aware branches
      (soft reset without tearing down the camera).
- [x] `process-signing-clip` returns `duration_sec` / `total_frames` so the
      client can identify tail-motion segments.

Open / to decide (de-prioritized to P2 — feature works end-to-end):
- [ ] Final default per mode after an accuracy + latency A/B (stream vs batch
      on the same camera / lighting; the config flag makes this a 1-line flip)
- [ ] Auto-stop on stillness: lightweight client-side pixel-diff (NO model)
      that fires Translate after ~2.5s of no motion, so the signer never
      reaches for the mouse -- removes the tail-motion problem at the source
- [ ] Lanyard-batch fps caveat: WiFi cam captures at 15fps but
      `HybridSegmenter.cooldown_frames=45` assumes ~30fps (=> 3s vs 1.5s to
      close a segment). Make cooldown fps-relative if Lanyard batch is adopted.
- [ ] Tune segmenter for deliberate paused isolated signing (adaptive p25
      motion threshold can pick up jitter on pause-heavy clips; `min_sign_frames`
      floor). Only if segment COUNT errors show up -- gloss errors are the
      classifier's limit, not segmentation.
- [x] Mid-session mode switch does NOT rebuild the pipeline -- **decided out of
      scope** (mode is chosen before Start Conversation, the normal flow).

## 10. Upload tab: paragraph-length video -> narrative transcript

New top-level Upload tab. User picks a domain, drops a pre-recorded
video (a paragraph of signing -- a story, a monologue, a recorded
conversation turn), submits, and gets back a multi-sentence narrative
transcript with per-sentence breakdown.

The pipeline is the batch translate pipeline we already shipped (item 9),
with the LLM stage swapped out for a paragraph-mode call. Everything else
re-uses what's already validated: motion-gate segmentation, per-sign
inference, unrecognized-segment tracking, CTQI scoring, Edit/Regenerate.

### Decisions (confirmed)

- **Sentence segmentation**: LLM-decides. Send the full ordered gloss
  list (with per-gloss start_sec/end_sec) to a new construct endpoint
  and let the LLM segment into sentences. Simpler than pause-heuristic,
  leverages the model's natural sentence-boundary capability.
- **Max upload size / length**: 100 MB / ~5 min. Bigger pushes us into
  chunked upload + persistent storage territory; 100 MB is enough for
  the demo target without that complexity.
- **Persistence**: configurable in code, on for now. Keep 2-3 demo
  uploads on R2 under `uploads/<upload_id>/` so the result page can
  replay the source video next to the transcript. A `PERSIST_UPLOADS`
  constant flips it off later.
- **Progress feedback**: spinner for MVP (single synchronous request
  with a busy indicator). Polling endpoint added later only if 5-min
  uploads start hitting proxy timeouts.

### Input expectation (UI helper text)

> "Best with deliberately-paced signing -- pause ~1 second between
> signs. Faster / fluent signing may merge words together or drop
> them; use the Edit chip on each sentence in the result page to fix
> any misses."

Reason: the motion gate's cooldown (~0.53s) + quiet window (0.8s)
need ~1s of stillness between signs to cleanly separate them. AND the
underlying WLASL model is trained on isolated signs, not co-articulated
continuous signing -- even with perfect segmentation, fluent signing
degrades recognition. Honest framing avoids over-promising.

### Reuses (no new logic)

- `/api/process-signing-clip` with `segmenter=motion_gate` -- decode,
  motion-gate segment, per-sign inference -> returns gloss list with
  start_sec/end_sec, plus the existing `confident` flag.
- `segment_clip_offline` -- same motion gate over the uploaded clip.
- `_active_glosses_for_model_dir` -- domain vocab for the LLM prompt.
- `_renderCaptionHtml` -- per-sentence card on the result page (CTQI
  badge, action chips, [unclear] hard-floor render).
- Unrecognized-segment tracking + warning UX -- applied per sentence
  AND aggregated ("3 signs not recognized across the paragraph").
- CTQI v3 (GA includes unclear) -- per sentence.
- R2 storage pattern from sign-bank + demo-samples -- new `uploads/`
  prefix when persistence is on.

### New work — **SHIPPED**

- [x] New `/api/construct-paragraph` route — `app.py:3109`. Input: full
      gloss list with timestamps, domain, optional `previous_attempt`
      for regenerate mode. Output: paragraph text + per-sentence
      plausibility.
- [x] Upload tab HTML in `index.html` — `templates/index.html:140`
      (`<div id="phaseUpload">`). Sidebar nav entry present.
- [x] `static/js/upload.js` — self-contained IIFE; file pick/drop,
      spinner, result rendering, auto-refine (`UPLOAD_AUTO_REFINE`).
- [x] Upload section in `main.css`.
- [x] Flask `MAX_CONTENT_LENGTH` bumped to 100 MB.
- [x] `PERSIST_UPLOADS = True` + R2 upload helper (`_r2_upload_bytes`).
- [x] Result page: video preview + paragraph block with Regenerate /
      Edit / Save / Copy / Download chips (paragraph-level, not
      per-sentence — restructured per user request mid-build).
- [x] Helper text in UI matching the "1 second between signs" guidance.
- [x] **Bonus shipped**: Library feature — Save button persists demo to
      R2 metadata, Library panel lists/loads saved demos (`app.py:3393`
      save route, `upload.js:517-601` library functions).

### Out of scope (for v1)

- Chunked upload / resumable upload (not needed at 100 MB).
- Background job queue with status polling (spinner is fine for MVP).
- Editing the source video (trim, crop) before processing.
- Multi-domain auto-detection within one paragraph.

## 11. Pose-velocity-based segmenter for moderately-fast signing (v2)

Follow-up to item 10. The current motion gate (pixel-diff + cooldown)
needs ~1s of stillness between signs and silently fails on faster
signing. Replace / augment with a velocity-based pass that runs full-clip
pose extraction first, then puts sign boundaries at local minima of
hand-keypoint velocity. That handles 0.3-0.5s gaps the pixel gate misses
-- when hands briefly slow between signs without coming to full rest.

Important caveat: this only fixes SEGMENTATION. The underlying WLASL
model is trained on isolated signs, so even with perfect boundaries
fluent co-articulated signing will still degrade recognition. True
fluent-signing support is a CSLR (Continuous Sign Language Recognition)
problem -- different model, different training data -- and out of scope.
This item is about getting moderate-pace continuous signing working
well enough; truly fluent signing remains a documented limitation.

### Open / to design

- [ ] Run full-clip MediaPipe pose extraction in one pass (cache the
      pose array so per-segment extraction below can slice into it
      instead of re-running pose on every segment -- net speed-up).
- [ ] Compute per-frame hand-velocity time series (avg over both hands).
- [ ] Place sign boundaries at velocity local-minima below a threshold
      (signs are high-velocity phases; transitions / brief settles are
      low-velocity).
- [ ] Tune the threshold + smoothing window. Compare segment count to
      pixel-gate baseline on a labeled paragraph clip.
- [ ] A/B against the pixel motion gate on the same uploads; keep both
      paths behind a `UPLOAD_SEGMENTER` config flag so we can revert.
- [ ] Document new timing rule of thumb in the Upload helper text once
      this lands (e.g. "0.3-0.5s between signs ok").

---

## Prioritization

**Re-tiered 2026-05-31 after audit.** Several items were marked open
in the doc but were actually shipped (1a, 1b, 7, 7b, 10, most of 6 and
9). The active queue is now small and focused: close the loose ends on
storage + per-mode polish, then ship the language-translation MVP.
Everything else slides to P2+ until those are done.

| Priority | Items                                                                                          |
|----------|------------------------------------------------------------------------------------------------|
| **P0**   | **3** (Sign-bank close-out — coverage audit + graceful 404 UI + storage doc), **4** (Per-mode setup close-out — mode-filtered Settings, lanyard pre-flight, mode-aware TTS defaults), **8 MVP** (Language translation via LLM, 4 languages, Send-time only) |
| P1       | —                                                                                              |
| P2       | 5 (Signer CTQI alert — needs signer-facing channel decision), 6 remaining (kiosk tap-to-repeat + failure logging), 9 polish (auto-stop on stillness, lanyard-batch fps fix) |
| P3       | 11 (Pose-velocity segmenter — experimental follow-up to 10)                                    |
| P4       | 8 v2 (split signer/speaker lang, dedicated translation vendor, STT in non-English)             |
| P5       | **1c** (Conf Call mode — deferred), 2 (WiFi cam microphone — deferred)                         |

### Already shipped (marked done in their sections above)

- 1a Kiosk mode, 1b Lanyard mode
- 7 Unified Settings tab, 7b Wake-word speaker activation (Chrome)
- 9 Per-mode translate stream-vs-batch (core flow; only polish remains)
- 10 Upload tab (+ Library bonus)
- 6 Graceful failure (CTQI badge + [unclear] floor + edit affordance)

### Process notes

- **Item 4 close-out is intentionally narrow.** The original "per-mode
  setup wizard" idea was scoped out — defaults baked into 1a/1b are
  sufficient. Only the three small polish items in §4 remain.
- **Item 8 MVP is intentionally LLM-only.** Cheapest path to a usable
  translation feature; we'll know whether translation quality matters
  to real users before committing to a dedicated vendor.
