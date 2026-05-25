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

### Implementation checklist (P0: Kiosk + Lanyard only)

- [ ] Mode selector UI inside Live tab (Kiosk / Lanyard; Conf Call
      placeholder disabled-until-built)
- [ ] Selected mode persisted in localStorage; returning user lands in
      their last-used mode
- [ ] Kiosk mode behavior: hide I Sign / I Speak, default camera =
      built-in
- [ ] Lanyard mode behavior: hide I Sign / I Speak, default camera =
      WiFi Camera (IP), auto-open URL prompt on mode select if not
      already connected
- [ ] "Enable TTS playback" toggle in Settings (per-device, defaults:
      off for Lanyard, off for Kiosk -- staff reads, opt-in for TTS)

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

- [ ] Audit: which signs are missing video samples, total size needed
- [ ] Decide storage: cloud (S3 / R2 / Cloudflare Stream / GitHub
      LFS / YouTube unlisted), local-bundled, or on-demand fetch
- [ ] Cost / latency / offline-availability tradeoff per option
- [ ] Migration: move existing videos to chosen storage
- [ ] App: update sign-bank fetch path; cache locally if remote
- [ ] Fallback when a video is missing or fetch fails (graceful UI)

## 4. Per-mode setup wizards (audio-in, audio-out, video-in, layout, devices)

For each of the three Live modes, work out:
- Audio input source(s) (device mic / WiFi-cam mic / phone mic)
- Audio output target (laptop speaker / phone / external)
- Video input source (device webcam / WiFi cam / multiple)
- Preview display layout (kiosk full-screen / lanyard transcript-only
  / conf-call PiP grid)
- Number of devices physically in the setup (1 vs N)
- Whether "I Sign" / "I Speak" mode toggle is still meaningful per mode

Sub-items:
- [ ] Per-mode setup walkthrough on first entry to that mode
- [ ] Settings page reflects only the knobs relevant to the active mode
- [ ] Persist per-mode config separately in localStorage
- [ ] Re-evaluate "I Sign" vs "I Speak" toggle per mode -- might be a
      single user role implied by mode, not a manual choice
- [ ] Validation: warn when picked config can't physically work (e.g.,
      "Lanyard mode needs the WiFi cam connected")

## 5. Signer alert when CTQI is low (regenerate / edit needed) -- visible from the lanyard

The CTQI score's value proposition is acted on by the signer when it's
low. In Lanyard mode the signer can't see the laptop screen, so the
alert needs to surface where the signer can perceive it.

- [ ] Decide signer-facing alert channel: phone vibrate? phone screen?
      a smartwatch ping? an audible cue the hearing person hears and
      relays? a small LED / haptic add-on on the lanyard?
- [ ] Implement the alert path (whatever channel wins above)
- [ ] Threshold + cooldown (don't ping on every short utterance)
- [ ] Make the regenerate / edit action reachable from the signer's
      side, not just the hearing person's UI
- [ ] Optional: a one-tap "I said that wrong, try again" gesture for
      the signer to manually trigger regenerate

## 6. Graceful failure when predictions are wrong

Connected to 5. The system must degrade well, not silently mislead.

- [ ] Visible confidence indicator on every emitted caption (already
      partially done via CTQI; verify it's always present)
- [ ] When confidence is below a hard floor, show "..." or "[unclear]"
      instead of guessing
- [ ] Always-available "this was wrong" affordance for the hearing
      person to correct in-place (text edit) -- already exists for
      live mode, audit completeness across modes
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

## 8. Language translation on send

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

- [ ] Decide translation engine (cloud Google Translate API / DeepL /
      a local model like NLLB) -- weigh cost, latency, offline
      capability, language coverage
- [ ] Trigger: run translation after sentence is finalized on Send,
      not on each interim rolling-caption update (avoid translating
      every partial)
- [ ] Decide: one device-wide language pref, or split into
      `signer_lang` + `speaker_lang`. (Lean toward split: a deaf user
      in Mexico signing ASL might want output in Spanish even though
      the model produces English first.)
- [ ] Default `signer_lang` = en-US (ASL model assumption); allow
      override
- [ ] Settings: language selector(s) under a new "Language" section
- [ ] Display: where does the translated text render? Replace the
      English caption, or show beneath it with the original visible?
- [ ] TTS in non-English: when the deaf user's translated sentence is
      spoken aloud to a hearing listener, ensure the Web Speech API
      voice matches the target language
- [ ] STT in non-English: if the hearing person speaks a non-English
      language, the speech-to-text pipeline needs to be configured
      for that language too (Web Speech API supports many; Whisper
      auto-detects)
- [ ] Graceful failure when translation API is down -- fall back to
      the original English with a small indicator

---

## Prioritization

(Filled in during the next working session. Each item gets a priority
tier and an ordering within tier.)

| Priority | Items |
|---|---|
| P0 | **1a** (Kiosk mode), **1b** (Lanyard mode), 4 (Per-mode setup -- now scoped to just the small per-mode defaults defined in 1a/1b, no separate wizard) |
| P1 | 5 (Signer CTQI alert), 6 (Graceful failure) |
| P2 | 7 (Unified Settings tab) |
| P3 | 3 (Sign-bank video storage) |
| P4 | 8 (Language translation on send) |
| P5 | **1c** (Conf Call mode -- deferred), 2 (WiFi cam microphone) |

**Process note on P0:** specs for Kiosk and Lanyard agreed during
mode-definition review (see sections 1a / 1b). Item 4's "per-mode setup
wizard" is **not** needed at this stage -- the per-mode defaults baked
into 1a/1b are sufficient. If a real wizard is needed later, revisit.
