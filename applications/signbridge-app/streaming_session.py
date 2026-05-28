"""
Streaming live mode — per-connection segmentation state machine.

Direct port of closed-captions/frontend_cv_service.py:
  - MotionDetector  -> _MotionDetector here
  - main loop's sign-detection state machine -> StreamingSession._tick_segmentation

Differences vs closed-captions:
  - Frames arrive over a WebSocket as JPEG bytes, not from a local cv2.VideoCapture.
  - No GUI / cv2.imshow; events are emitted via the send_json callback.
  - Thresholds default to 320x240 / 15 fps (browser stream) instead of 640x480 / 30 fps.
  - Vertical signing-zone mask: filters out approach/hand-drop motion below the chest.
  - Hysteresis: separate open vs continue thresholds. Tolerates subtle motion
    (fingerspelling, slow holds) within an in-progress sign.
  - Hand-presence gate at segment open: MediaPipe Hands must see a hand inside the
    zone before a new segment can start. This is the primary defense against
    "chair scrape" / "body shift" false segments where motion exists but no hand is
    actually doing anything.
  - Inference dispatch is decoupled: callers register a `dispatch_segment` callable
    that gets the captured frame list when a segment closes (wired up in step 3).

Step 2 scope: motion gate + segmentation only. No pose extraction, no inference.
"""

import time
import threading
import collections

import cv2
import numpy as np

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False


class _MotionDetector:
    """
    Motion detector with a vertical signing-zone gate.

    Per frame:
        gray = cvtColor + GaussianBlur(21x21)
        delta = absdiff(prev_gray, gray)
        thresh = (delta > 25) | dilate(2 iterations)
        score = sum(thresh[zone_top : zone_bottom, :])   # mask outside zone
        is_active = score > threshold

    The zone is the strip of the frame where signing actually happens
    (roughly forehead to chest). Motion below `zone_bottom` (lap, hands at
    rest on the desk) is ignored, which is what kills the "approach" and
    "hand-drop" false segments — those motions live mostly outside the zone.

    `zone_top` / `zone_bottom` are normalized [0.0 .. 1.0] from the top of
    the frame. Defaults: top=0.0 (entire top is included), bottom=0.70
    (bottom 30% of the frame is ignored).

    Tracks `frames_since_motion` so callers can ask `is_signing_complete()`
    after the cooldown elapses with no motion.
    """

    def __init__(self, threshold, threshold_continue, cooldown_frames,
                 zone_top=0.0, zone_bottom=1.0):
        # Hysteresis: `threshold` opens a segment (gates noise), `threshold_continue`
        # keeps an open segment alive (tolerates subtle motion lulls like fingerspelling
        # within a sign). The state machine passes `is_signing` to `update()` so we
        # know which one to apply.
        self.threshold = threshold
        self.threshold_continue = threshold_continue
        self.cooldown_frames = cooldown_frames
        self.zone_top = zone_top
        self.zone_bottom = zone_bottom
        self.prev_gray = None
        self.frames_since_motion = 0

    def update(self, frame_bgr, is_signing=False):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_gray is None:
            self.prev_gray = gray
            return False, 0

        delta = cv2.absdiff(self.prev_gray, gray)
        thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        h = thresh.shape[0]
        top = max(0, min(h, int(round(h * self.zone_top))))
        bot = max(top + 1, min(h, int(round(h * self.zone_bottom))))
        score = int(np.sum(thresh[top:bot, :]))

        self.prev_gray = gray
        # Hysteresis: low bar to stay active, high bar to become active
        active_threshold = self.threshold_continue if is_signing else self.threshold
        is_active = score > active_threshold
        if is_active:
            self.frames_since_motion = 0
        else:
            self.frames_since_motion += 1
        return is_active, score

    def is_signing_complete(self):
        return self.frames_since_motion >= self.cooldown_frames

    def reset(self):
        self.prev_gray = None
        self.frames_since_motion = 0


class _HandDetector:
    """
    Cheap MediaPipe Hands wrapper used as a gate at segment open.

    Runs only on demand (not every frame) so the per-second cost is negligible:
    a typical signing session opens a segment once every few seconds, so this
    fires a handful of times per minute. Idle scenes never trigger MediaPipe.

    Returns whether *any* detected hand has its mean landmark y inside the
    vertical signing zone [zone_top, zone_bottom]. We don't care about hand
    identity (left/right) — only presence in the zone.
    """

    def __init__(self, model_complexity=0, min_detection_confidence=0.55):
        self._mp = None
        self._lock = threading.Lock()
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence

    def _ensure(self):
        if self._mp is None and _MP_AVAILABLE:
            self._mp = mp.solutions.hands.Hands(
                static_image_mode=True,  # we call it sporadically; static mode
                                         # avoids the "non-monotonic timestamp"
                                         # warnings from tracking-mode.
                max_num_hands=2,
                model_complexity=self.model_complexity,
                min_detection_confidence=self.min_detection_confidence,
            )

    def detect_in_zone(self, frame_bgr, zone_top, zone_bottom):
        """Returns (any_hand_in_zone: bool, num_hands_total: int)."""
        if not _MP_AVAILABLE:
            return True, 0  # fail open: if mediapipe missing, don't block segments
        with self._lock:
            self._ensure()
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = self._mp.process(rgb)
        if not results.multi_hand_landmarks:
            return False, 0
        num = len(results.multi_hand_landmarks)
        for hand in results.multi_hand_landmarks:
            # Mean y of all 21 hand landmarks. Normalized [0, 1].
            avg_y = sum(lm.y for lm in hand.landmark) / len(hand.landmark)
            if zone_top <= avg_y <= zone_bottom:
                return True, num
        return False, num

    def close(self):
        with self._lock:
            if self._mp is not None:
                try:
                    self._mp.close()
                except Exception:
                    pass
                self._mp = None


# One process-wide hand detector. MediaPipe is heavy to initialize (~hundreds
# of ms) and we don't need per-session isolation since the lock serializes calls.
_HAND_DETECTOR = _HandDetector()


class StreamingSession:
    """
    Per-WebSocket-connection segmentation state machine.

    Drives the same idle -> signing -> close-on-cooldown flow as the closed-captions
    main loop, but fed by JPEG frames arriving over a WebSocket. Emits events via
    `send_json` (a callable of one dict argument) so the WS handler can ship them
    to the browser.

    `dispatch_segment(frames)` is invoked exactly once per accepted segment, on the
    same thread as `on_frame`. It is intentionally None in step 2; step 3 will set
    it to a function that submits the frame list to a pose-extract + inference
    worker so the segmentation loop never blocks on the model.
    """

    # Defaults assume the browser streams at 320x240 / 15 fps with JPEG q ~ 0.6.
    # Closed-captions ran 640x480 / 30 fps with raw frames; pixel area is 4x smaller
    # here, so the motion-score threshold scales accordingly. JPEG noise can lift
    # the baseline a bit — expose this so the test page can tune live.
    DEFAULTS = {
        # OPEN threshold — must be exceeded to start a new segment. Set above
        # noise floor so head fidgets, camera grain etc don't trigger.
        'motion_threshold': 325_000,
        # CONTINUE threshold — used while a segment is already open. Set just
        # above the noise floor so subtle motion (fingerspelling, slow handshape
        # change within a held sign) keeps the segment alive instead of
        # prematurely closing it and letting the hand-drop start a new segment.
        'motion_threshold_continue': 120_000,
        'cooldown_frames': 8,          # ~530ms @ 15 fps
        'min_sign_frames': 10,         # ~670ms @ 15 fps  (closed-captions: 15 @ 30)
        'sign_debounce_s': 0.5,        # at segment close: reject if total time too short
        # post_segment_quiet_s: at segment OPEN, reject new motion that arrives within
        # this window after the previous accepted segment closed.
        'post_segment_quiet_s': 0.8,
        # Signing zone — fractions of the frame height from the top. Motion outside
        # [zone_top, zone_bottom] is masked out, which is the primary fix for
        # "approach from lap" and "hand-drop to lap" false segments.
        'zone_top': 0.0,
        'zone_bottom': 0.90,
        # If True, run MediaPipe Hands once at the rising edge of motion (when
        # we'd otherwise open a segment) and abort the open if no hand is in the
        # zone. This filters chair-scrape, posture-shift, and similar pixel motion
        # that has no hand behind it. Cost is negligible — only runs at segment
        # candidates, not per frame.
        'require_hand_present': True,
        'buffer_seconds': 10,
        'fps': 15,
    }

    def __init__(self, send_json, **overrides):
        self.cfg = {**self.DEFAULTS, **{k: v for k, v in overrides.items() if v is not None}}
        self.send_json = send_json
        # Injectable clock. Live path uses wall time; the offline clip driver
        # overrides this with a virtual clock that advances at the clip's frame
        # cadence, so the time-based gates (quiet window, debounce) behave the
        # same offline as they do live.
        self.now_fn = time.time
        self.motion = _MotionDetector(
            threshold=self.cfg['motion_threshold'],
            threshold_continue=self.cfg['motion_threshold_continue'],
            cooldown_frames=self.cfg['cooldown_frames'],
            zone_top=self.cfg['zone_top'],
            zone_bottom=self.cfg['zone_bottom'],
        )
        self.frame_deque = collections.deque(
            maxlen=int(self.cfg['buffer_seconds'] * self.cfg['fps'])
        )
        self.is_signing = False
        self.signing_frames = []
        self.last_sign_ts = 0.0
        self.frame_count = 0
        self.segment_id = 0
        self.frame_ms = 1000.0 / max(self.cfg['fps'], 1)
        self._suppressing = False  # one-shot flag so segment_suppressed only fires on rising edge

        # Step 3 will set this to push frames into an inference worker.
        # Signature: dispatch_segment(segment_id: int, frames: list[np.ndarray]) -> None
        self.dispatch_segment = None

    # ------------------------------------------------------------------ control

    def reconfigure(self, **overrides):
        """Apply runtime overrides from the client (e.g. tuning sliders)."""
        for k, v in overrides.items():
            if v is None or k not in self.DEFAULTS:
                continue
            self.cfg[k] = v
        self.motion.threshold = self.cfg['motion_threshold']
        self.motion.threshold_continue = self.cfg['motion_threshold_continue']
        self.motion.cooldown_frames = self.cfg['cooldown_frames']
        self.motion.zone_top = self.cfg['zone_top']
        self.motion.zone_bottom = self.cfg['zone_bottom']
        self.frame_deque = collections.deque(
            maxlen=int(self.cfg['buffer_seconds'] * self.cfg['fps']),
        )
        self.frame_ms = 1000.0 / max(self.cfg['fps'], 1)

    def reset(self):
        self.motion.reset()
        self.is_signing = False
        self.signing_frames = []
        self.last_sign_ts = 0.0
        self.segment_id = 0
        self._suppressing = False

    # ------------------------------------------------------------------ ingest

    def on_frame(self, jpeg_bytes):
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            self.send_json({"type": "error", "where": "decode", "msg": "imdecode returned None"})
            return

        self.frame_count += 1
        self.frame_deque.append(frame)

        # Pass current signing state so the detector applies hysteresis correctly.
        is_active, score = self.motion.update(frame, is_signing=self.is_signing)

        # Surface the effective threshold so the test page can render it on the bar.
        eff_threshold = (self.cfg['motion_threshold_continue']
                         if self.is_signing else self.cfg['motion_threshold'])

        self.send_json({
            "type": "motion",
            "frame": self.frame_count,
            "score": score,
            "active": is_active,
            "signing": self.is_signing,
            "segment_frames": len(self.signing_frames),
            "threshold": eff_threshold,
        })

        self._tick_segmentation(is_active, frame)

    # ------------------------------------------------------------------ state machine

    def _tick_segmentation(self, is_active, frame):
        # Reset suppression flag whenever motion goes quiet — so the next rising
        # edge gets a fresh `segment_suppressed` notification.
        if not is_active:
            self._suppressing = False

        # Open a new segment on the rising edge of motion — UNLESS we're still
        # in the post-segment quiet window, in which case suppress (likely hand-drop).
        if is_active and not self.is_signing:
            now = self.now_fn()
            quiet = self.cfg['post_segment_quiet_s']
            gap = now - self.last_sign_ts
            if self.last_sign_ts > 0 and quiet > 0 and gap < quiet:
                # Emit a one-shot "suppressed" event only on the rising edge so the
                # client sees it once per attempted re-open, not once per frame.
                if not self._suppressing:
                    self.send_json({
                        "type": "segment_suppressed",
                        "reason": "quiet_window",
                        "frame": self.frame_count,
                        "gap_s": round(gap, 3),
                        "quiet_s": quiet,
                    })
                    self._suppressing = True
                return

            # Hand-presence gate: motion exists but is it actually a hand?
            # Run MediaPipe Hands once now (not per-frame) to confirm before
            # spending capacity on a segment. Blocks chair-scrape / body-shift.
            if self.cfg['require_hand_present']:
                t0 = time.time()
                in_zone, num_hands = _HAND_DETECTOR.detect_in_zone(
                    frame, self.cfg['zone_top'], self.cfg['zone_bottom']
                )
                ms = (time.time() - t0) * 1000.0
                if not in_zone:
                    if not self._suppressing:
                        self.send_json({
                            "type": "segment_suppressed",
                            "reason": "no_hand_in_zone",
                            "frame": self.frame_count,
                            "num_hands_detected": num_hands,
                            "mediapipe_ms": round(ms, 1),
                        })
                        self._suppressing = True
                    return

            self._suppressing = False
            self.is_signing = True
            self.signing_frames = []
            self.segment_id += 1
            self.send_json({
                "type": "segment_start",
                "segment_id": self.segment_id,
                "frame": self.frame_count,
                "ts": now,
            })

        if not self.is_signing:
            return

        # Accumulate every frame while in the signing state, including the
        # tail-of-motion frames where score falls below threshold but cooldown
        # hasn't elapsed yet. Closed-captions does the same.
        self.signing_frames.append(frame)

        if not is_active and self.motion.is_signing_complete():
            self._close_segment()

    def _close_segment(self):
        n = len(self.signing_frames)
        now = self.now_fn()

        rejected = None
        if n < self.cfg['min_sign_frames']:
            rejected = f"too_short ({n} < {self.cfg['min_sign_frames']} frames)"
        elif self.cfg['sign_debounce_s'] > 0 and (now - self.last_sign_ts) < self.cfg['sign_debounce_s']:
            gap = now - self.last_sign_ts
            rejected = f"debounce ({gap:.2f}s < {self.cfg['sign_debounce_s']}s)"

        event = {
            "type": "segment_end",
            "segment_id": self.segment_id,
            "frames": n,
            "duration_ms": int(n * self.frame_ms),
            "accepted": rejected is None,
        }
        if rejected is not None:
            event["rejected_reason"] = rejected
        self.send_json(event)

        if rejected is None:
            self.last_sign_ts = now
            if self.dispatch_segment is not None:
                try:
                    self.dispatch_segment(self.segment_id, list(self.signing_frames))
                except Exception as e:
                    self.send_json({"type": "error", "where": "dispatch", "msg": str(e)})

        self.is_signing = False
        self.signing_frames = []


# Reference capture size + fps the motion thresholds are calibrated against.
# Matches the live streamer (streaming_live.js captures 480x360 @ 15 fps), so
# resizing offline clip frames to this size makes the same thresholds behave
# the same way they do live.
_REF_W, _REF_H, _REF_FPS = 480, 360, 15


def segment_clip_offline(frames, fps, zone_top=None, zone_bottom=None):
    """
    Run the LIVE motion gate over a pre-recorded clip and return accepted
    segment boundaries as (start_idx, end_idx) tuples into `frames`.

    This gives the batch ("record then Translate") flow the same hand-gated
    segmentation the streaming WebSocket path produces -- signing-zone mask +
    motion hysteresis + MediaPipe hand-presence + quiet-window/debounce --
    instead of HybridSegmenter's pixel-only boundaries.

    Faithfulness to the live path is what matters here:
      * Frames are SUBSAMPLED to the reference fps (15) before scoring, so the
        frame-to-frame motion deltas match the calibrated thresholds. Driving
        every frame of a 30 fps clip would halve the delta interval, shrink the
        motion score, and split single signs into fragments.
      * Frames are resized to the reference capture size (480x360) for scoring.
      * A VIRTUAL CLOCK advances at the reference cadence so the time-based
        gates (post_segment_quiet_s, sign_debounce_s) behave exactly as live --
        these are what stop a brief mid-sign motion lull from closing a segment
        and immediately reopening a second one.

    `frames` are full-resolution BGR frames; the returned indices point into
    that original full-fps list (so the caller slices full temporal resolution
    for pose extraction).
    """
    if not frames:
        return []
    fps = fps or _REF_FPS

    # Subsample original frame indices down to ~_REF_FPS.
    stride = max(fps / _REF_FPS, 1.0)
    sample_idx = []
    i = 0.0
    while int(round(i)) < len(frames):
        sample_idx.append(int(round(i)))
        i += stride
    if not sample_idx:
        return []

    overrides = {'fps': _REF_FPS}
    if zone_top is not None:
        overrides['zone_top'] = zone_top
    if zone_bottom is not None:
        overrides['zone_bottom'] = zone_bottom

    session = StreamingSession(send_json=lambda *_a, **_k: None, **overrides)

    # Virtual clock at the reference cadence.
    clock = {'t': 0.0}
    dt = 1.0 / _REF_FPS
    session.now_fn = lambda: clock['t']

    segments = []
    cur = {'k': -1}

    def _dispatch(_seg_id, seg_frames):
        end_k = cur['k']
        start_k = max(0, end_k - len(seg_frames) + 1)
        segments.append((sample_idx[start_k], sample_idx[min(end_k, len(sample_idx) - 1)]))

    session.dispatch_segment = _dispatch

    for k, oidx in enumerate(sample_idx):
        cur['k'] = k
        clock['t'] += dt
        small = cv2.resize(frames[oidx], (_REF_W, _REF_H))
        is_active, _score = session.motion.update(small, is_signing=session.is_signing)
        session._tick_segmentation(is_active, small)

    # Flush a segment still open at end-of-clip (no trailing cooldown to close it).
    if session.is_signing and len(session.signing_frames) >= session.cfg['min_sign_frames']:
        end_k = cur['k']
        start_k = max(0, end_k - len(session.signing_frames) + 1)
        segments.append((sample_idx[start_k], sample_idx[min(end_k, len(sample_idx) - 1)]))

    return segments
