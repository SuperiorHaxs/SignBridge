/**
 * StreamingLiveMode — replacement for SignSegmenter in the live mode path.
 *
 * Opens a WebSocket to /ws/live-stream, captures frames from a <video> element,
 * downsamples them to 320x240 JPEG @ 15 fps, and emits glosses produced by the
 * server-side motion gate + pose extraction + OpenHands inference.
 *
 * Exposes a SignSegmenter-shaped surface so app.js wiring barely changes:
 *   .init(videoEl)          open WS + start frame loop
 *   .start(stream, trackOnly)   no-op (kept for call-site compatibility)
 *   .stop()                 close WS, stop frame loop, clear state
 *   .clearGlosses()         drop the accumulated gloss buffer
 *   .getCollectedGlosses()  snapshot of accumulated glosses
 *
 * Callbacks (set after construction):
 *   .onGloss(prediction)        a confident new gloss arrived
 *   .onReadyToSend(glosses)     LLM trigger: glossBatchSize new glosses have
 *                               arrived. (No idle flush -- use Send Message
 *                               for manual flush via app.js sendNowAndPause.)
 *   .onStateChange(state)       'signing' | 'analyzing' | 'idle'
 *   .onMotion({score, signing}) per-frame motion debug
 *
 * Configurable knobs (any source wins in this order):
 *   URL query   ?glossBatchSize=3
 *   localStorage signbridge.glossBatchSize
 *   constructor options
 *   Runtime:    streamingLive.glossBatchSize = 5 (takes effect next gloss)
 */

(function (global) {
  'use strict';

  function readConfig(name, fallback) {
    try {
      const params = new URLSearchParams(global.location.search);
      const fromUrl = params.get(name);
      if (fromUrl !== null) {
        const n = parseFloat(fromUrl);
        if (!isNaN(n)) return n;
      }
      const fromLs = global.localStorage.getItem('signbridge.' + name);
      if (fromLs !== null) {
        const n = parseFloat(fromLs);
        if (!isNaN(n)) return n;
      }
    } catch (_) {}
    return fallback;
  }

  class StreamingLiveMode {
    constructor(opts) {
      opts = opts || {};
      this.domain = opts.domain || 'emergency';

      // LLM-trigger knob. The idle timer was removed -- it fired
      // automatically whenever the user paused for idleSendMs, which felt
      // unpredictable. Now the LLM only fires on (a) glossBatchSize new
      // glosses or (b) explicit Send Message click. Manual flush via
      // sendNowAndPause() in app.js handles the "I'm done" intent.
      // WebSocket path. Defaults to /ws/live-stream (conversation streaming);
      // practice mode passes '/ws/practice-stream' here.
      this.wsPath = opts.wsPath || '/ws/live-stream';

      this.glossBatchSize = readConfig('glossBatchSize', opts.glossBatchSize != null ? opts.glossBatchSize : 3);

      // Initial signing-zone values. Sent with the start message so the
      // server picks up the UI-configured zone immediately, not just on
      // subsequent tune() calls. Server has its own defaults if these are
      // null (zone_top=0.0, zone_bottom=0.90 in streaming_session.py).
      this.zoneTop    = opts.zoneTop    != null ? opts.zoneTop    : null;
      this.zoneBottom = opts.zoneBottom != null ? opts.zoneBottom : null;

      // Capture settings. Bumped 320x240 -> 480x360 to give MediaPipe Hands
      // enough hand pixels for reliable detection from a lanyard-mounted
      // WiFi camera (~3 ft signer distance). 4:3 aspect retained so the
      // ESP32 VGA source crops correctly. Device camera path still works
      // (its native frame is downscaled to 480x360, no quality loss).
      this.captureWidth  = 480;
      this.captureHeight = 360;
      this.jpegQuality   = 0.6;
      this.targetFps     = 15;

      // Digital zoom factor for _sendFrame's center-crop. 1.0 = no zoom
      // (just aspect-correct crop). >1 = tighter crop = signer fills more
      // of the frame. Used to match laptop-camera FOV (~60deg) on the
      // wider ESP32 OV2640 lens (~75deg) -- set to ~1.3 when WiFi cam
      // connects. Safe up to ~1.33 with VGA source (still downsampling
      // into the target, no info-loss upscale).
      this.zoomFactor = opts.zoomFactor != null ? opts.zoomFactor : 1.0;

      // Horizontal flip applied INSIDE _sendFrame, before drawImage. The
      // display preview is independently CSS-mirrored via .camera-view >
      // video {scaleX(-1)}; this flag only affects what's sent to the
      // server for inference. Set true when the source frame has signer's
      // right hand on viewer's RIGHT (opposite of standard training data
      // convention) -- e.g. the case-mounted ESP32 cam where hmirror+vflip
      // produces the wrong handedness. Without this, ASL signs that differ
      // only by hand dominance get misclassified.
      this.mirrorBeforeSend = !!opts.mirrorBeforeSend;

      // Callbacks (set by the integration site).
      this.onGloss        = null;
      this.onReadyToSend  = null;
      this.onStateChange  = null;
      this.onMotion       = null;
      this.onConfig       = null;   // (cfg) => void  — fired with server's effective config

      // Internal state.
      this._ws          = null;
      this._canvas      = null;
      this._ctx         = null;
      this._tickHandle  = null;
      this._keepaliveHandle = null;  // ping interval while paused
      this._pendingCount = 0;
      this._collected   = [];
      this._lastState   = null;
      this._videoEl     = null;
      this._stopped     = false;
      // Number of segments currently in flight (opened but not yet resolved
      // to a gloss / gloss_error / non-accepted segment_end). The idle timer
      // must NOT fire while this > 0, otherwise we flush partial sentences
      // every time inference latency exceeds idleSendMs.
      this._inFlight    = 0;
    }

    async init(videoEl) {
      this._videoEl = videoEl;
      this._stopped = false;
      this._openWS();
      this._startFrameLoop();
    }

    // Call-site compatibility — SignSegmenter took (cameraStream, trackOnly).
    // The WS pipeline reads from the video element directly, so we don't need
    // the stream reference. Accept and ignore.
    start(_stream, _trackOnly) {}

    // Pause: stop pushing frames to the server. The WS stays open but the
    // server's motion gate has nothing to chew on, so no new segments open.
    // Existing in-flight inference still completes and gloss events still
    // arrive. Used by the "Send Message" button to freeze detection.
    //
    // While paused we send a `ping` every 20s as a keep-alive -- otherwise
    // the server's 30s receive timeout closes the WS if the user takes too
    // long to click Resume Signing.
    pause() {
      if (this._tickHandle) { clearInterval(this._tickHandle); this._tickHandle = null; }
      // Reset segmenter state on the server so a tiny tail of motion in the
      // next resume doesn't reopen a phantom segment from before the pause.
      if (this._ws && this._ws.readyState === WebSocket.OPEN) {
        try { this._ws.send(JSON.stringify({ type: 'reset' })); } catch (_) {}
      }
      if (this._keepaliveHandle) clearInterval(this._keepaliveHandle);
      this._keepaliveHandle = setInterval(() => {
        if (this._ws && this._ws.readyState === WebSocket.OPEN) {
          try { this._ws.send(JSON.stringify({ type: 'ping' })); } catch (_) {}
        }
      }, 20000);
    }

    async resume() {
      if (this._keepaliveHandle) { clearInterval(this._keepaliveHandle); this._keepaliveHandle = null; }
      // If the WS died while paused (e.g. server timeout exceeded keep-alive
      // for any reason), reopen it before restarting the frame loop. Without
      // this, the frame loop runs and silently sends to a dead socket -- the
      // user clicks Resume and sees "no recognition" with no obvious cause.
      const ws = this._ws;
      if (!ws || ws.readyState === WebSocket.CLOSED || ws.readyState === WebSocket.CLOSING) {
        console.log('[StreamingLive] resume: WS dead, reopening');
        this._ws = null;
        this._openWS();
        // Brief wait so the new WS finishes its open handshake before
        // _sendFrame fires (it no-ops on non-OPEN, so this is just to avoid
        // a few dropped frames at the start of the resume).
        await new Promise(r => setTimeout(r, 300));
      }
      if (this._tickHandle) return;
      const intervalMs = 1000 / this.targetFps;
      this._tickHandle = setInterval(() => this._sendFrame(), intervalMs);
    }

    isPaused() { return this._tickHandle == null; }

    // Internal getter so the app-level "Send" can wait for inference to drain.
    inFlightCount() { return this._inFlight; }

    stop() {
      this._stopped = true;
      if (this._tickHandle) { clearInterval(this._tickHandle); this._tickHandle = null; }
      if (this._keepaliveHandle) { clearInterval(this._keepaliveHandle); this._keepaliveHandle = null; }
      if (this._ws && this._ws.readyState === WebSocket.OPEN) {
        try { this._ws.send(JSON.stringify({ type: 'stop' })); } catch (_) {}
        try { this._ws.close(); } catch (_) {}
      }
      this._ws = null;
      this._collected = [];
      this._pendingCount = 0;
      this._lastState = null;
      this._inFlight = 0;
    }

    clearGlosses() {
      this._collected = [];
      this._pendingCount = 0;
    }

    getCollectedGlosses() {
      return this._collected.slice();
    }

    // Client-side digital zoom. 1.0 = no zoom, 1.3 ~= laptop-cam FOV match
    // for the ESP32 wide lens. Pure center-crop in _sendFrame, takes effect
    // on the next tick (~67ms). Console-tunable: signSegmenter.setZoom(1.5)
    setZoom(z) {
      const v = Math.max(1.0, Math.min(3.0, Number(z) || 1.0));
      this.zoomFactor = v;
      console.log('[StreamingLive] zoom set to', v);
    }

    // Toggle horizontal flip on frames sent to inference. Display is
    // unaffected (separate CSS mirror). Console-tunable:
    //   signSegmenter.setMirror(true)   // flip (use for ESP32 wifi cam)
    //   signSegmenter.setMirror(false)  // no flip (device cam default)
    setMirror(m) {
      this.mirrorBeforeSend = !!m;
      console.log('[StreamingLive] mirrorBeforeSend =', this.mirrorBeforeSend);
    }

    // Optional: change domain mid-session (e.g. user switches scenarios).
    setDomain(domain) {
      this.domain = domain;
      if (this._ws && this._ws.readyState === WebSocket.OPEN) {
        try { this._ws.send(JSON.stringify({ type: 'tune', domain })); } catch (_) {}
      }
    }

    // Live-tune any of the server-side knobs (zone_top, zone_bottom,
    // motion_threshold, motion_threshold_continue, cooldown_frames,
    // min_sign_frames, sign_debounce_s, post_segment_quiet_s, require_hand_present).
    // The server echoes back the effective config via ack-control, which
    // triggers `onConfig` so the UI can re-sync.
    tune(overrides) {
      if (!this._ws || this._ws.readyState !== WebSocket.OPEN) return;
      try {
        this._ws.send(JSON.stringify({ type: 'tune', ...overrides }));
      } catch (e) {
        console.warn('[StreamingLive] tune send failed:', e.message);
      }
    }

    // ────────────────────────── internal ──────────────────────────

    _openWS() {
      const proto = global.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const url   = `${proto}//${global.location.host}${this.wsPath}`;
      console.log('[StreamingLive] opening', url, 'domain=', this.domain);

      const ws = new WebSocket(url);
      ws.binaryType = 'arraybuffer';
      this._ws = ws;

      ws.onopen = () => {
        try {
          const startMsg = {
            type: 'start',
            domain: this.domain,
            fps: this.targetFps,
          };
          if (this.zoneTop    != null) startMsg.zone_top    = this.zoneTop;
          if (this.zoneBottom != null) startMsg.zone_bottom = this.zoneBottom;
          ws.send(JSON.stringify(startMsg));
        } catch (e) {
          console.warn('[StreamingLive] start send failed:', e.message);
        }
      };
      ws.onmessage = (ev) => {
        if (typeof ev.data !== 'string') return;
        let m;
        try { m = JSON.parse(ev.data); } catch (_) { return; }
        this._handleMessage(m);
      };
      ws.onclose = (ev) => {
        console.log('[StreamingLive] ws closed', ev.code, ev.reason);
        if (!this._stopped) {
          this._emitState('idle');
        }
      };
      ws.onerror = () => {
        console.warn('[StreamingLive] ws error');
      };
    }

    _startFrameLoop() {
      this._canvas = document.createElement('canvas');
      this._canvas.width  = this.captureWidth;
      this._canvas.height = this.captureHeight;
      this._ctx = this._canvas.getContext('2d');

      const intervalMs = 1000 / this.targetFps;
      this._tickHandle = setInterval(() => this._sendFrame(), intervalMs);
    }

    _sendFrame() {
      const ws = this._ws;
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      const v = this._videoEl;
      if (!v || v.readyState < 2 || !v.videoWidth) return;

      try {
        // Step 1: aspect-correct center-crop (so 16:9 source doesn't get
        // stretched into 4:3).
        // Step 2: digital zoom by tightening that crop by `zoomFactor`,
        // re-centering on the same point. zoom=1.0 -> aspect crop only;
        // zoom=1.3 -> takes the centermost ~77% of the aspect-cropped
        // region (matches laptop-cam FOV vs the wider ESP32 OV2640 lens).
        const sw = v.videoWidth, sh = v.videoHeight;
        const targetAspect = this.captureWidth / this.captureHeight;
        const srcAspect = sw / sh;
        let cw = sw, ch = sh;
        if (srcAspect > targetAspect) {
          cw = sh * targetAspect;
        } else if (srcAspect < targetAspect) {
          ch = sw / targetAspect;
        }
        const z = this.zoomFactor || 1.0;
        if (z > 1.0) {
          cw = cw / z;
          ch = ch / z;
        }
        const cx = (sw - cw) / 2;
        const cy = (sh - ch) / 2;
        if (this.mirrorBeforeSend) {
          // Draw mirrored: translate to the right edge, flip the X axis,
          // then drawImage at (0,0) of the now-flipped coordinate system.
          this._ctx.save();
          this._ctx.translate(this.captureWidth, 0);
          this._ctx.scale(-1, 1);
          this._ctx.drawImage(v, cx, cy, cw, ch, 0, 0, this.captureWidth, this.captureHeight);
          this._ctx.restore();
        } else {
          this._ctx.drawImage(v, cx, cy, cw, ch, 0, 0, this.captureWidth, this.captureHeight);
        }

        this._canvas.toBlob((blob) => {
          if (!blob) return;
          const w = this._ws;
          if (!w || w.readyState !== WebSocket.OPEN) return;
          try { w.send(blob); } catch (e) { /* socket buffer pressure; drop frame */ }
        }, 'image/jpeg', this.jpegQuality);
      } catch (e) {
        // drawImage can throw if the source isn't decode-ready yet — ignore
      }
    }

    _handleMessage(m) {
      switch (m.type) {
        case 'motion': {
          if (this.onMotion) this.onMotion({ score: m.score, signing: m.signing, active: m.active });
          // Motion events fire at 15fps. Only flip TO 'signing' here; the
          // transition out of 'signing' is driven by segment_end (the
          // server's authoritative close). Critically, we MUST NOT emit
          // 'idle' while in 'analyzing' -- otherwise per-frame motion
          // ticks during inference would clobber "Recognizing..." within
          // 67ms of it being set.
          if (m.signing) {
            this._emitState('signing');
          } else if (this._lastState !== 'analyzing') {
            this._emitState('idle');
          }
          break;
        }
        case 'segment_start':
          // Track in-flight segments so Send Message can wait for inference
          // to drain before flushing the final LLM call.
          this._inFlight += 1;
          break;
        case 'segment_end':
          // Rejected segments will never produce a gloss; close their slot
          // and return to idle. Accepted segments stay in-flight until
          // gloss/gloss_error and we flip to 'analyzing' immediately so
          // the user sees "Recognizing..." from segment close until the
          // gloss arrives.
          if (m.accepted === false) {
            this._inFlight = Math.max(0, this._inFlight - 1);
            this._emitState('idle');
          } else {
            this._emitState('analyzing');
          }
          break;
        case 'segment_suppressed':
          // Server suppressed this segment before opening it (quiet window
          // or no-hand-in-zone). No in-flight counter to decrement.
          break;
        case 'inference_start':
          // Reaffirms analyzing; already set on segment_end (accepted) but
          // keeping this is a no-op due to dedup in _emitState.
          this._emitState('analyzing');
          break;
        case 'gloss': {
          // Segment that was in-flight is now resolved.
          this._inFlight = Math.max(0, this._inFlight - 1);

          const pred = {
            gloss: m.gloss,
            confidence: m.confidence,
            confident: m.confident,
            top_k: m.top_k || [],
          };

          // Decide whether this gloss should be COLLECTED (added to the buffer
          // that feeds the LLM). Confident, non-duplicate predictions go in;
          // low-confidence ones do NOT. But we still report every gloss to the
          // consumer so it can show them (greyed-out) in the diagnostic panel
          // -- otherwise the user gets total silence when the model is
          // unsure, with no idea why.
          let collected = false;
          if (m.confident) {
            const last = this._collected[this._collected.length - 1];
            const isDup = last && last.gloss &&
              last.gloss.toLowerCase() === String(m.gloss || '').toLowerCase();
            if (!isDup) {
              this._collected.push(pred);
              this._pendingCount += 1;
              collected = true;
            }
          }

          try { if (this.onGloss) this.onGloss(pred, collected); } catch (e) { console.error(e); }
          this._emitState('idle');

          if (collected) this._checkTrigger();
          break;
        }
        case 'gloss_error':
          // Segment in-flight resolved (negatively).
          this._inFlight = Math.max(0, this._inFlight - 1);
          console.warn('[StreamingLive] gloss_error:', m.msg);
          this._emitState('idle');
          break;
        case 'ack-control':
          // Server echoes its effective config every time we send a start/tune
          // message. Surface it via onConfig so UI can sync sliders + overlays.
          if (m.config && this.onConfig) {
            try { this.onConfig(m.config); } catch (e) { console.error(e); }
          }
          break;
      }
    }

    _checkTrigger() {
      // Only the batch path remains. Idle-flush behavior was removed --
      // it fired the LLM in the background without an explicit user signal,
      // which felt unpredictable. Use Send Message to flush on demand.
      if (this._pendingCount >= this.glossBatchSize) {
        this._fireReady();
      }
    }

    _fireReady() {
      if (this._pendingCount === 0) return;
      this._pendingCount = 0;
      try {
        if (this.onReadyToSend) this.onReadyToSend(this._collected.slice());
      } catch (e) { console.error(e); }
    }

    _emitState(state) {
      if (state === this._lastState) return;
      this._lastState = state;
      try { if (this.onStateChange) this.onStateChange(state, {}); } catch (e) { console.error(e); }
    }
  }

  global.StreamingLiveMode = StreamingLiveMode;
})(window);
