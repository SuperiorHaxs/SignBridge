// ══════════════════════════════════════════════════════════════
// framing_live.js — real-time camera/viewpoint framing feedback
// ══════════════════════════════════════════════════════════════
// Runs MediaPipe Pose (client-side) on the practice camera at ~5fps and colors
// a border + a bottom message bar on the camera window so the signer can fix
// their framing BEFORE signing. Mirrors the server-side framing_diag signals
// (pitch, aspect, nose proportion, torso coverage) and uses the same baseline
// (served at /api/framing-baseline), so the live border agrees with the
// after-sign camera-fit score.
//
// Body-only (33-pt) — enough for framing. Coordinates are normalized [0,1] over
// the raw (un-mirrored) video; coverage is checked against the 4:3 center-crop
// the streaming pipeline actually sends, so it reflects what the model sees.

window.FramingLive = (function () {
    'use strict';
    const NOSE = 0, L_SH = 11, R_SH = 12, L_HIP = 23, R_HIP = 24;

    function vis(lm) {
        if (!lm) return false;
        return (lm.visibility == null) ? true : lm.visibility > 0.4;
    }
    function z(v, b) { return (b && b.std > 1e-6) ? (v - b.mean) / b.std : null; }

    // Is a normalized point inside the 4:3 center-crop the client sends upstream?
    // MediaPipe extrapolates out-of-frame landmarks with coords beyond [0,1], so
    // first reject anything outside the actual video frame (this is what caught a
    // cut-off torso being wrongly reported as "in view").
    function inCrop(x, y, vw, vh) {
        if (x < 0 || x > 1 || y < 0 || y > 1) return false;
        const tA = 4 / 3, sA = (vw && vh) ? vw / vh : tA;
        if (sA > tA) { const kw = tA / sA, lo = (1 - kw) / 2; return x >= lo && x <= 1 - lo; }
        if (sA < tA) { const kh = sA / tA, lo = (1 - kh) / 2; return y >= lo && y <= 1 - lo; }
        return true;
    }

    // Coarse, robust "are you usable" check. Deliberately geometric only (present
    // + reasonable distance) -- NOT a baseline-relative conformance check. The
    // WLASL baseline's viewpoint stats are far too tight to compare a real laptop
    // camera against (aspect/anatomy differ), so gating on them just nags. `bl`
    // is accepted but unused here.
    function evaluate(lms, bl, vw, vh) {
        if (!lms) return { level: 'bad', message: 'Step into the frame — no one is detected.' };
        const lSh = lms[L_SH], rSh = lms[R_SH];
        if (!(vis(lSh) && vis(rSh)))
            return { level: 'bad', message: 'Move so both your shoulders are in view.' };
        const msY = (lSh.y + rSh.y) / 2;
        const sw = Math.hypot(lSh.x - rSh.x, lSh.y - rSh.y);
        if (sw < 1e-4) return { level: 'warn', message: 'Face the camera straight on.' };

        // Distance = room below the shoulders for the signing space, in
        // shoulder-widths (aspect-independent). Lenient: only very close is red.
        let cropBottom = 1;
        if (vh > vw) { const kh = (vw / vh) / (4 / 3); cropBottom = 1 - (1 - kh) / 2; }
        const spaceBelow = (cropBottom - msY) / sw;
        if (spaceBelow < 0.45)
            return { level: 'bad', message: 'Too close — move back so your hands have room below your shoulders.' };
        if (spaceBelow < 0.75)
            return { level: 'warn', message: 'A little close — move back slightly.' };
        if (sw < 0.08)
            return { level: 'warn', message: 'A little far — move slightly closer.' };
        return { level: 'good', message: 'Framing looks good — ready to sign.' };
    }

    return { evaluate };
})();


class PracticeFramingMonitor {
    constructor() {
        this.pose = null; this.video = null; this.box = null; this.bar = null;
        this.running = false; this.initialized = false; this.baseline = null;
        this._pc = null; this._pctx = null;
        this._cand = null; this._candN = 0; this._shown = null;
        this.onLevel = null;   // callback(level) — 'good'|'warn'|'bad'|null — for gating
        this.level = null;
    }

    async init(videoElement) {
        if (this.initialized) { this.video = videoElement; this._ensureBar(); return; }
        if (!window.Pose) { console.warn('[FramingMonitor] MediaPipe Pose not available'); return; }
        this.video = videoElement;
        this._ensureBar();
        try { this.baseline = await (await fetch('/api/framing-baseline')).json(); } catch (_) { this.baseline = {}; }

        this._pc = document.createElement('canvas'); this._pc.width = 320; this._pc.height = 240;
        this._pctx = this._pc.getContext('2d');

        this.pose = new window.Pose({ locateFile: (f) => `/static/vendor/mediapipe-pose/${f}` });
        this.pose.setOptions({ modelComplexity: 0, smoothLandmarks: true, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });
        this.pose.onResults((r) => this._onResults(r));
        const w = document.createElement('canvas'); w.width = 64; w.height = 64; w.getContext('2d').fillRect(0, 0, 64, 64);
        try { await this.pose.send({ image: w }); } catch (_) {}
        this.initialized = true;
    }

    _ensureBar() {
        this.box = this.video ? (this.video.closest('.prac-camera-box') || this.video.parentElement) : null;
        if (this.box && !this.bar) {
            this.bar = this.box.querySelector('.prac-framing-bar');
            if (!this.bar) { this.bar = document.createElement('div'); this.bar.className = 'prac-framing-bar'; this.box.appendChild(this.bar); }
        }
    }

    start() {
        if (!this.initialized || this.running) return;
        this.running = true;
        this._cand = null; this._candN = 0; this._shown = null;
        this._loop();
    }

    stop() {
        this.running = false;
        this._apply(null, '');
    }

    _loop() {
        if (!this.running) return;
        const v = this.video;
        if (v && v.readyState >= 2 && v.videoWidth) {
            try { this._pctx.drawImage(v, 0, 0, 320, 240); this.pose.send({ image: this._pc }).catch(() => {}); } catch (_) {}
        }
        if (this.running) setTimeout(() => this._loop(), 200);
    }

    _onResults(r) {
        if (!this.running) return;
        const vw = this.video.videoWidth || 4, vh = this.video.videoHeight || 3;
        const res = window.FramingLive.evaluate(r.poseLandmarks || null, this.baseline, vw, vh);
        // Debounce the border level: only switch after 2 consecutive same readings
        // (avoids flicker); the message text always reflects the latest reading.
        if (res.level === this._cand) this._candN++;
        else { this._cand = res.level; this._candN = 1; }
        if (this._candN >= 2) this._shown = res.level;
        this._apply(this._shown || res.level, res.message);
    }

    _apply(level, msg) {
        if (this.box) {
            this.box.classList.remove('framing-good', 'framing-warn', 'framing-bad');
            if (level) this.box.classList.add('framing-' + level);
        }
        if (this.bar) {
            if (!level) { this.bar.style.display = 'none'; }
            else { this.bar.textContent = msg || ''; this.bar.className = 'prac-framing-bar ' + level; this.bar.style.display = 'block'; }
        }
        if (level !== this.level) {
            this.level = level;
            if (this.onLevel) { try { this.onLevel(level); } catch (_) {} }
        }
    }
}
window.PracticeFramingMonitor = PracticeFramingMonitor;
