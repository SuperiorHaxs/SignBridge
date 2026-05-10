// ══════════════════════════════════════════════════════════════
// PosePlayer — Renders a pose sequence as an animated stick figure
// on a canvas. Used by the Practice diagnostic to compare user's
// signing against a reference training sample.
// ══════════════════════════════════════════════════════════════

class PosePlayer {
    constructor(canvas, options = {}) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.frames = null;           // (N, 83, 3) array of landmarks
        this.currentFrame = 0;
        this.fps = options.fps || 15;
        this.playing = false;
        this.animHandle = null;
        this.lastFrameTime = 0;

        this.bodyColor = options.bodyColor || 'rgba(100, 200, 255, 0.85)';
        this.handColor = options.handColor || 'rgba(180, 130, 255, 0.9)';
        this.boneWidth = 2.5;
        this.jointRadius = 3.5;
        this.mirrorX = options.mirrorX !== false; // default mirror
        // Hide body landmarks whose visibility (channel 3) is below this threshold.
        // 0 = always draw (legacy behaviour for 3-channel data without visibility).
        this.minVisibility = options.minVisibility !== undefined ? options.minVisibility : 0.5;
        // Smooth playback by lerping between captured frames at the screen refresh rate.
        this.interpolate = options.interpolate !== false;
        this._continuousFrame = 0;
    }

    load(frames) {
        this.frames = frames;
        this.currentFrame = 0;
        this._normalizeBounds();
        this.drawFrame(0);
    }

    // Compute bounds from landmarks we actually draw (exclude legs which MediaPipe
    // often extrapolates off-screen, stretching the bounding box)
    _normalizeBounds() {
        if (!this.frames || this.frames.length === 0) return;

        // Drawn indices: upper body + both hands (33-74) + face (75-82)
        const UPPER_BODY = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24];
        const HANDS = [];
        for (let i = 33; i <= 74; i++) HANDS.push(i);
        const FACE = [75, 76, 77, 78, 79, 80, 81, 82];
        const DRAWN = [...UPPER_BODY, ...HANDS, ...FACE];

        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        for (const frame of this.frames) {
            for (const idx of DRAWN) {
                const pt = frame[idx];
                if (!pt || pt.length < 2) continue;
                const x = pt[0], y = pt[1];
                if (x === 0 && y === 0) continue; // skip missing points
                if (x < minX) minX = x;
                if (y < minY) minY = y;
                if (x > maxX) maxX = x;
                if (y > maxY) maxY = y;
            }
        }
        if (minX === Infinity) return;

        const padX = (maxX - minX) * 0.08 || 10;
        const padY = (maxY - minY) * 0.08 || 10;
        this._bounds = {
            minX: minX - padX,
            minY: minY - padY,
            maxX: maxX + padX,
            maxY: maxY + padY,
        };
    }

    _mapPoint(pt, minVis = 0) {
        if (!this._bounds || !pt || pt.length < 2) return null;
        // Skip missing points (both coords exactly 0)
        if (pt[0] === 0 && pt[1] === 0) return null;
        // Visibility gate: only applies when 4-channel data is provided
        if (minVis > 0 && pt.length >= 4 && pt[3] !== undefined && pt[3] < minVis) return null;

        const { minX, minY, maxX, maxY } = this._bounds;
        const spanX = maxX - minX || 1;
        const spanY = maxY - minY || 1;

        // Preserve aspect ratio — use the larger span to scale both axes equally
        const span = Math.max(spanX, spanY);
        const offsetX = (span - spanX) / 2;
        const offsetY = (span - spanY) / 2;

        let nx = (pt[0] - minX + offsetX) / span;
        const ny = (pt[1] - minY + offsetY) / span;

        if (this.mirrorX) nx = 1 - nx;
        return [nx * this.canvas.width, ny * this.canvas.height];
    }

    play() {
        if (!this.frames || this.frames.length === 0) return;
        this.playing = true;
        this.lastFrameTime = performance.now();
        this._animLoop();
    }

    pause() {
        this.playing = false;
        if (this.animHandle) cancelAnimationFrame(this.animHandle);
    }

    togglePlay() {
        if (this.playing) this.pause();
        else this.play();
    }

    _animLoop() {
        if (!this.playing) return;
        const now = performance.now();
        const dtSec = (now - this.lastFrameTime) / 1000;
        this.lastFrameTime = now;

        // Advance fractional frame index; loop seamlessly. Drawing happens every
        // requestAnimationFrame (~60Hz) regardless of capture fps, so playback is
        // smooth even when only 15 frames per second of pose data exist.
        const N = this.frames.length;
        if (this.interpolate && N > 1) {
            this._continuousFrame = (this._continuousFrame + dtSec * this.fps) % N;
            this.drawFrameAt(this._continuousFrame);
        } else {
            this._continuousFrame = (this._continuousFrame + dtSec * this.fps) % N;
            this.currentFrame = Math.floor(this._continuousFrame);
            this.drawFrame(this.currentFrame);
        }
        this.animHandle = requestAnimationFrame(() => this._animLoop());
    }

    drawFrame(idx) {
        if (!this.frames || !this.frames[idx]) return;
        this._renderFrame(this.frames[idx]);
    }

    drawFrameAt(t) {
        if (!this.frames || this.frames.length === 0) return;
        const N = this.frames.length;
        const idx = Math.max(0, Math.min(N - 1, Math.floor(t)));
        const tFrac = t - idx;
        // Don't interpolate across the loop boundary — lerping from the last
        // captured frame back to the first creates a synthetic "fly back" motion
        // that reads as the sign continuing past where it ended.
        const atBoundary = idx >= N - 1;
        const frame = (this.interpolate && N > 1 && !atBoundary)
            ? this._lerpFrame(this.frames[idx], this.frames[idx + 1], tFrac)
            : this.frames[idx];
        this._renderFrame(frame);
    }

    _renderFrame(frame) {
        const ctx = this.ctx;
        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        // Landmark layout (83 points): 0-32 body, 33-53 left hand, 54-74 right hand, 75-82 face
        this._drawBody(frame);
        this._drawHand(frame, 33, 53);  // left hand
        this._drawHand(frame, 54, 74);  // right hand
    }

    // Linear-interpolate two frames, but treat (0, 0) landmarks as "missing" —
    // never blend a real point with a missing one (would drag toward origin).
    _lerpFrame(a, b, t) {
        if (!a || !b) return a || b;
        const out = new Array(a.length);
        for (let j = 0; j < a.length; j++) {
            const pa = a[j], pb = b[j];
            if (!pa || !pb) { out[j] = pa || pb || [0, 0, 0, 0]; continue; }
            const aMissing = pa[0] === 0 && pa[1] === 0;
            const bMissing = pb[0] === 0 && pb[1] === 0;
            if (aMissing && bMissing) { out[j] = [0, 0, 0, 0]; continue; }
            if (aMissing) { out[j] = pb; continue; }
            if (bMissing) { out[j] = pa; continue; }
            const len = Math.min(pa.length, pb.length);
            const lerped = new Array(len);
            for (let k = 0; k < len; k++) {
                lerped[k] = pa[k] * (1 - t) + pb[k] * t;
            }
            out[j] = lerped;
        }
        return out;
    }

    _drawBody(frame) {
        // MediaPipe's body model always estimates wrist positions, even when its
        // dedicated hand detector finds no hand on that side. If we drew the
        // forearm + wrist-dot anyway, a one-handed sign would visually read as
        // two-handed (a phantom arm-stub ending in what looks like a hand).
        // So gate the forearm bone and body-wrist joint on whether the hand
        // wireframe is actually present (idx 33 = left-hand wrist landmark,
        // idx 54 = right-hand wrist landmark).
        const leftHandDetected = frame[33] && (frame[33][0] !== 0 || frame[33][1] !== 0);
        const rightHandDetected = frame[54] && (frame[54][0] !== 0 || frame[54][1] !== 0);

        const CONNS = [
            [11, 12], [11, 23], [12, 24], [23, 24],          // torso
            [11, 13], [12, 14],                                // upper arms
            [0, 2], [2, 7], [0, 5], [5, 8],                    // face frame
        ];
        if (leftHandDetected)  CONNS.push([13, 15]);           // left forearm
        if (rightHandDetected) CONNS.push([14, 16]);           // right forearm

        const JOINTS = [0, 2, 5, 7, 8, 11, 12, 13, 14, 23, 24];
        if (leftHandDetected)  JOINTS.push(15);
        if (rightHandDetected) JOINTS.push(16);

        this.ctx.strokeStyle = this.bodyColor;
        this.ctx.lineWidth = this.boneWidth;
        this.ctx.lineCap = 'round';

        for (const [a, b] of CONNS) {
            const pa = this._mapPoint(frame[a], this.minVisibility);
            const pb = this._mapPoint(frame[b], this.minVisibility);
            if (!pa || !pb) continue;
            this.ctx.beginPath();
            this.ctx.moveTo(pa[0], pa[1]);
            this.ctx.lineTo(pb[0], pb[1]);
            this.ctx.stroke();
        }

        this.ctx.fillStyle = this.bodyColor;
        for (const idx of JOINTS) {
            const p = this._mapPoint(frame[idx], this.minVisibility);
            if (!p) continue;
            this.ctx.beginPath();
            this.ctx.arc(p[0], p[1], this.jointRadius, 0, Math.PI * 2);
            this.ctx.fill();
        }
    }

    // Hand indices are relative — startIdx = first landmark of this hand
    _drawHand(frame, startIdx, endIdx) {
        // MediaPipe Hands connections (21 landmarks per hand, offset by startIdx)
        const CONNS = [
            [0, 1], [1, 2], [2, 3], [3, 4],
            [0, 5], [5, 6], [6, 7], [7, 8],
            [5, 9], [9, 10], [10, 11], [11, 12],
            [9, 13], [13, 14], [14, 15], [15, 16],
            [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
        ];

        this.ctx.strokeStyle = this.handColor;
        this.ctx.lineWidth = this.boneWidth * 0.7;

        let hasAny = false;
        for (const [a, b] of CONNS) {
            const pa = this._mapPoint(frame[startIdx + a]);
            const pb = this._mapPoint(frame[startIdx + b]);
            if (!pa || !pb) continue;
            hasAny = true;
            this.ctx.beginPath();
            this.ctx.moveTo(pa[0], pa[1]);
            this.ctx.lineTo(pb[0], pb[1]);
            this.ctx.stroke();
        }

        if (!hasAny) return;

        this.ctx.fillStyle = this.handColor;
        const r = this.jointRadius * 0.7;
        for (let i = startIdx; i <= endIdx; i++) {
            const p = this._mapPoint(frame[i]);
            if (!p) continue;
            this.ctx.beginPath();
            this.ctx.arc(p[0], p[1], r, 0, Math.PI * 2);
            this.ctx.fill();
        }
    }
}

window.PosePlayer = PosePlayer;
