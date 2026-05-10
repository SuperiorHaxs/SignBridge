// ══════════════════════════════════════════════════════════════
// PoseOverlay — Real-time upper body + hand skeleton drawn
// on a canvas layered over the video element
// ══════════════════════════════════════════════════════════════

class PoseOverlay {
    constructor(options = {}) {
        this.videoElement = null;
        this.canvas = null;
        this.ctx = null;
        this.pose = null;
        this.isRunning = false;
        this.isInitialized = false;
        this.visible = options.visible !== false; // default on

        // Styling
        this.jointColor = options.jointColor || 'rgba(100, 200, 255, 0.85)';
        this.boneColor = options.boneColor || 'rgba(100, 200, 255, 0.5)';
        this.handJointColor = options.handJointColor || 'rgba(180, 130, 255, 0.9)';
        this.handBoneColor = options.handBoneColor || 'rgba(180, 130, 255, 0.55)';
        this.jointRadius = options.jointRadius || 4;
        this.boneWidth = options.boneWidth || 2.5;

        // Latest results (shared with SignSegmenter if needed)
        this.latestPoseLandmarks = null;
        this.latestHandLandmarks = null;  // from SignSegmenter's Hands

        // Processing canvas (shared with segmenter to avoid double-processing)
        this._processCanvas = null;
        this._processCtx = null;
    }

    async init(videoElement) {
        this.videoElement = videoElement;

        // Create overlay canvas — positioned over the video via CSS
        this.canvas = document.createElement('canvas');
        this.canvas.className = 'pose-overlay-canvas';
        // Mirror the canvas to match the CSS-mirrored video (scaleX(-1) on .camera-view > video)
        this.canvas.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:2;transform:scaleX(-1);';

        // Insert canvas as sibling of video inside the camera-view container
        const container = videoElement.closest('.camera-view') || videoElement.parentElement;
        if (container) {
            container.style.position = 'relative';
            container.appendChild(this.canvas);
        }

        // Processing canvas (downscaled for speed)
        this._processCanvas = document.createElement('canvas');
        this._processCanvas.width = 320;
        this._processCanvas.height = 240;
        this._processCtx = this._processCanvas.getContext('2d');

        // Initialize MediaPipe Pose — sequential init to avoid WASM conflicts
        this.pose = new window.Pose({
            locateFile: (file) => `/static/vendor/mediapipe-pose/${file}`
        });

        this.pose.setOptions({
            modelComplexity: 0,   // 0 = lite — uses pose_landmark_lite.tflite (in packed assets)
            smoothLandmarks: true,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5,
        });

        this.pose.onResults((results) => this._onPoseResults(results));

        // First send() triggers WASM + model loading
        console.log('[PoseOverlay] Loading MediaPipe Pose...');
        const warmupCanvas = document.createElement('canvas');
        warmupCanvas.width = 64;
        warmupCanvas.height = 64;
        warmupCanvas.getContext('2d').fillRect(0, 0, 64, 64);

        await this.pose.send({ image: warmupCanvas });
        console.log('[PoseOverlay] MediaPipe Pose ready');

        this.isInitialized = true;
    }

    start() {
        if (!this.isInitialized) return;
        this.isRunning = true;
        this._processLoop();
        console.log('[PoseOverlay] Started');
    }

    stop() {
        this.isRunning = false;
        this._clearCanvas();
    }

    toggle() {
        this.visible = !this.visible;
        if (!this.visible) this._clearCanvas();
        return this.visible;
    }

    setVisible(v) {
        this.visible = v;
        if (!this.visible) this._clearCanvas();
    }

    // Accept hand landmarks from SignSegmenter (avoids running Hands twice)
    updateHandLandmarks(multiHandLandmarks) {
        this.latestHandLandmarks = multiHandLandmarks;
    }

    // ── Processing loop — ~10fps, offset from segmenter to spread load ──
    _processLoop() {
        if (!this.isRunning) return;

        const video = this.videoElement;
        if (video && video.readyState >= 2 && this.visible) {
            this._processCtx.drawImage(video, 0, 0, 320, 240);
            this.pose.send({ image: this._processCanvas }).catch(() => {});
        }

        if (this.isRunning) {
            setTimeout(() => this._processLoop(), 100);
        }
    }

    _onPoseResults(results) {
        this.latestPoseLandmarks = results.poseLandmarks || null;
        this._draw();
        // Feed debug panel if available
        if (window.updateDebugPanel) {
            window.updateDebugPanel({ pose: !!this.latestPoseLandmarks });
        }
    }

    // Compute the actual video drawn area inside the container (accounts for object-fit: contain letterboxing)
    _getVideoDrawRect() {
        const video = this.videoElement;
        if (!video || !video.videoWidth || !video.videoHeight) return null;

        const rect = video.getBoundingClientRect();
        const containerW = rect.width;
        const containerH = rect.height;
        const videoW = video.videoWidth;
        const videoH = video.videoHeight;

        const containerAspect = containerW / containerH;
        const videoAspect = videoW / videoH;

        let drawW, drawH, offsetX, offsetY;

        if (videoAspect > containerAspect) {
            // Video is wider — black bars top/bottom
            drawW = containerW;
            drawH = containerW / videoAspect;
            offsetX = 0;
            offsetY = (containerH - drawH) / 2;
        } else {
            // Video is taller — black bars left/right
            drawH = containerH;
            drawW = containerH * videoAspect;
            offsetX = (containerW - drawW) / 2;
            offsetY = 0;
        }

        return { drawW, drawH, offsetX, offsetY };
    }

    _draw() {
        if (!this.visible || !this.canvas) return;

        const video = this.videoElement;
        if (!video) return;

        const rect = video.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        const w = rect.width * dpr;
        const h = rect.height * dpr;

        if (this.canvas.width !== w || this.canvas.height !== h) {
            this.canvas.width = w;
            this.canvas.height = h;
        }

        const ctx = this.ctx || (this.ctx = this.canvas.getContext('2d'));
        ctx.clearRect(0, 0, w, h);
        ctx.save();
        ctx.scale(dpr, dpr);

        // Get actual video area within the letterboxed container
        const vr = this._getVideoDrawRect();
        if (!vr) { ctx.restore(); return; }

        const { drawW, drawH, offsetX, offsetY } = vr;

        // Draw upper body pose skeleton
        if (this.latestPoseLandmarks) {
            this._drawUpperBody(ctx, this.latestPoseLandmarks, drawW, drawH, offsetX, offsetY);
        }

        // Draw hand skeletons (from SignSegmenter's MediaPipe Hands)
        if (this.latestHandLandmarks) {
            for (const hand of this.latestHandLandmarks) {
                this._drawHand(ctx, hand, drawW, drawH, offsetX, offsetY);
            }
        }

        ctx.restore();
    }

    // ── Upper body skeleton (no legs) ──
    // MediaPipe Pose landmark indices:
    //  0: nose, 2: left eye, 5: right eye, 7: left ear, 8: right ear
    // 11: left shoulder, 12: right shoulder
    // 13: left elbow, 14: right elbow
    // 15: left wrist, 16: right wrist
    // 23: left hip, 24: right hip
    _drawUpperBody(ctx, landmarks, w, h, ox = 0, oy = 0) {
        const UPPER_BODY_CONNECTIONS = [
            [11, 12], [11, 23], [12, 24], [23, 24],
            [11, 13], [13, 15], [12, 14], [14, 16],
            [0, 2], [2, 7], [0, 5], [5, 8],
        ];

        const UPPER_BODY_JOINTS = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24];

        ctx.strokeStyle = this.boneColor;
        ctx.lineWidth = this.boneWidth;
        ctx.lineCap = 'round';

        for (const [a, b] of UPPER_BODY_CONNECTIONS) {
            const la = landmarks[a];
            const lb = landmarks[b];
            if (!la || !lb || la.visibility < 0.4 || lb.visibility < 0.4) continue;

            ctx.beginPath();
            ctx.moveTo(ox + la.x * w, oy + la.y * h);
            ctx.lineTo(ox + lb.x * w, oy + lb.y * h);
            ctx.stroke();
        }

        ctx.fillStyle = this.jointColor;
        for (const idx of UPPER_BODY_JOINTS) {
            const l = landmarks[idx];
            if (!l || l.visibility < 0.4) continue;
            ctx.beginPath();
            ctx.arc(ox + l.x * w, oy + l.y * h, this.jointRadius, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    // ── Hand skeleton ──
    // MediaPipe Hands: 21 landmarks per hand
    _drawHand(ctx, landmarks, w, h, ox = 0, oy = 0) {
        const HAND_CONNECTIONS = [
            [0, 1], [1, 2], [2, 3], [3, 4],
            [0, 5], [5, 6], [6, 7], [7, 8],
            [5, 9], [9, 10], [10, 11], [11, 12],
            [9, 13], [13, 14], [14, 15], [15, 16],
            [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
        ];

        ctx.strokeStyle = this.handBoneColor;
        ctx.lineWidth = this.boneWidth * 0.8;
        ctx.lineCap = 'round';

        for (const [a, b] of HAND_CONNECTIONS) {
            const la = landmarks[a];
            const lb = landmarks[b];
            if (!la || !lb) continue;

            ctx.beginPath();
            ctx.moveTo(ox + la.x * w, oy + la.y * h);
            ctx.lineTo(ox + lb.x * w, oy + lb.y * h);
            ctx.stroke();
        }

        ctx.fillStyle = this.handJointColor;
        const r = this.jointRadius * 0.7;
        for (const l of landmarks) {
            if (!l) continue;
            ctx.beginPath();
            ctx.arc(ox + l.x * w, oy + l.y * h, r, 0, Math.PI * 2);
            ctx.fill();
        }
    }

    _clearCanvas() {
        if (this.canvas && this.ctx) {
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        }
    }
}

window.PoseOverlay = PoseOverlay;
