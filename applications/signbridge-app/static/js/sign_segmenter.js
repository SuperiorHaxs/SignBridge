// ══════════════════════════════════════════════════════════════
// SignSegmenter — Real-time sign boundary detection using
// MediaPipe Hands wrist velocity + sliding window fallback
// ══════════════════════════════════════════════════════════════

class SignSegmenter {
    constructor(options = {}) {
        // ── Tunable parameters ──
        this.restVelocityThreshold = options.restVelocityThreshold || 0.012;  // normalized wrist velocity below this = rest
        this.minSignDurationMs     = options.minSignDurationMs     || 400;    // ignore segments shorter than this
        this.maxSignDurationMs     = options.maxSignDurationMs     || 4000;   // force-split segments longer than this
        this.restDurationMs        = options.restDurationMs        || 300;    // must be at rest this long to trigger boundary
        this.slidingWindowMs       = options.slidingWindowMs       || 2500;   // fallback window size for fluent signing
        this.slidingWindowStride   = options.slidingWindowStride   || 800;    // stride for fallback windows
        this.confidenceThreshold   = options.confidenceThreshold   || 0.15;   // skip predictions below this
        this.deduplicateConsecutive = options.deduplicateConsecutive !== false; // skip same gloss twice in a row
        this.velocitySmoothing     = options.velocitySmoothing     || 5;      // frames to average velocity over
        this.autoSendDelayMs       = options.autoSendDelayMs       || 3000;   // auto-send after this much idle time with glosses

        // ── Internal state ──
        this.hands = null;
        this.camera = null;
        this.videoElement = null;
        this.canvasElement = null;
        this.canvasCtx = null;
        this.isRunning = false;
        this.isInitialized = false;

        // ── Tracking state ──
        this.prevWristPositions = { left: null, right: null };
        this.velocityHistory = [];       // recent velocity values for smoothing
        this.currentVelocity = 0;        // smoothed wrist velocity
        this.signingState = 'idle';      // idle | signing | rest | analyzing
        this.signStartTime = null;       // when current sign started
        this.restStartTime = null;       // when rest period started
        this.lastBoundaryTime = null;    // when last sign ended
        this.frameTimestamp = 0;
        this.noHandsCount = 0;          // consecutive frames with no hands
        this.noHandsWarned = false;      // avoid spamming warnings

        // ── Recording state ──
        this.mediaRecorder = null;
        this.recordedChunks = [];
        this.isRecording = false;
        this.signRecordStartTime = null;

        // ── Sliding window fallback ──
        this.noRestSince = null;         // track how long since last rest-based segment
        this.fallbackActive = false;
        this.lastFallbackTime = null;

        // ── Auto-send timer ──
        this._autoSendTimer = null;      // setTimeout handle
        this._idleWithGlossesSince = null;

        // ── Callbacks ──
        this.onSignCaptured = null;      // (videoBlob, metadata) => {}
        this.onStateChange = null;       // (state, info) => {}
        this.onReadyToSend = null;       // (glosses) => {} — called when idle long enough with glosses
        this.onVelocityUpdate = null;    // (velocity) => {} for debug visualization
        this.onHandLandmarks = null;     // (multiHandLandmarks) => {} — for pose overlay

        // ── Collected results ──
        this.collectedGlosses = [];
        this.lastPredictedGloss = null;
    }

    // ── Initialize MediaPipe Hands ──
    async init(videoElement) {
        this.videoElement = videoElement;

        // Create offscreen canvas for MediaPipe processing
        this.canvasElement = document.createElement('canvas');
        this.canvasElement.width = 320;
        this.canvasElement.height = 240;
        this.canvasCtx = this.canvasElement.getContext('2d');

        // Load MediaPipe Hands — must fully initialize before Pose starts
        this.hands = new window.Hands({
            locateFile: (file) => `/static/vendor/mediapipe-hands/${file}`
        });

        // Set options BEFORE first send — the "options before graph loaded" warning
        // is non-fatal (options get queued), but modelComplexity MUST be set before
        // send() or it defaults to 1 (full) which requires a model not in packed assets.
        this.hands.setOptions({
            maxNumHands: 2,
            modelComplexity: 0,   // 0 = lite — uses hand_landmark_lite.tflite (in packed assets)
            minDetectionConfidence: 0.7,
            minTrackingConfidence: 0.6,
        });

        this.hands.onResults((results) => this._onHandResults(results));

        // First send() triggers WASM + model loading
        console.log('[SignSegmenter] Loading MediaPipe Hands...');
        const warmupCanvas = document.createElement('canvas');
        warmupCanvas.width = 64;
        warmupCanvas.height = 64;
        warmupCanvas.getContext('2d').fillRect(0, 0, 64, 64);

        await this.hands.send({ image: warmupCanvas });
        console.log('[SignSegmenter] MediaPipe Hands ready');

        this.isInitialized = true;
    }

    // ── Start processing frames ──
    // trackOnly=true: only run hand detection for overlay visualization, no recording/capture
    start(cameraStream, trackOnly = false) {
        if (!this.isInitialized) {
            console.error('[SignSegmenter] Not initialized — call init() first');
            return;
        }

        this.isRunning = true;
        this.trackOnly = trackOnly;
        this.signingState = 'idle';
        this.collectedGlosses = [];
        this.lastPredictedGloss = null;
        this.prevWristPositions = { left: null, right: null };
        this.velocityHistory = [];
        this.noRestSince = Date.now();
        this.fallbackActive = false;
        this.noHandsCount = 0;
        this.noHandsWarned = false;
        this._sendInFlight = false;
        this._loggedFirstFrame = false;
        this._loggedFirstResult = false;
        this._loggedWaiting = false;
        this._loggedSendError = false;

        // Start the processing loop
        this._processLoop();
        console.log(`[SignSegmenter] Started (trackOnly=${trackOnly})`);
    }

    stop() {
        this.isRunning = false;
        this._stopRecording();
        this.signingState = 'idle';
        this.fallbackActive = false;
        this._cancelAutoSend();
        console.log('[SignSegmenter] Stopped');
    }

    getCollectedGlosses() {
        return [...this.collectedGlosses];
    }

    clearGlosses() {
        this.collectedGlosses = [];
        this.lastPredictedGloss = null;
    }

    // ── Main processing loop — runs at ~10fps to stay lightweight ──
    _processLoop() {
        if (!this.isRunning) return;

        const video = this.videoElement;
        if (video && video.readyState >= 2) {
            if (!this._sendInFlight) {
                // Draw video frame to offscreen canvas (downscaled for speed)
                this.canvasCtx.drawImage(video, 0, 0, 320, 240);
                this._sendInFlight = true;
                this.hands.send({ image: this.canvasElement })
                    .then(() => { this._sendInFlight = false; })
                    .catch((e) => {
                        this._sendInFlight = false;
                        if (!this._loggedSendError) {
                            console.error('[SignSegmenter] hands.send() error:', e);
                            this._loggedSendError = true;
                        }
                    });
            }
            // Log once when processing starts
            if (!this._loggedFirstFrame) {
                this._loggedFirstFrame = true;
                console.log('[SignSegmenter] Processing frames (video ready, size:',
                    video.videoWidth + 'x' + video.videoHeight + ')');
            }
        } else {
            // Log waiting for video
            if (!this._loggedWaiting) {
                this._loggedWaiting = true;
                console.log('[SignSegmenter] Waiting for video readyState...',
                    video ? 'readyState=' + video.readyState : 'no video element');
            }
        }

        // ~10 fps processing (100ms interval) — gentler on CPU
        if (this.isRunning) {
            setTimeout(() => this._processLoop(), 100);
        }
    }

    // ── MediaPipe results callback ──
    _onHandResults(results) {
        this.frameTimestamp = Date.now();
        const landmarks = results.multiHandLandmarks || [];
        const handedness = results.multiHandedness || [];

        // Log first result to confirm pipeline is working
        if (!this._loggedFirstResult) {
            this._loggedFirstResult = true;
            console.log('[SignSegmenter] First MediaPipe result received, hands detected:', landmarks.length);
        }

        // Track consecutive no-hands frames
        if (landmarks.length === 0) {
            this.noHandsCount++;
            if (this.noHandsCount >= 30 && !this.noHandsWarned) {
                this.noHandsWarned = true;
                this._emitState('no_hands', { reason: 'not_detected' });
            }
            // Clear hand landmarks from overlay so stale fingers don't linger
            if (this.onHandLandmarks) {
                this.onHandLandmarks([]);
            }
            this.prevWristPositions = { left: null, right: null };
            if (!this.trackOnly) this._updateState(false);
            return;
        }

        // Hands found — reset no-hands tracking
        if (this.noHandsWarned) {
            this.noHandsWarned = false;
            this._emitState('idle', { reason: 'hands_found' });
        }
        this.noHandsCount = 0;

        // Extract wrist positions (landmark 0 = wrist)
        let leftWrist = null;
        let rightWrist = null;

        for (let i = 0; i < landmarks.length; i++) {
            const hand = landmarks[i];
            const label = handedness[i]?.label;
            const wrist = hand[0]; // landmark 0 = wrist

            // MediaPipe mirrors: "Left" in results = right hand visually
            if (label === 'Left') {
                rightWrist = { x: wrist.x, y: wrist.y, z: wrist.z };
            } else {
                leftWrist = { x: wrist.x, y: wrist.y, z: wrist.z };
            }
        }

        // Compute velocity
        const velocity = this._computeVelocity(leftWrist, rightWrist);
        this.prevWristPositions = { left: leftWrist, right: rightWrist };

        // Smooth velocity
        this.velocityHistory.push(velocity);
        if (this.velocityHistory.length > this.velocitySmoothing) {
            this.velocityHistory.shift();
        }
        this.currentVelocity = this.velocityHistory.reduce((a, b) => a + b, 0) / this.velocityHistory.length;

        if (this.onVelocityUpdate) {
            this.onVelocityUpdate(this.currentVelocity);
        }

        // Share hand landmarks with pose overlay
        if (this.onHandLandmarks) {
            this.onHandLandmarks(landmarks);
        }

        // In trackOnly mode, skip recording/capture state machine
        if (this.trackOnly) return;

        // State machine
        this._updateState(true);
    }

    // ── Compute max wrist velocity across both hands ──
    _computeVelocity(leftWrist, rightWrist) {
        let maxVel = 0;

        if (leftWrist && this.prevWristPositions.left) {
            const dx = leftWrist.x - this.prevWristPositions.left.x;
            const dy = leftWrist.y - this.prevWristPositions.left.y;
            maxVel = Math.max(maxVel, Math.sqrt(dx * dx + dy * dy));
        }

        if (rightWrist && this.prevWristPositions.right) {
            const dx = rightWrist.x - this.prevWristPositions.right.x;
            const dy = rightWrist.y - this.prevWristPositions.right.y;
            maxVel = Math.max(maxVel, Math.sqrt(dx * dx + dy * dy));
        }

        // If no hands detected, velocity = 0 (rest)
        return maxVel;
    }

    // ── State machine: idle → signing → rest → capture ──
    _updateState(handsDetected) {
        const now = this.frameTimestamp;
        const isResting = this.currentVelocity < this.restVelocityThreshold;

        switch (this.signingState) {
            case 'idle':
                if (!isResting && handsDetected) {
                    // Hands started moving — begin signing
                    this.signingState = 'signing';
                    this.signStartTime = now;
                    this.restStartTime = null;
                    this.noRestSince = now;
                    this._startRecording();
                    this._emitState('signing', { velocity: this.currentVelocity });
                }
                break;

            case 'signing':
                if (isResting) {
                    // Might be resting — start rest timer
                    if (!this.restStartTime) {
                        this.restStartTime = now;
                    }

                    const restDuration = now - this.restStartTime;
                    if (restDuration >= this.restDurationMs) {
                        // Confirmed rest — sign ended
                        const signDuration = now - this.signStartTime;
                        if (signDuration >= this.minSignDurationMs) {
                            this.signingState = 'analyzing';
                            this._emitState('analyzing', { signDurationMs: signDuration });
                            this._captureSign(signDuration);
                        } else {
                            // Too short — discard and go back to idle
                            this.signingState = 'idle';
                            this._stopRecording();
                            this._emitState('idle', { reason: 'too_short', durationMs: signDuration });
                        }
                        this.restStartTime = null;
                        this.noRestSince = now;
                    }
                } else {
                    // Still signing — reset rest timer
                    this.restStartTime = null;

                    // Check max sign duration — force split if too long
                    const elapsed = now - this.signStartTime;
                    if (elapsed >= this.maxSignDurationMs) {
                        this.signingState = 'analyzing';
                        this._emitState('analyzing', { signDurationMs: elapsed, reason: 'max_duration' });
                        this._captureSign(elapsed);
                    }
                }
                break;

            case 'analyzing':
                // Wait for inference to complete (set back to idle/signing by _captureSign callback)
                break;
        }

        // ── Phase 2: Sliding window fallback ──
        // If signing continuously with no rest detected for a long time,
        // fall back to fixed windows with overlap
        if (this.signingState === 'signing' && this.noRestSince) {
            const timeSinceLastSegment = now - this.noRestSince;
            if (timeSinceLastSegment >= this.slidingWindowMs) {
                // No rest-based boundary found — force a sliding window capture
                this.fallbackActive = true;
                const elapsed = now - this.signStartTime;
                this._emitState('analyzing', { signDurationMs: elapsed, reason: 'fallback_window' });
                this._captureSign(elapsed, true); // isFallback = true
            }
        }
    }

    // ── Recording management ──
    _startRecording() {
        if (this.isRecording) return;
        if (!this.videoElement || !this.videoElement.srcObject) return;

        const stream = this.videoElement.srcObject;
        this.recordedChunks = [];

        try {
            this.mediaRecorder = new MediaRecorder(stream, { mimeType: 'video/webm' });
        } catch (e) {
            try {
                this.mediaRecorder = new MediaRecorder(stream);
            } catch (e2) {
                console.error('[SignSegmenter] Cannot create MediaRecorder:', e2);
                return;
            }
        }

        this.mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) this.recordedChunks.push(e.data);
        };

        this.mediaRecorder.start(100); // collect data every 100ms for fine-grained segments
        this.isRecording = true;
        this.signRecordStartTime = Date.now();
    }

    _stopRecording() {
        return new Promise((resolve) => {
            if (!this.isRecording || !this.mediaRecorder) {
                resolve(null);
                return;
            }

            this.mediaRecorder.onstop = () => {
                const blob = new Blob(this.recordedChunks, { type: 'video/webm' });
                this.recordedChunks = [];
                this.isRecording = false;
                resolve(blob.size > 0 ? blob : null);
            };

            try {
                if (this.mediaRecorder.state === 'recording') {
                    this.mediaRecorder.stop();
                } else {
                    this.isRecording = false;
                    resolve(null);
                }
            } catch (e) {
                this.isRecording = false;
                resolve(null);
            }
        });
    }

    // ── Capture a completed sign and send for inference ──
    async _captureSign(durationMs, isFallback = false) {
        const blob = await this._stopRecording();

        if (!blob) {
            this.signingState = 'idle';
            this._emitState('idle', { reason: 'no_video' });
            return;
        }

        const metadata = {
            durationMs,
            isFallback,
            velocity: this.currentVelocity,
            timestamp: Date.now(),
        };

        // Notify the app to run inference
        if (this.onSignCaptured) {
            try {
                const result = await this.onSignCaptured(blob, metadata);

                // Phase 2: Confidence filtering + deduplication
                if (result && result.success) {
                    const dominated = this._shouldFilter(result);
                    if (!dominated) {
                        this.collectedGlosses.push({
                            gloss: result.gloss,
                            confidence: result.confidence,
                            top_k: result.top_k || [],
                            metadata,
                        });
                        this.lastPredictedGloss = result.gloss.toLowerCase();
                    }
                }
            } catch (e) {
                console.error('[SignSegmenter] Inference callback error:', e);
            }
        }

        // Return to appropriate state
        // If velocity is still high, go straight to signing (re-start recording)
        if (this.isRunning) {
            if (this.currentVelocity >= this.restVelocityThreshold) {
                this.signingState = 'signing';
                this.signStartTime = Date.now();
                this._startRecording();
                this._emitState('signing', { reason: 'continued' });
            } else {
                this.signingState = 'idle';
                this._emitState('idle', { reason: 'rest_after_capture' });
            }
        }
    }

    // ── Phase 2: Confidence filtering + deduplication ──
    _shouldFilter(result) {
        // Skip low confidence
        if (result.confidence < this.confidenceThreshold) {
            console.log(`[SignSegmenter] Filtered: ${result.gloss} (confidence ${(result.confidence * 100).toFixed(1)}% < ${(this.confidenceThreshold * 100).toFixed(1)}%)`);
            return true;
        }

        // Deduplicate consecutive same-gloss predictions
        if (this.deduplicateConsecutive && this.lastPredictedGloss === result.gloss.toLowerCase()) {
            console.log(`[SignSegmenter] Filtered duplicate: ${result.gloss}`);
            return true;
        }

        return false;
    }

    _emitState(state, info = {}) {
        // In trackOnly mode, don't emit state changes (no recording, no auto-send)
        if (this.trackOnly) return;

        if (this.onStateChange) {
            this.onStateChange(state, info);
        }

        // ── Auto-send logic ──
        // When idle with collected glosses, start a countdown to auto-send
        if (state === 'idle' && this.collectedGlosses.length > 0) {
            this._startAutoSend();
        } else if (state === 'signing' || state === 'analyzing') {
            // Signing resumed — cancel any pending auto-send
            this._cancelAutoSend();
        }
    }

    _startAutoSend() {
        if (this._autoSendTimer) return; // already running
        this._idleWithGlossesSince = Date.now();

        this._autoSendTimer = setTimeout(() => {
            this._autoSendTimer = null;
            if (this.collectedGlosses.length > 0 && this.signingState === 'idle' && this.isRunning) {
                console.log(`[SignSegmenter] Auto-send: ${this.collectedGlosses.length} glosses after ${this.autoSendDelayMs}ms idle`);
                if (this.onReadyToSend) {
                    this.onReadyToSend([...this.collectedGlosses]);
                }
            }
        }, this.autoSendDelayMs);
    }

    _cancelAutoSend() {
        if (this._autoSendTimer) {
            clearTimeout(this._autoSendTimer);
            this._autoSendTimer = null;
        }
        this._idleWithGlossesSince = null;
    }
}

// Export for use by app.js
window.SignSegmenter = SignSegmenter;
