/**
 * Closed Captions Mode - Real-time ASL to English Captioning
 *
 * Provides continuous sign detection with context-aware caption generation.
 * Adapted from live_record.js with caption polling and LLM context support.
 */

// ============================================================
// STATE
// ============================================================

let stream = null;
let isRunning = false;
let isSigning = false;
let isProcessing = false;
let signStartTime = null;
let lastMotionTime = null;
let signRecorder = null;
let signChunks = [];
let config = null;
let detectedGlosses = [];

// Caption polling
let captionPollInterval = null;
let lastCaption = '';

// Motion detection
let motionCanvas = null;
let motionCtx = null;
let previousFrame = null;
let frameCount = 0;  // For skipping initial frames

// DOM Elements
let webcamVideo, captionText, statusDot, statusText, glossCount;
let startBtn, stopBtn, resetBtn, glossesList, glossesPanel;
let warmupOverlay, warmupCountdown, readyIndicator;

// ============================================================
// INITIALIZATION
// ============================================================

document.addEventListener('DOMContentLoaded', async () => {
    initializeElements();
    await loadConfig();
    await initializeWebcam();
    setupEventListeners();
});

function initializeElements() {
    webcamVideo = document.getElementById('cc-webcam');
    captionText = document.getElementById('cc-caption-text');
    statusDot = document.getElementById('cc-status-dot');
    statusText = document.getElementById('cc-status-text');
    glossCount = document.getElementById('cc-gloss-count');
    startBtn = document.getElementById('cc-start-btn');
    stopBtn = document.getElementById('cc-stop-btn');
    resetBtn = document.getElementById('cc-reset-btn');
    glossesList = document.getElementById('cc-glosses-list');
    glossesPanel = document.getElementById('cc-glosses-panel');
    warmupOverlay = document.getElementById('cc-warmup-overlay');
    warmupCountdown = document.getElementById('cc-warmup-countdown');
    readyIndicator = document.getElementById('cc-ready-indicator');
}

async function loadConfig() {
    try {
        const response = await fetch('/api/cc/config');
        config = await response.json();
        console.log('[CC] Config loaded:', config);
    } catch (error) {
        console.error('[CC] Failed to load config:', error);
        config = {
            motion_config: {
                cooldown_ms: 1000,
                min_sign_ms: 500,
                max_sign_ms: 5000,
                motion_threshold: 30,
                motion_area_threshold: 0.02
            },
            caption_buffer_size: 3
        };
    }
}

async function initializeWebcam() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
            audio: false
        });
        webcamVideo.srcObject = stream;
        await webcamVideo.play();
        updateStatus('ready', 'Ready');
        console.log('[CC] Webcam initialized');
    } catch (error) {
        console.error('[CC] Webcam error:', error);
        updateStatus('error', 'Webcam Error');
        alert('Could not access webcam: ' + error.message);
    }
}

function setupEventListeners() {
    startBtn.addEventListener('click', startCaptioning);
    stopBtn.addEventListener('click', stopCaptioning);
    resetBtn.addEventListener('click', resetSession);
}

// ============================================================
// CAPTIONING CONTROL
// ============================================================

async function startCaptioning() {
    if (isRunning) return;

    try {
        // Start backend service
        const response = await fetch('/api/cc/start', { method: 'POST' });
        const result = await response.json();

        if (!result.success) {
            console.error('[CC] Failed to start service:', result.error);
            return;
        }

        isRunning = true;
        startBtn.disabled = true;
        stopBtn.disabled = false;

        // Initialize motion detection
        initMotionDetection();

        // Start motion detection loop
        requestAnimationFrame(motionDetectionLoop);

        // Start caption polling
        startCaptionPolling();

        // Show glosses panel
        glossesPanel.style.display = 'block';

        updateStatus('ready', 'Listening...');
        console.log('[CC] Captioning started');

    } catch (error) {
        console.error('[CC] Error starting captioning:', error);
    }
}

async function stopCaptioning() {
    if (!isRunning) return;

    isRunning = false;

    // Cancel any active recording
    if (signRecorder && signRecorder.state !== 'inactive') {
        signRecorder.stop();
    }

    // Stop caption polling
    stopCaptionPolling();

    // Stop backend service and get final caption
    try {
        const response = await fetch('/api/cc/stop', { method: 'POST' });
        const result = await response.json();

        // Update with final caption (includes any flushed remaining glosses)
        if (result.final_caption) {
            updateCaption(result.final_caption);
        }
    } catch (error) {
        console.error('[CC] Error stopping service:', error);
    }

    // Hide any overlays
    warmupOverlay.style.display = 'none';
    readyIndicator.style.display = 'none';

    startBtn.disabled = false;
    stopBtn.disabled = true;
    updateStatus('ready', 'Stopped');
    console.log('[CC] Captioning stopped');
}

async function resetSession() {
    // Stop if running
    if (isRunning) {
        await stopCaptioning();
    }

    // Reset backend
    try {
        await fetch('/api/cc/reset', { method: 'POST' });
    } catch (error) {
        console.error('[CC] Error resetting session:', error);
    }

    // Reset UI
    detectedGlosses = [];
    lastCaption = '';
    captionText.textContent = 'Start signing to see captions...';
    glossCount.textContent = '0';
    glossesList.innerHTML = '';
    glossesPanel.style.display = 'none';

    updateStatus('ready', 'Ready');
    console.log('[CC] Session reset');
}

// ============================================================
// MOTION DETECTION
// ============================================================

function initMotionDetection() {
    motionCanvas = document.createElement('canvas');
    motionCanvas.width = 160;
    motionCanvas.height = 120;
    motionCtx = motionCanvas.getContext('2d', { willReadFrequently: true });
    previousFrame = null;
    frameCount = 0;  // Reset frame counter

    // Show warmup overlay with countdown
    warmupOverlay.style.display = 'flex';
    readyIndicator.style.display = 'none';
    warmupCountdown.textContent = '3';
}

function motionDetectionLoop() {
    if (!isRunning) return;

    if (!webcamVideo || webcamVideo.readyState < 2) {
        requestAnimationFrame(motionDetectionLoop);
        return;
    }

    try {
        const now = Date.now();
        const motionConfig = config.motion_config;

        // Skip first 90 frames (~3 seconds at 30fps) to let camera stabilize
        frameCount++;
        const warmupFrames = motionConfig.warmup_frames || 90;
        if (frameCount <= warmupFrames) {
            // Update countdown display (3, 2, 1)
            const secondsLeft = Math.ceil((warmupFrames - frameCount) / 30);
            if (warmupCountdown.textContent !== String(secondsLeft)) {
                warmupCountdown.textContent = secondsLeft;
            }
            requestAnimationFrame(motionDetectionLoop);
            return;
        }

        // Warmup just ended - hide overlay and show ready indicator
        if (frameCount === warmupFrames + 1) {
            warmupOverlay.style.display = 'none';
            readyIndicator.style.display = 'block';
            // Ready indicator auto-hides via CSS animation
            setTimeout(() => {
                readyIndicator.style.display = 'none';
            }, 2000);
        }

        // Draw current frame
        motionCtx.drawImage(webcamVideo, 0, 0, motionCanvas.width, motionCanvas.height);
        const currentFrame = motionCtx.getImageData(0, 0, motionCanvas.width, motionCanvas.height);

        let motionScore = 0;
        if (previousFrame) {
            motionScore = calculateMotionScore(previousFrame.data, currentFrame.data);
        }
        previousFrame = currentFrame;

        // motionScore is now a ratio (0-1) of pixels that changed significantly
        // motion_area_threshold is the minimum fraction of pixels that must change (e.g., 0.02 = 2%)
        const threshold = motionConfig.motion_area_threshold || 0.02;

        // Detect motion
        if (motionScore > threshold) {
            lastMotionTime = now;

            if (!isSigning && !isProcessing) {
                // Start signing
                isSigning = true;
                signStartTime = now;
                updateStatus('signing', 'Signing...');
                startSignRecording();
            }
        } else if (isSigning) {
            const cooldownMs = motionConfig.cooldown_ms || 1000;
            const timeSinceMotion = now - lastMotionTime;

            if (timeSinceMotion >= cooldownMs) {
                const signDuration = now - signStartTime;
                const minSignMs = motionConfig.min_sign_ms || 500;

                if (signDuration >= minSignMs) {
                    console.log(`[CC] Sign ended after ${signDuration}ms`);
                    stopSignRecordingAndProcess();
                } else {
                    console.log(`[CC] Sign too short (${signDuration}ms), ignoring`);
                    cancelSignRecording();
                }
                isSigning = false;
            }
        }

        // Check max duration
        if (isSigning) {
            const maxSignMs = motionConfig.max_sign_ms || 5000;
            const signDuration = now - signStartTime;
            if (signDuration >= maxSignMs) {
                console.log(`[CC] Sign max duration reached (${signDuration}ms)`);
                stopSignRecordingAndProcess();
                isSigning = false;
            }
        }

    } catch (error) {
        console.error('[CC] Motion detection error:', error);
    }

    requestAnimationFrame(motionDetectionLoop);
}

function calculateMotionScore(prevData, currData) {
    let changedPixels = 0;
    const length = prevData.length;

    // Per-pixel threshold to filter camera noise
    const pixelThreshold = config.motion_config.motion_threshold || 30;

    // Compare every 4th pixel (RGBA) for speed
    for (let i = 0; i < length; i += 16) {
        const rDiff = Math.abs(prevData[i] - currData[i]);
        const gDiff = Math.abs(prevData[i + 1] - currData[i + 1]);
        const bDiff = Math.abs(prevData[i + 2] - currData[i + 2]);
        const avgDiff = (rDiff + gDiff + bDiff) / 3;

        // Only count pixels that changed significantly (above noise threshold)
        if (avgDiff > pixelThreshold) {
            changedPixels++;
        }
    }

    // Return percentage of changed pixels (0-1 range)
    const totalSampledPixels = length / 16;
    return changedPixels / totalSampledPixels;
}

// ============================================================
// SIGN RECORDING
// ============================================================

function startSignRecording() {
    if (!stream) {
        console.error('[CC] No stream available');
        isSigning = false;
        return;
    }

    try {
        signChunks = [];
        signRecorder = new MediaRecorder(stream, {
            mimeType: 'video/webm;codecs=vp9'
        });

        signRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                signChunks.push(event.data);
            }
        };

        signRecorder.start();
        console.log('[CC] Sign recording started');
    } catch (error) {
        console.error('[CC] Error starting recording:', error);
        isSigning = false;
    }
}

function cancelSignRecording() {
    if (signRecorder && signRecorder.state !== 'inactive') {
        signRecorder.stop();
    }
    signChunks = [];
    updateStatus('ready', 'Listening...');
}

async function stopSignRecordingAndProcess() {
    if (!signRecorder || signRecorder.state === 'inactive') {
        updateStatus('ready', 'Listening...');
        return;
    }

    isProcessing = true;
    updateStatus('processing', 'Processing...');

    return new Promise((resolve) => {
        signRecorder.onstop = async () => {
            try {
                const blob = new Blob(signChunks, { type: 'video/webm' });
                signChunks = [];

                if (blob.size < 1000) {
                    console.log('[CC] Recording too small, ignoring');
                    updateStatus('ready', 'Listening...');
                    isProcessing = false;
                    resolve();
                    return;
                }

                // Send to backend
                const formData = new FormData();
                formData.append('video', blob, 'sign.webm');

                const response = await fetch('/api/cc/process-sign', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();

                if (result.success) {
                    console.log(`[CC] Detected: ${result.gloss} (${(result.confidence * 100).toFixed(1)}%)`);

                    // Add to local list
                    detectedGlosses.push({
                        gloss: result.gloss,
                        confidence: result.confidence
                    });

                    // Update UI
                    updateGlossesList();
                    glossCount.textContent = detectedGlosses.length;
                } else {
                    console.error('[CC] Processing failed:', result.error);
                }

            } catch (error) {
                console.error('[CC] Error processing sign:', error);
            }

            isProcessing = false;
            updateStatus('ready', 'Listening...');
            resolve();
        };

        signRecorder.stop();
    });
}

// ============================================================
// CAPTION POLLING
// ============================================================

function startCaptionPolling() {
    // Poll every 500ms
    captionPollInterval = setInterval(pollCaption, 500);
    console.log('[CC] Caption polling started');
}

function stopCaptionPolling() {
    if (captionPollInterval) {
        clearInterval(captionPollInterval);
        captionPollInterval = null;
    }
    console.log('[CC] Caption polling stopped');
}

async function pollCaption() {
    try {
        const response = await fetch('/api/cc/get-caption');
        const result = await response.json();

        if (result.caption && result.caption !== lastCaption) {
            lastCaption = result.caption;
            updateCaption(result.caption);
        }

        // Update gloss count from server
        if (result.gloss_count !== undefined) {
            glossCount.textContent = result.gloss_count;
        }

    } catch (error) {
        console.error('[CC] Caption poll error:', error);
    }
}

function updateCaption(caption) {
    captionText.classList.add('updating');
    captionText.textContent = caption;

    // Remove updating class after animation
    setTimeout(() => {
        captionText.classList.remove('updating');
    }, 300);
}

// ============================================================
// UI UPDATES
// ============================================================

function updateStatus(status, text) {
    statusDot.className = 'cc-status-dot ' + status;
    statusText.textContent = text;
}

function updateGlossesList() {
    // Show last 10 glosses
    const recentGlosses = detectedGlosses.slice(-10).reverse();

    glossesList.innerHTML = recentGlosses.map(g => `
        <div class="cc-gloss-item">
            <span>${g.gloss}</span>
            <span class="cc-gloss-confidence">${(g.confidence * 100).toFixed(0)}%</span>
        </div>
    `).join('');
}

// ============================================================
// CLEANUP
// ============================================================

window.addEventListener('beforeunload', () => {
    if (isRunning) {
        stopCaptioning();
    }
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
});
