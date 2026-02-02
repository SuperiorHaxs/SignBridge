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
let signRecorder = null;
let signChunks = [];
let config = null;
let detectedGlosses = [];

// Caption polling
let captionPollInterval = null;
let lastCaption = '';

// Motion detection - using common MotionDetector module
let motionDetector = null;

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
    // Using common camera UI component IDs (cc- prefix)
    webcamVideo = document.getElementById('cc-video');
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

        // Initialize motion detection using common module
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
// MOTION DETECTION (using common MotionDetector module)
// ============================================================

function initMotionDetection() {
    // Create motion detector with config from server
    const motionConfig = config.motion_config || {};
    motionDetector = new MotionDetector(motionConfig);

    // Set up callbacks
    motionDetector.onWarmupProgress = (secondsLeft, frameCount, totalFrames) => {
        warmupOverlay.style.display = 'flex';
        if (warmupCountdown.textContent !== String(secondsLeft)) {
            warmupCountdown.textContent = secondsLeft;
        }
    };

    motionDetector.onWarmupComplete = () => {
        warmupOverlay.style.display = 'none';
        readyIndicator.style.display = 'block';
        setTimeout(() => {
            readyIndicator.style.display = 'none';
        }, 2000);
    };

    motionDetector.onSignStart = () => {
        updateStatus('signing', 'Signing...');
        startSignRecording();
    };

    motionDetector.onSignEnd = (duration) => {
        console.log(`[CC] Sign ended after ${duration}ms`);
        stopSignRecordingAndProcess();
    };

    motionDetector.onSignTooShort = (duration) => {
        console.log(`[CC] Sign too short (${duration}ms), ignoring`);
        cancelSignRecording();
    };

    // Show warmup overlay
    warmupOverlay.style.display = 'flex';
    readyIndicator.style.display = 'none';
    warmupCountdown.textContent = '3';
}

function motionDetectionLoop() {
    if (!isRunning) return;

    try {
        motionDetector.processFrame(webcamVideo);
    } catch (error) {
        console.error('[CC] Motion detection error:', error);
    }

    requestAnimationFrame(motionDetectionLoop);
}

// ============================================================
// SIGN RECORDING
// ============================================================

function startSignRecording() {
    if (!stream) {
        console.error('[CC] No stream available');
        motionDetector.reset();
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
        updateStatus('recording', 'Recording sign...');
    } catch (error) {
        console.error('[CC] Error starting recording:', error);
        motionDetector.reset();
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

    motionDetector.setProcessing(true);
    updateStatus('processing', 'Processing...');

    return new Promise((resolve) => {
        signRecorder.onstop = async () => {
            try {
                const blob = new Blob(signChunks, { type: 'video/webm' });
                signChunks = [];

                if (blob.size < 1000) {
                    console.log('[CC] Recording too small, ignoring');
                    updateStatus('ready', 'Listening...');
                    motionDetector.setProcessing(false);
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
                    updateStatus('error', 'Processing failed: ' + (result.error || 'unknown'));
                }

            } catch (error) {
                console.error('[CC] Error processing sign:', error);
                updateStatus('error', 'Network error: ' + error.message);
            }

            motionDetector.setProcessing(false);
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

        // Update gloss list with used/dropped status
        if (result.gloss_status && result.gloss_status.length > 0) {
            updateGlossesListWithStatus(result.gloss_status);
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
    statusDot.className = 'camera-status-dot ' + status;
    statusText.textContent = text;
}

function updateGlossesList() {
    // Show last 10 glosses (without status info - basic display)
    const recentGlosses = detectedGlosses.slice(-10).reverse();

    glossesList.innerHTML = recentGlosses.map(g => `
        <div class="camera-gloss-item">
            <span>${g.gloss}</span>
            <span class="camera-gloss-confidence">${(g.confidence * 100).toFixed(0)}%</span>
        </div>
    `).join('');
}

function updateGlossesListWithStatus(glossStatus) {
    // Show glosses with used (green) / dropped (red) / pending (yellow) status
    glossesList.innerHTML = glossStatus.map(g => {
        let statusClass, statusIcon;
        if (g.used === null || g.used === undefined) {
            // Pending - not yet processed by LLM
            statusClass = 'gloss-pending';
            statusIcon = '○';
        } else if (g.used) {
            // Used in sentence
            statusClass = 'gloss-used';
            statusIcon = '✓';
        } else {
            // Dropped/skipped
            statusClass = 'gloss-dropped';
            statusIcon = '✗';
        }
        return `
            <div class="camera-gloss-item ${statusClass}">
                <span class="gloss-status-icon">${statusIcon}</span>
                <span>${g.gloss}</span>
                <span class="camera-gloss-confidence">${(g.confidence * 100).toFixed(0)}%</span>
            </div>
        `;
    }).join('');

    // Show the panel
    glossesPanel.style.display = 'block';
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
