/**
 * Live Mode Record - Webcam Recording with Real-Time Sign Detection
 * Adapted from convert.js live demo functionality
 */

// State
let reference = '';
let glosses = [];
let stream = null;
let liveDemoActive = false;
let detectedGlosses = [];
let isSigning = false;
let isProcessing = false;
let signStartTime = null;
let lastMotionTime = null;
let signRecorder = null;
let signChunks = [];
let liveConfig = null;

// Motion detection
let motionDetectionCanvas = null;
let motionDetectionCtx = null;
let previousFrame = null;

// DOM Elements
let webcamVideo, referenceDisplay, glossesDisplay;
let statusBadge, liveStatusDot, liveStatusText, motionScoreValue;
let currentPrediction, predictionConfidence, allPredictions;
let detectedSignsSection, detectedSignsList;
let startBtn, stopBtn, backBtn, finishBtn, restartBtn;

// Initialize on page load
document.addEventListener('DOMContentLoaded', async () => {
    // Check mode - redirect if not Live mode
    if (getMode() !== AppMode.LIVE) {
        window.location.href = '/';
        return;
    }

    // Get data from session storage
    reference = sessionStorage.getItem('liveReference') || '';
    const glossesJson = sessionStorage.getItem('liveGlosses');

    if (!glossesJson || !reference) {
        window.location.href = '/live-setup';
        return;
    }

    glosses = JSON.parse(glossesJson);

    initializeElements();
    await loadConfig();
    await initializeWebcam();
    setupEventListeners();
});

function initializeElements() {
    webcamVideo = document.getElementById('webcam-video');
    referenceDisplay = document.getElementById('reference-display');
    glossesDisplay = document.getElementById('glosses-display');
    statusBadge = document.getElementById('record-status');
    liveStatusDot = document.getElementById('live-status-dot');
    liveStatusText = document.getElementById('live-status-text');
    motionScoreValue = document.getElementById('motion-score-value');
    currentPrediction = document.getElementById('current-prediction');
    predictionConfidence = document.getElementById('prediction-confidence');
    allPredictions = document.getElementById('all-predictions');
    detectedSignsSection = document.getElementById('detected-signs-section');
    detectedSignsList = document.getElementById('detected-signs-list');
    startBtn = document.getElementById('start-btn');
    stopBtn = document.getElementById('stop-btn');
    backBtn = document.getElementById('back-btn');
    finishBtn = document.getElementById('finish-btn');
    restartBtn = document.getElementById('restart-btn');

    // Display reference and glosses
    referenceDisplay.textContent = reference;
    glossesDisplay.textContent = glosses.join(' → ');
}

async function loadConfig() {
    try {
        const response = await fetch('/api/live-config');
        liveConfig = await response.json();
    } catch (error) {
        console.error('Failed to load live config:', error);
        liveConfig = {
            motion_config: {
                cooldown_ms: 1000,
                min_sign_ms: 500,
                max_sign_ms: 5000,
                motion_threshold: 30,
                motion_area_threshold: 0.02
            }
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
        updateStatus('Ready', 'ready');
    } catch (error) {
        console.error('Webcam error:', error);
        updateStatus('Webcam Error', 'error');
        alert('Could not access webcam: ' + error.message);
    }
}

function setupEventListeners() {
    startBtn.addEventListener('click', startRecording);
    stopBtn.addEventListener('click', stopRecording);
    backBtn.addEventListener('click', () => window.location.href = '/live-learn');
    finishBtn.addEventListener('click', finishAndNavigate);
    restartBtn.addEventListener('click', restartRecording);
}

async function restartRecording() {
    console.log('Restarting recording...');

    // Stop any active recording
    if (liveDemoActive) {
        liveDemoActive = false;
        cancelSignRecording();
    }

    // Reset state
    detectedGlosses = [];
    isSigning = false;
    isProcessing = false;
    signStartTime = null;
    lastMotionTime = null;

    // Reset server session
    try {
        await fetch('/api/live-reset', { method: 'POST' });
    } catch (e) {
        console.log('Live reset failed, continuing anyway');
    }

    // Reset UI
    startBtn.disabled = false;
    stopBtn.disabled = true;
    finishBtn.disabled = true;
    updateStatus('Ready', 'ready');
    updateLiveStatus('READY', 'ready');
    currentPrediction.textContent = '';
    predictionConfidence.textContent = 'Waiting for signing...';
    allPredictions.textContent = '';
    detectedSignsSection.style.display = 'none';
    detectedSignsList.innerHTML = '';
}

function updateStatus(text, type) {
    statusBadge.textContent = text;
    statusBadge.className = 'status-badge status-' + type;
}

function updateLiveStatus(text, type) {
    liveStatusText.textContent = text;
    liveStatusDot.className = 'status-dot status-' + type;
}

// ============================================================
// LIVE DEMO MODE (adapted from convert.js)
// ============================================================

async function startRecording() {
    if (!stream) {
        alert('Please wait for webcam to initialize');
        return;
    }

    console.log('Starting live recording...');

    // Reset state
    try {
        await fetch('/api/live-reset', { method: 'POST' });
    } catch (e) {
        console.log('Live reset failed, continuing anyway');
    }

    liveDemoActive = true;
    detectedGlosses = [];
    isSigning = false;
    isProcessing = false;

    // Initialize motion detection
    initMotionDetection();

    // Update UI
    startBtn.disabled = true;
    stopBtn.disabled = false;
    updateStatus('Recording', 'recording');
    updateLiveStatus('READY', 'ready');
    currentPrediction.textContent = '';
    predictionConfidence.textContent = 'Waiting for signing...';
    allPredictions.textContent = '';
    detectedSignsSection.style.display = 'none';

    // Start motion detection loop
    requestAnimationFrame(motionDetectionLoop);
}

async function stopRecording() {
    console.log('Stopping live recording...');
    liveDemoActive = false;

    // Cancel any in-progress sign recording
    cancelSignRecording();

    // Update UI
    startBtn.disabled = false;
    stopBtn.disabled = true;
    updateStatus('Processing...', 'pending');
    updateLiveStatus('STOPPED', 'stopped');

    // Process detected glosses
    if (detectedGlosses.length > 0) {
        await processDetectedGlossesWithLLM();
    } else {
        updateStatus('No signs detected', 'error');
        alert('No signs were detected. Try again and make sure to sign clearly.');
    }
}

async function processDetectedGlossesWithLLM() {
    console.log('Processing detected glosses with LLM...', detectedGlosses);
    updateStatus('Constructing sentence...', 'pending');

    try {
        // Convert detected glosses to the format expected by /api/construct
        const predictions = detectedGlosses.map(g => ({
            top_1: g.gloss,
            confidence: g.confidence,
            top_k: g.top_k || [{ gloss: g.gloss, confidence: g.confidence }]
        }));

        // Call the construct API
        const response = await fetch('/api/construct', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ predictions: predictions })
        });

        const result = await response.json();

        if (result.success) {
            // Store results in sessionStorage for evaluate page
            sessionStorage.setItem('rawSentence', result.raw_sentence);
            sessionStorage.setItem('llmSentence', result.llm_sentence);
            sessionStorage.setItem('segments', JSON.stringify(predictions));
            sessionStorage.setItem('segmentCount', predictions.length.toString());
            // Reference is already in session storage as 'liveReference'
            sessionStorage.setItem('referenceSentence', reference);

            console.log('LLM processing complete:', {
                raw: result.raw_sentence,
                llm: result.llm_sentence
            });

            updateStatus('Complete', 'success');
            finishBtn.disabled = false;

            // Show option to navigate
            showCompletionMessage(result.raw_sentence, result.llm_sentence);

        } else {
            throw new Error(result.error || 'LLM construction failed');
        }

    } catch (error) {
        console.error('LLM processing error:', error);
        updateStatus('Error', 'error');
        alert('Failed to construct sentence: ' + error.message);
    }
}

function showCompletionMessage(rawSentence, llmSentence) {
    detectedSignsSection.style.display = 'block';
    detectedSignsList.innerHTML = `
        <div class="completion-message">
            <p><strong>Detected signs:</strong> ${detectedGlosses.map(g => g.gloss).join(' → ')}</p>
            <p><strong>Raw sentence:</strong> ${rawSentence}</p>
            <p><strong>LLM sentence:</strong> ${llmSentence}</p>
        </div>
    `;
}

function finishAndNavigate() {
    window.location.href = '/evaluate';
}

// ============================================================
// MOTION DETECTION (from convert.js)
// ============================================================

function initMotionDetection() {
    motionDetectionCanvas = document.createElement('canvas');
    motionDetectionCanvas.width = 160;
    motionDetectionCanvas.height = 120;
    motionDetectionCtx = motionDetectionCanvas.getContext('2d', { willReadFrequently: true });
    previousFrame = null;
}

function motionDetectionLoop() {
    if (!liveDemoActive) return;

    if (!webcamVideo || webcamVideo.readyState < 2) {
        requestAnimationFrame(motionDetectionLoop);
        return;
    }

    try {
        // Draw current frame to detection canvas
        motionDetectionCtx.drawImage(webcamVideo, 0, 0, motionDetectionCanvas.width, motionDetectionCanvas.height);
        const currentFrame = motionDetectionCtx.getImageData(0, 0, motionDetectionCanvas.width, motionDetectionCanvas.height);

        let motionScore = 0;
        if (previousFrame) {
            motionScore = calculateMotionScore(previousFrame.data, currentFrame.data);
        }
        previousFrame = currentFrame;

        // Update motion score display
        if (motionScoreValue) {
            motionScoreValue.textContent = motionScore.toFixed(0);
        }

        // Motion detection state machine
        const config = liveConfig?.motion_config || {};
        const motionThreshold = config.motion_area_threshold || 0.02;
        const isMotionDetected = motionScore > motionThreshold * 10000;

        const now = Date.now();

        if (isMotionDetected) {
            lastMotionTime = now;

            if (!isSigning && !isProcessing) {
                // Start signing
                isSigning = true;
                signStartTime = now;
                updateLiveStatus('SIGNING', 'signing');
                console.log('Sign started');
                startSignRecording();
            }
        } else if (isSigning) {
            const cooldownMs = config.cooldown_ms || 1000;
            const timeSinceMotion = now - lastMotionTime;

            if (timeSinceMotion > cooldownMs) {
                const signDuration = now - signStartTime;
                const minSignMs = config.min_sign_ms || 500;

                if (signDuration >= minSignMs) {
                    console.log(`Sign ended after ${signDuration}ms`);
                    stopSignRecordingAndProcess();
                } else {
                    console.log(`Sign too short (${signDuration}ms), ignoring`);
                    cancelSignRecording();
                }
                isSigning = false;
            }
        }

        // Check for max sign duration
        if (isSigning) {
            const maxSignMs = config.max_sign_ms || 5000;
            const signDuration = now - signStartTime;
            if (signDuration >= maxSignMs) {
                console.log(`Sign max duration reached (${signDuration}ms)`);
                stopSignRecordingAndProcess();
                isSigning = false;
            }
        }
    } catch (error) {
        console.error('Error in motion detection loop:', error);
        // Reset state on error to allow recovery
        if (isSigning && !isProcessing) {
            isSigning = false;
            cancelSignRecording();
        }
    }

    // Always schedule next frame, even after errors
    requestAnimationFrame(motionDetectionLoop);
}

function calculateMotionScore(prevData, currData) {
    let diff = 0;
    const length = prevData.length;

    for (let i = 0; i < length; i += 4) {
        const dr = Math.abs(prevData[i] - currData[i]);
        const dg = Math.abs(prevData[i + 1] - currData[i + 1]);
        const db = Math.abs(prevData[i + 2] - currData[i + 2]);
        if (dr + dg + db > 30) {
            diff++;
        }
    }

    return diff;
}

// ============================================================
// SIGN RECORDING (from convert.js)
// ============================================================

function startSignRecording() {
    if (!stream) {
        console.error('Cannot start recording: no stream available');
        isSigning = false;
        return;
    }

    try {
        signChunks = [];
        const mimeType = MediaRecorder.isTypeSupported('video/webm;codecs=vp9')
            ? 'video/webm;codecs=vp9'
            : 'video/webm';

        signRecorder = new MediaRecorder(stream, { mimeType });

        signRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                signChunks.push(event.data);
            }
        };

        signRecorder.onerror = (event) => {
            console.error('MediaRecorder error:', event.error);
            isSigning = false;
            updateLiveStatus('READY', 'ready');
        };

        signRecorder.start(100);
    } catch (error) {
        console.error('Failed to start sign recording:', error);
        isSigning = false;
        updateLiveStatus('READY', 'ready');
    }
}

function cancelSignRecording() {
    if (signRecorder && signRecorder.state !== 'inactive') {
        signRecorder.stop();
    }
    signChunks = [];
    updateLiveStatus('READY', 'ready');
}

async function stopSignRecordingAndProcess() {
    if (!signRecorder || signRecorder.state === 'inactive') {
        updateLiveStatus('READY', 'ready');
        return;
    }

    isProcessing = true;
    updateLiveStatus('PROCESSING', 'processing');

    return new Promise((resolve) => {
        signRecorder.onstop = async () => {
            try {
                if (signChunks.length === 0) {
                    console.log('No sign chunks recorded, skipping processing');
                    return;
                }

                const signBlob = new Blob(signChunks, { type: 'video/webm' });

                const formData = new FormData();
                formData.append('video', signBlob, 'sign.webm');

                const response = await fetch('/api/process-sign', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();

                if (result.success && result.gloss) {
                    // API returns: { success, gloss, confidence, top_k }
                    detectedGlosses.push({
                        gloss: result.gloss,
                        confidence: result.confidence,
                        top_k: result.top_k
                    });

                    // Update UI - show current prediction with animation
                    currentPrediction.textContent = result.gloss.toUpperCase();
                    currentPrediction.classList.add('new');
                    setTimeout(() => currentPrediction.classList.remove('new'), 300);

                    // Show confidence percentage
                    const confPercent = (result.confidence * 100).toFixed(1);
                    let confidenceText = `Confidence: ${confPercent}%`;

                    // Show top-k alternatives if available
                    if (result.top_k && result.top_k.length > 1) {
                        const alts = result.top_k.slice(1).map(p =>
                            `${p.gloss} (${(p.confidence * 100).toFixed(0)}%)`
                        ).join(', ');
                        confidenceText += ` | Alt: ${alts}`;
                    }
                    predictionConfidence.textContent = confidenceText;

                    // Show all detected glosses as styled chips
                    const allGlossesHtml = detectedGlosses.map(g =>
                        `<span class="gloss">${g.gloss}</span>`
                    ).join('');
                    allPredictions.innerHTML = allGlossesHtml;

                    // Update detected signs display
                    updateDetectedSignsDisplay();

                    console.log('Sign detected:', result.gloss, result.confidence, 'top_k:', result.top_k);
                } else if (!result.success) {
                    console.error('Sign processing failed:', result.error);
                }

            } catch (error) {
                console.error('Sign processing error:', error);
            } finally {
                // Always reset state to allow next sign detection
                isProcessing = false;
                updateLiveStatus('READY', 'ready');
                resolve();
            }
        };

        signRecorder.stop();
    });
}

function updateDetectedSignsDisplay() {
    detectedSignsSection.style.display = 'block';
    detectedSignsList.innerHTML = detectedGlosses.map((g, idx) => `
        <span class="detected-sign">
            <span class="sign-num">${idx + 1}</span>
            <span class="sign-gloss">${g.gloss}</span>
            <span class="sign-conf">${(g.confidence * 100).toFixed(0)}%</span>
        </span>
    `).join('');
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
    }
});
