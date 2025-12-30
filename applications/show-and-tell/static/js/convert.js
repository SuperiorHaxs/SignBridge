/**
 * Phase 1: Convert Video to Pose
 * Handles webcam/upload/pose mode toggle, recording, and pose conversion
 * Supports both Live and Demo modes
 */

// State management
const state = {
    inputMode: 'webcam', // 'webcam', 'upload', or 'pose'
    stream: null,
    mediaRecorder: null,
    recordedChunks: [],
    isRecording: false,
    recordingStartTime: null,
    recordingTimer: null,
    uploadedFile: null,
    uploadedPoseFile: null,
    hasRecording: false,
    // Demo mode state
    samples: [],
    selectedSample: null,
    // Live demo state
    liveDemoActive: false,
    liveConfig: null,
    previousFrame: null,
    motionDetectionCanvas: null,
    motionDetectionCtx: null,
    isSigning: false,
    signStartTime: null,
    signFrames: [],
    lastMotionTime: 0,
    detectedGlosses: [],
    isProcessing: false
};

// DOM Elements - Live Mode
const liveElements = {
    container: document.getElementById('live-mode-content'),
    referenceSentence: document.getElementById('reference-sentence'),
    modeWebcam: document.getElementById('mode-webcam'),
    modeUpload: document.getElementById('mode-upload'),
    modePose: document.getElementById('mode-pose'),
    webcamMode: document.getElementById('webcam-mode'),
    uploadMode: document.getElementById('upload-mode'),
    poseMode: document.getElementById('pose-mode'),
    inputPanelTitle: document.getElementById('input-panel-title'),
    webcamVideo: document.getElementById('webcam-video'),
    webcamOverlay: document.getElementById('webcam-overlay'),
    overlayText: document.getElementById('overlay-text'),
    btnRecordToggle: document.getElementById('btn-record-toggle'),
    recordIcon: document.getElementById('record-icon'),
    recordLabel: document.getElementById('record-label'),
    recordingIndicator: document.getElementById('recording-indicator'),
    recordingTime: document.getElementById('recording-time'),
    uploadArea: document.getElementById('upload-area'),
    uploadInput: document.getElementById('upload-input'),
    uploadedVideo: document.getElementById('uploaded-video'),
    uploadControls: document.getElementById('upload-controls'),
    btnChangeVideo: document.getElementById('btn-change-video'),
    poseUploadArea: document.getElementById('pose-upload-area'),
    poseInput: document.getElementById('pose-input'),
    poseFileInfo: document.getElementById('pose-file-info'),
    poseFilename: document.getElementById('pose-filename'),
    poseControls: document.getElementById('pose-controls'),
    btnChangePose: document.getElementById('btn-change-pose'),
    btnConvert: document.getElementById('btn-convert'),
    btnContinuePose: document.getElementById('btn-continue-pose'),
    convertHint: document.getElementById('convert-hint'),
    inputStatus: document.getElementById('input-status'),
    // Live demo overlay elements
    liveStatusOverlay: document.getElementById('live-status-overlay'),
    livePredictionOverlay: document.getElementById('live-prediction-overlay'),
    liveStatusDot: document.getElementById('live-status-dot'),
    liveStatusText: document.getElementById('live-status-text'),
    motionScoreValue: document.getElementById('motion-score-value'),
    currentPrediction: document.getElementById('current-prediction'),
    predictionConfidence: document.getElementById('prediction-confidence'),
    allPredictions: document.getElementById('all-predictions')
};

// DOM Elements - Demo Mode
const demoElements = {
    container: document.getElementById('demo-mode-content'),
    referenceSentence: document.getElementById('demo-reference-sentence'),
    samplesGrid: document.getElementById('samples-grid'),
    btnContinue: document.getElementById('btn-demo-continue'),
    hint: document.getElementById('demo-hint')
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', initialize);

async function initialize() {
    // Wait for mode system to initialize
    window.addEventListener('modeInitialized', (e) => {
        setupForMode(e.detail.mode);
    });

    // If mode already initialized (mode.js loaded first)
    if (typeof getMode === 'function') {
        setupForMode(getMode());
    }
}

function setupForMode(mode) {
    if (mode === 'demo') {
        setupDemoMode();
    } else {
        // Both 'live' and 'live_detailed' use live mode UI
        setupLiveMode();
    }
}

// ============================================================================
// LIVE MODE
// ============================================================================

async function setupLiveMode() {
    // Show/hide appropriate content
    liveElements.container.style.display = 'block';
    demoElements.container.style.display = 'none';

    // Check mode to show/hide appropriate input options
    const isLiveDetailed = typeof isLiveDetailedMode === 'function' && isLiveDetailedMode();
    const isFast = typeof isFastMode === 'function' && isFastMode();

    if (isFast) {
        // Live (fast) mode: Only webcam for real-time detection
        liveElements.modeWebcam.style.display = '';
        liveElements.modeUpload.style.display = 'none';
        liveElements.modePose.style.display = 'none';
    } else if (isLiveDetailed) {
        // Live Demo mode: Hide webcam, show upload and pose
        liveElements.modeWebcam.style.display = 'none';
        liveElements.modeUpload.style.display = '';
        liveElements.modePose.style.display = '';
    } else {
        // Show all options
        liveElements.modeWebcam.style.display = '';
        liveElements.modeUpload.style.display = '';
        liveElements.modePose.style.display = '';
    }

    // Load saved reference sentence if exists
    const savedRef = sessionStorage.getItem('referenceSentence');
    if (savedRef) {
        liveElements.referenceSentence.value = savedRef;
    }

    // Save reference sentence on change
    liveElements.referenceSentence.addEventListener('input', () => {
        sessionStorage.setItem('referenceSentence', liveElements.referenceSentence.value);
    });

    // Mode toggle listeners
    liveElements.modeWebcam.addEventListener('click', () => switchInputMode('webcam'));
    liveElements.modeUpload.addEventListener('click', () => switchInputMode('upload'));
    liveElements.modePose.addEventListener('click', () => switchInputMode('pose'));

    // Webcam controls - single toggle button
    liveElements.btnRecordToggle.addEventListener('click', toggleRecording);

    // Upload controls
    liveElements.uploadArea.addEventListener('click', () => liveElements.uploadInput.click());
    liveElements.uploadArea.addEventListener('dragover', handleDragOver);
    liveElements.uploadArea.addEventListener('dragleave', handleDragLeave);
    liveElements.uploadArea.addEventListener('drop', handleVideoDrop);
    liveElements.uploadInput.addEventListener('change', handleVideoSelect);
    liveElements.btnChangeVideo.addEventListener('click', clearUploadedVideo);

    // Pose file controls
    liveElements.poseUploadArea.addEventListener('click', () => liveElements.poseInput.click());
    liveElements.poseUploadArea.addEventListener('dragover', handleDragOver);
    liveElements.poseUploadArea.addEventListener('dragleave', handlePoseDragLeave);
    liveElements.poseUploadArea.addEventListener('drop', handlePoseDrop);
    liveElements.poseInput.addEventListener('change', handlePoseSelect);
    liveElements.btnChangePose.addEventListener('click', clearUploadedPose);

    // Convert/Continue buttons
    liveElements.btnConvert.addEventListener('click', convertVideo);
    liveElements.btnContinuePose.addEventListener('click', continueToPoseFile);

    // Load live demo configuration (for Live mode + webcam)
    await loadLiveConfig();

    // In Live Demo mode, default to upload; in Live mode, default to webcam
    if (isLiveDetailed) {
        await switchInputMode('upload');
    } else {
        await initializeWebcam();
    }
}

async function switchInputMode(mode) {
    state.inputMode = mode;

    // Update toggle buttons
    liveElements.modeWebcam.classList.toggle('active', mode === 'webcam');
    liveElements.modeUpload.classList.toggle('active', mode === 'upload');
    liveElements.modePose.classList.toggle('active', mode === 'pose');

    // Show/hide mode-specific content
    liveElements.webcamMode.style.display = mode === 'webcam' ? 'block' : 'none';
    liveElements.uploadMode.style.display = mode === 'upload' ? 'block' : 'none';
    liveElements.poseMode.style.display = mode === 'pose' ? 'block' : 'none';

    // Update panel title
    const titles = {
        webcam: 'Webcam Capture',
        upload: 'Upload Video',
        pose: 'Upload Pose File'
    };
    liveElements.inputPanelTitle.textContent = titles[mode];

    // Handle webcam stream
    if (mode === 'webcam') {
        await initializeWebcam();
    } else {
        stopWebcam();
    }

    // Update button visibility
    updateLiveButtonState();
}

async function initializeWebcam() {
    try {
        updateStatus('Initializing...', 'pending');

        const constraints = {
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user'
            },
            audio: false
        };

        state.stream = await navigator.mediaDevices.getUserMedia(constraints);
        liveElements.webcamVideo.srcObject = state.stream;

        updateStatus('Ready', 'success');
        liveElements.btnRecordToggle.disabled = false;

    } catch (error) {
        console.error('Webcam initialization failed:', error);
        updateStatus('Error', 'error');
        showOverlay('Failed to access webcam. Please allow camera permissions.');
    }
}

function stopWebcam() {
    if (state.stream) {
        state.stream.getTracks().forEach(track => track.stop());
        state.stream = null;
    }
}

function toggleRecording() {
    // Toggle between start and stop based on current state
    if (state.isRecording || state.liveDemoActive) {
        stopRecording();
    } else {
        startRecording();
    }
}

function updateRecordButton(isRecording) {
    if (isRecording) {
        liveElements.btnRecordToggle.classList.remove('btn-primary');
        liveElements.btnRecordToggle.classList.add('btn-danger');
        liveElements.recordIcon.innerHTML = '&#9632;'; // Stop icon (square)
        liveElements.recordLabel.textContent = 'Stop Recording';
    } else {
        liveElements.btnRecordToggle.classList.remove('btn-danger');
        liveElements.btnRecordToggle.classList.add('btn-primary');
        liveElements.recordIcon.innerHTML = '&#9679;'; // Record icon (circle)
        liveElements.recordLabel.textContent = 'Start Recording';
    }
}

function startRecording() {
    if (!state.stream) {
        alert('Webcam not initialized');
        return;
    }

    // In Live mode (fast mode) + webcam, use real-time sign detection
    if (typeof isFastMode === 'function' && isFastMode() && state.inputMode === 'webcam') {
        startLiveDemo();
        return;
    }

    // Normal recording mode for Live Detailed and Demo modes
    state.recordedChunks = [];

    const mimeType = MediaRecorder.isTypeSupported('video/webm;codecs=vp9')
        ? 'video/webm;codecs=vp9'
        : 'video/webm';

    state.mediaRecorder = new MediaRecorder(state.stream, { mimeType });

    state.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
            state.recordedChunks.push(event.data);
        }
    };

    state.mediaRecorder.onstop = () => {
        console.log('Recording stopped, chunks:', state.recordedChunks.length);
        state.hasRecording = true;
        updateLiveButtonState();
    };

    state.mediaRecorder.start(100);
    state.isRecording = true;
    state.recordingStartTime = Date.now();

    // Update UI
    updateRecordButton(true);
    liveElements.btnConvert.disabled = true;
    liveElements.recordingIndicator.classList.add('active');
    updateStatus('Recording', 'recording');

    // Start timer
    state.recordingTimer = setInterval(updateRecordingTime, 100);
}

function stopRecording() {
    // If in live demo mode, stop it
    if (state.liveDemoActive) {
        stopLiveDemo();
        return;
    }

    if (!state.mediaRecorder || !state.isRecording) return;

    state.mediaRecorder.stop();
    state.isRecording = false;

    // Update UI
    updateRecordButton(false);
    liveElements.recordingIndicator.classList.remove('active');
    updateStatus('Recorded', 'success');

    clearInterval(state.recordingTimer);
}

function updateRecordingTime() {
    const elapsed = Date.now() - state.recordingStartTime;
    const seconds = Math.floor(elapsed / 1000);
    const minutes = Math.floor(seconds / 60);
    const secs = seconds % 60;
    liveElements.recordingTime.textContent =
        `${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

// Video upload handlers
function handleDragOver(e) {
    e.preventDefault();
    e.currentTarget.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    liveElements.uploadArea.classList.remove('dragover');
}

function handlePoseDragLeave(e) {
    e.preventDefault();
    liveElements.poseUploadArea.classList.remove('dragover');
}

function handleVideoDrop(e) {
    e.preventDefault();
    liveElements.uploadArea.classList.remove('dragover');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleUploadedVideo(files[0]);
    }
}

function handleVideoSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        handleUploadedVideo(files[0]);
    }
}

function handleUploadedVideo(file) {
    if (!file.type.startsWith('video/')) {
        alert('Please select a video file');
        return;
    }

    state.uploadedFile = file;
    state.hasRecording = true;

    // Show preview
    const url = URL.createObjectURL(file);
    liveElements.uploadedVideo.src = url;
    liveElements.uploadedVideo.style.display = 'block';
    liveElements.uploadArea.style.display = 'none';
    liveElements.uploadControls.style.display = 'flex';

    updateStatus('Video Loaded', 'success');
    updateLiveButtonState();
}

function clearUploadedVideo() {
    state.uploadedFile = null;
    state.hasRecording = false;
    liveElements.uploadedVideo.src = '';
    liveElements.uploadedVideo.style.display = 'none';
    liveElements.uploadArea.style.display = 'flex';
    liveElements.uploadControls.style.display = 'none';
    liveElements.uploadInput.value = '';

    updateStatus('Ready', 'success');
    updateLiveButtonState();
}

// Pose file handlers
function handlePoseDrop(e) {
    e.preventDefault();
    liveElements.poseUploadArea.classList.remove('dragover');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleUploadedPose(files[0]);
    }
}

function handlePoseSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        handleUploadedPose(files[0]);
    }
}

function handleUploadedPose(file) {
    if (!file.name.endsWith('.pose')) {
        alert('Please select a .pose file');
        return;
    }

    state.uploadedPoseFile = file;

    // Show file info
    liveElements.poseUploadArea.style.display = 'none';
    liveElements.poseFileInfo.style.display = 'flex';
    liveElements.poseFilename.textContent = file.name;
    liveElements.poseControls.style.display = 'flex';

    updateStatus('Pose File Loaded', 'success');
    updateLiveButtonState();
}

function clearUploadedPose() {
    state.uploadedPoseFile = null;
    liveElements.poseFileInfo.style.display = 'none';
    liveElements.poseUploadArea.style.display = 'flex';
    liveElements.poseControls.style.display = 'none';
    liveElements.poseInput.value = '';

    updateStatus('Ready', 'success');
    updateLiveButtonState();
}

function updateLiveButtonState() {
    const isPoseMode = state.inputMode === 'pose';
    const hasVideoInput = (state.inputMode === 'webcam' && state.recordedChunks.length > 0) ||
                          (state.inputMode === 'upload' && state.uploadedFile);
    const hasPoseInput = state.uploadedPoseFile !== null;

    // Show appropriate button
    liveElements.btnConvert.style.display = isPoseMode ? 'none' : 'inline-block';
    liveElements.btnContinuePose.style.display = isPoseMode ? 'inline-block' : 'none';

    // Enable/disable
    liveElements.btnConvert.disabled = !hasVideoInput;
    liveElements.btnContinuePose.disabled = !hasPoseInput;

    // Update hint text
    if (isPoseMode) {
        liveElements.convertHint.textContent = hasPoseInput ? 'Ready to continue' : 'Upload a .pose file first';
    } else {
        liveElements.convertHint.textContent = hasVideoInput ? 'Ready to convert' : 'Record or upload a video first';
    }
}

async function convertVideo() {
    // Validate reference sentence
    const reference = liveElements.referenceSentence.value.trim();
    if (!reference) {
        alert('Please enter the reference sentence first');
        liveElements.referenceSentence.focus();
        return;
    }

    liveElements.btnConvert.disabled = true;

    // Check if fast mode (Live) - use full pipeline endpoint
    const fastMode = typeof isFastMode === 'function' ? isFastMode() : false;
    console.log('Convert - Mode:', getMode(), 'Fast mode:', fastMode);

    if (fastMode) {
        await processFullPipeline(reference);
    } else {
        await convertVideoDetailed();
    }
}

async function processFullPipeline(reference) {
    showOverlay('Processing full pipeline...');
    liveElements.convertHint.textContent = 'Converting → Segmenting → Predicting → Constructing...';

    try {
        const formData = new FormData();

        if (state.inputMode === 'webcam') {
            const blob = new Blob(state.recordedChunks, { type: 'video/webm' });
            formData.append('video', blob, 'recording.webm');
        } else {
            formData.append('video', state.uploadedFile);
        }

        const response = await fetch('/api/process-full', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.success) {
            // Store all results for evaluate page
            sessionStorage.setItem('rawSentence', result.raw_sentence);
            sessionStorage.setItem('llmSentence', result.llm_sentence);
            sessionStorage.setItem('segments', JSON.stringify(result.predictions));
            sessionStorage.setItem('segmentCount', result.segment_count);

            // Go directly to evaluate page
            window.location.href = '/evaluate';
        } else {
            throw new Error(result.error || 'Pipeline failed');
        }

    } catch (error) {
        console.error('Pipeline error:', error);
        hideOverlay();
        showOverlay('Processing failed: ' + error.message);
        liveElements.btnConvert.disabled = false;
        liveElements.convertHint.textContent = 'Processing failed. Try again.';
    }
}

async function convertVideoDetailed() {
    showOverlay('Converting video to pose keypoints...');
    liveElements.convertHint.textContent = 'Converting...';

    try {
        const formData = new FormData();

        if (state.inputMode === 'webcam') {
            const blob = new Blob(state.recordedChunks, { type: 'video/webm' });
            formData.append('video', blob, 'recording.webm');
        } else {
            formData.append('video', state.uploadedFile);
        }

        // Not fast mode - do visualization
        formData.append('fast_mode', 'false');

        const response = await fetch('/api/convert', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.success) {
            sessionStorage.setItem('poseFile', result.pose_file);
            window.location.href = '/segment';
        } else {
            throw new Error(result.error || 'Conversion failed');
        }

    } catch (error) {
        console.error('Conversion error:', error);
        hideOverlay();
        showOverlay('Conversion failed: ' + error.message);
        liveElements.btnConvert.disabled = false;
        liveElements.convertHint.textContent = 'Conversion failed. Try again.';
    }
}

async function continueToPoseFile() {
    // Validate reference sentence
    const reference = liveElements.referenceSentence.value.trim();
    if (!reference) {
        alert('Please enter the reference sentence first');
        liveElements.referenceSentence.focus();
        return;
    }

    liveElements.btnContinuePose.disabled = true;

    // Check if fast mode (Live) - use full pipeline endpoint
    const fastMode = typeof isFastMode === 'function' ? isFastMode() : false;
    console.log('Pose upload - Mode:', getMode(), 'Fast mode:', fastMode);

    if (fastMode) {
        await processPoseFullPipeline(reference);
    } else {
        await uploadPoseDetailed();
    }
}

async function processPoseFullPipeline(reference) {
    showOverlay('Processing full pipeline...');
    liveElements.convertHint.textContent = 'Segmenting → Predicting → Constructing...';

    try {
        const formData = new FormData();
        formData.append('pose_file', state.uploadedPoseFile);

        const response = await fetch('/api/process-pose-full', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.success) {
            // Store all results for evaluate page
            sessionStorage.setItem('rawSentence', result.raw_sentence);
            sessionStorage.setItem('llmSentence', result.llm_sentence);
            sessionStorage.setItem('segments', JSON.stringify(result.predictions));
            sessionStorage.setItem('segmentCount', result.segment_count);

            // Go directly to evaluate page
            window.location.href = '/evaluate';
        } else {
            throw new Error(result.error || 'Pipeline failed');
        }

    } catch (error) {
        console.error('Pipeline error:', error);
        hideOverlay();
        showOverlay('Processing failed: ' + error.message);
        liveElements.btnContinuePose.disabled = false;
        liveElements.convertHint.textContent = 'Processing failed. Try again.';
    }
}

async function uploadPoseDetailed() {
    showOverlay('Uploading pose file...');

    try {
        const formData = new FormData();
        formData.append('pose_file', state.uploadedPoseFile);

        const response = await fetch('/api/upload-pose', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (result.success) {
            sessionStorage.setItem('poseFile', result.pose_file);
            window.location.href = '/segment';
        } else {
            throw new Error(result.error || 'Upload failed');
        }

    } catch (error) {
        console.error('Pose upload error:', error);
        hideOverlay();
        alert('Failed to upload pose file: ' + error.message);
        liveElements.btnContinuePose.disabled = false;
    }
}

// UI Helpers
function updateStatus(text, type) {
    liveElements.inputStatus.textContent = text;
    liveElements.inputStatus.className = 'status-badge status-' + type;
}

function showOverlay(message) {
    liveElements.overlayText.textContent = message;
    liveElements.webcamOverlay.classList.add('visible');
}

function hideOverlay() {
    liveElements.webcamOverlay.classList.remove('visible');
}

// ============================================================================
// DEMO MODE
// ============================================================================

async function setupDemoMode() {
    // Show/hide appropriate content
    liveElements.container.style.display = 'none';
    demoElements.container.style.display = 'block';

    // Stop webcam if running
    stopWebcam();

    // Load samples
    await loadSamples();

    // Button listener
    demoElements.btnContinue.addEventListener('click', continueWithSample);
}

async function loadSamples() {
    try {
        const response = await fetch('/api/samples');
        const data = await response.json();

        state.samples = data.samples || [];

        if (state.samples.length === 0) {
            demoElements.samplesGrid.innerHTML = `
                <div class="no-samples">
                    <p>No demo samples available.</p>
                    <p class="hint">Use prepare_demo_sample.py to create samples.</p>
                </div>
            `;
            return;
        }

        renderSamples();

    } catch (error) {
        console.error('Failed to load samples:', error);
        demoElements.samplesGrid.innerHTML = `
            <div class="sample-error">
                <p>Failed to load demo samples.</p>
                <p class="hint">${error.message}</p>
            </div>
        `;
    }
}

function renderSamples() {
    demoElements.samplesGrid.innerHTML = '';

    state.samples.forEach(sample => {
        const card = document.createElement('div');
        card.className = 'sample-card';
        card.dataset.sampleId = sample.id;

        // Use original video if available, otherwise fall back to pose video
        const videoFile = sample.has_original_video && sample.original_video
            ? sample.original_video
            : 'pose_video.mp4';
        const isOriginalVideo = sample.has_original_video && sample.original_video;

        card.innerHTML = `
            <div class="sample-thumbnail ${isOriginalVideo ? 'has-original' : 'pose-only'}">
                <video src="/demo-data/samples/${sample.id}/${videoFile}" muted loop playsinline></video>
                ${isOriginalVideo ? '<span class="video-badge">Video</span>' : '<span class="video-badge pose-badge">Pose</span>'}
            </div>
            <div class="sample-info">
                <h4 class="sample-name">${sample.name}</h4>
            </div>
            <div class="sample-selected-indicator">&#10003;</div>
        `;

        // Hover to play video
        const video = card.querySelector('video');
        card.addEventListener('mouseenter', () => video.play());
        card.addEventListener('mouseleave', () => {
            video.pause();
            video.currentTime = 0;
        });

        // Click to select
        card.addEventListener('click', () => selectSample(sample.id));

        demoElements.samplesGrid.appendChild(card);
    });
}

async function selectSample(sampleId) {
    // Update UI selection
    document.querySelectorAll('.sample-card').forEach(card => {
        card.classList.toggle('selected', card.dataset.sampleId === sampleId);
    });

    // Load sample metadata
    try {
        const response = await fetch(`/api/samples/${sampleId}`);
        const sample = await response.json();

        state.selectedSample = sample;

        // Update reference sentence
        demoElements.referenceSentence.value = sample.reference_sentence;
        sessionStorage.setItem('referenceSentence', sample.reference_sentence);

        // Enable continue button
        demoElements.btnContinue.disabled = false;
        demoElements.hint.textContent = `Selected: ${sample.name}`;

    } catch (error) {
        console.error('Failed to load sample:', error);
        alert('Failed to load sample details');
    }
}

function continueWithSample() {
    if (!state.selectedSample) {
        alert('Please select a sample first');
        return;
    }

    // Store sample data in session
    setSelectedSample(state.selectedSample);
    sessionStorage.setItem('poseFile', `demo:${state.selectedSample.id}`);

    // Navigate to segment page
    window.location.href = '/segment';
}

// ============================================================================
// LIVE DEMO MODE
// Real-time sign detection and gloss prediction (no LLM)
// ============================================================================

async function loadLiveConfig() {
    try {
        const response = await fetch('/api/live-config');
        state.liveConfig = await response.json();
        console.log('Live config loaded:', state.liveConfig);
    } catch (error) {
        console.error('Failed to load live config:', error);
        // Use defaults
        state.liveConfig = {
            segmentation_type: 'motion',
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

async function startLiveDemo() {
    if (!state.stream) {
        alert('Please wait for webcam to initialize');
        return;
    }

    console.log('Starting live demo mode (Live mode + Webcam)...');

    // Reset live demo state on backend (clears previous segments)
    try {
        await fetch('/api/live-reset', { method: 'POST' });
    } catch (e) {
        console.log('Live reset failed, continuing anyway');
    }

    state.liveDemoActive = true;
    state.detectedGlosses = [];
    state.isSigning = false;
    state.isProcessing = false;

    // Initialize motion detection canvas
    initMotionDetection();

    // Update UI - use toggle button
    document.querySelector('.live-mode-content').classList.add('live-demo-active');
    updateRecordButton(true);
    liveElements.btnConvert.style.display = 'none';
    liveElements.convertHint.textContent = 'Real-time sign detection active';

    updateStatus('Live Demo', 'recording');
    updateLiveStatus('READY', 'ready');
    liveElements.currentPrediction.textContent = '';
    liveElements.predictionConfidence.textContent = 'Waiting for signing...';
    liveElements.allPredictions.textContent = '';

    // Start motion detection loop
    requestAnimationFrame(motionDetectionLoop);
}

async function stopLiveDemo() {
    console.log('Stopping live demo mode...');
    state.liveDemoActive = false;

    // Cancel any in-progress sign recording
    cancelSignRecording();

    // Check if we have detected glosses to process
    if (state.detectedGlosses.length > 0) {
        // Process detected glosses through LLM
        await processDetectedGlossesWithLLM();
        return;
    }

    // No glosses detected - restore normal UI
    restoreLiveDemoUI();
}

function restoreLiveDemoUI() {
    // Update UI - restore normal button states
    document.querySelector('.live-mode-content').classList.remove('live-demo-active');
    updateRecordButton(false);
    liveElements.btnConvert.style.display = 'inline-block';

    updateLiveButtonState();
    updateStatus('Ready', 'success');
    liveElements.convertHint.textContent = 'Record a video first';
}

async function processDetectedGlossesWithLLM() {
    console.log('Processing detected glosses with LLM...', state.detectedGlosses);

    // Show processing overlay
    showOverlay('Constructing sentence with LLM...');
    liveElements.convertHint.textContent = 'Processing Top-3 predictions with LLM...';
    updateStatus('Processing', 'pending');

    try {
        // Convert detected glosses to the format expected by /api/construct
        // The API expects: { predictions: [{ top_1: string, top_k: [...] }] }
        const predictions = state.detectedGlosses.map(g => ({
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

            console.log('LLM processing complete:', {
                raw: result.raw_sentence,
                llm: result.llm_sentence
            });

            // Navigate to evaluate page
            window.location.href = '/evaluate';
        } else {
            throw new Error(result.error || 'LLM construction failed');
        }

    } catch (error) {
        console.error('LLM processing error:', error);
        hideOverlay();
        alert('Failed to construct sentence: ' + error.message);

        // Restore UI on error
        restoreLiveDemoUI();
    }
}

function initMotionDetection() {
    // Create offscreen canvas for motion detection
    state.motionDetectionCanvas = document.createElement('canvas');
    state.motionDetectionCanvas.width = 160;  // Downscale for performance
    state.motionDetectionCanvas.height = 120;
    state.motionDetectionCtx = state.motionDetectionCanvas.getContext('2d', { willReadFrequently: true });
    state.previousFrame = null;
}

function motionDetectionLoop() {
    if (!state.liveDemoActive) return;

    const video = liveElements.webcamVideo;
    if (!video || video.readyState < 2) {
        requestAnimationFrame(motionDetectionLoop);
        return;
    }

    // Draw current frame to detection canvas
    const ctx = state.motionDetectionCtx;
    const canvas = state.motionDetectionCanvas;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const currentFrame = ctx.getImageData(0, 0, canvas.width, canvas.height);
    let motionScore = 0;

    if (state.previousFrame) {
        // Calculate motion score
        motionScore = calculateMotionScore(state.previousFrame.data, currentFrame.data);
    }

    state.previousFrame = currentFrame;

    // Update motion score display
    if (liveElements.motionScoreValue) {
        liveElements.motionScoreValue.textContent = motionScore.toFixed(0);
    }

    // Motion detection state machine
    const config = state.liveConfig?.motion_config || {};
    const motionThreshold = config.motion_area_threshold || 0.02;
    const isMotionDetected = motionScore > motionThreshold * 10000;

    const now = Date.now();

    if (isMotionDetected) {
        state.lastMotionTime = now;

        if (!state.isSigning && !state.isProcessing) {
            // Start signing
            state.isSigning = true;
            state.signStartTime = now;
            state.signFrames = [];
            updateLiveStatus('SIGNING', 'signing');
            console.log('Sign started');

            // Start recording the sign
            startSignRecording();
        }
    } else if (state.isSigning) {
        const cooldownMs = config.cooldown_ms || 1000;
        const timeSinceMotion = now - state.lastMotionTime;

        if (timeSinceMotion > cooldownMs) {
            // Sign ended
            const signDuration = now - state.signStartTime;
            const minSignMs = config.min_sign_ms || 500;

            if (signDuration >= minSignMs) {
                console.log(`Sign ended after ${signDuration}ms`);
                stopSignRecordingAndProcess();
            } else {
                console.log(`Sign too short (${signDuration}ms), ignoring`);
                cancelSignRecording();
            }
            state.isSigning = false;
        }
    }

    // Force end if sign too long
    if (state.isSigning) {
        const maxSignMs = config.max_sign_ms || 5000;
        const signDuration = now - state.signStartTime;

        if (signDuration >= maxSignMs) {
            console.log('Sign too long, forcing end');
            stopSignRecordingAndProcess();
            state.isSigning = false;
        }
    }

    requestAnimationFrame(motionDetectionLoop);
}

function calculateMotionScore(prev, curr) {
    let diff = 0;
    // Compare pixels (grayscale approximation)
    for (let i = 0; i < prev.length; i += 4) {
        const prevGray = (prev[i] + prev[i + 1] + prev[i + 2]) / 3;
        const currGray = (curr[i] + curr[i + 1] + curr[i + 2]) / 3;
        const pixelDiff = Math.abs(prevGray - currGray);

        if (pixelDiff > (state.liveConfig?.motion_config?.motion_threshold || 30)) {
            diff++;
        }
    }
    return diff;
}

let signMediaRecorder = null;
let signChunks = [];

function startSignRecording() {
    if (!state.stream) return;

    signChunks = [];

    const mimeType = MediaRecorder.isTypeSupported('video/webm;codecs=vp9')
        ? 'video/webm;codecs=vp9'
        : 'video/webm';

    signMediaRecorder = new MediaRecorder(state.stream, { mimeType });

    signMediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
            signChunks.push(event.data);
        }
    };

    signMediaRecorder.start(100);
}

function cancelSignRecording() {
    if (signMediaRecorder && signMediaRecorder.state !== 'inactive') {
        signMediaRecorder.stop();
    }
    signChunks = [];
    updateLiveStatus('READY', 'ready');
}

async function stopSignRecordingAndProcess() {
    if (!signMediaRecorder || signMediaRecorder.state === 'inactive') {
        updateLiveStatus('READY', 'ready');
        return;
    }

    state.isProcessing = true;
    updateLiveStatus('PROCESSING', 'processing');
    liveElements.currentPrediction.textContent = 'Processing';
    liveElements.currentPrediction.classList.add('processing');

    // Stop recording and wait for data
    return new Promise((resolve) => {
        signMediaRecorder.onstop = async () => {
            if (signChunks.length === 0) {
                state.isProcessing = false;
                updateLiveStatus('READY', 'ready');
                liveElements.currentPrediction.classList.remove('processing');
                resolve();
                return;
            }

            const blob = new Blob(signChunks, { type: 'video/webm' });
            signChunks = [];

            try {
                const result = await sendSignForPrediction(blob);

                if (result.success) {
                    displayPrediction(result);
                } else {
                    console.error('Prediction failed:', result.error);
                    liveElements.currentPrediction.textContent = 'Error';
                    liveElements.currentPrediction.classList.remove('processing');
                }
            } catch (error) {
                console.error('Failed to process sign:', error);
                liveElements.currentPrediction.textContent = 'Error';
                liveElements.currentPrediction.classList.remove('processing');
            }

            state.isProcessing = false;
            updateLiveStatus('READY', 'ready');
            resolve();
        };

        signMediaRecorder.stop();
    });
}

async function sendSignForPrediction(videoBlob) {
    const formData = new FormData();
    formData.append('video', videoBlob, 'sign.webm');

    const response = await fetch('/api/process-sign', {
        method: 'POST',
        body: formData
    });

    return await response.json();
}

function displayPrediction(result) {
    liveElements.currentPrediction.classList.remove('processing');

    // Add to detected glosses
    state.detectedGlosses.push({
        gloss: result.gloss,
        confidence: result.confidence,
        top_k: result.top_k
    });

    // Show current prediction with animation
    liveElements.currentPrediction.textContent = result.gloss;
    liveElements.currentPrediction.classList.add('new');
    setTimeout(() => liveElements.currentPrediction.classList.remove('new'), 300);

    // Show confidence
    const confPercent = (result.confidence * 100).toFixed(1);
    liveElements.predictionConfidence.textContent = `Confidence: ${confPercent}%`;

    // Show top-k alternatives if available
    if (result.top_k && result.top_k.length > 1) {
        const alts = result.top_k.slice(1).map(p =>
            `${p.gloss} (${(p.confidence * 100).toFixed(0)}%)`
        ).join(', ');
        liveElements.predictionConfidence.textContent += ` | Alt: ${alts}`;
    }

    // Show all detected glosses
    const allGlossesHtml = state.detectedGlosses.map(g =>
        `<span class="gloss">${g.gloss}</span>`
    ).join('');
    liveElements.allPredictions.innerHTML = allGlossesHtml;

    console.log('Prediction:', result.gloss, `(${confPercent}%)`);
}

function updateLiveStatus(text, statusClass) {
    if (liveElements.liveStatusText) {
        liveElements.liveStatusText.textContent = text;
    }
    if (liveElements.liveStatusDot) {
        liveElements.liveStatusDot.className = 'status-dot';
        if (statusClass) {
            liveElements.liveStatusDot.classList.add(statusClass);
        }
    }
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (state.liveDemoActive) {
        stopLiveDemo();
    }
    stopWebcam();
});
