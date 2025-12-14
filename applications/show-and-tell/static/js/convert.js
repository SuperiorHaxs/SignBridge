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
    selectedSample: null
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
    btnStart: document.getElementById('btn-start'),
    btnStop: document.getElementById('btn-stop'),
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
    inputStatus: document.getElementById('input-status')
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

    // Webcam controls
    liveElements.btnStart.addEventListener('click', startRecording);
    liveElements.btnStop.addEventListener('click', stopRecording);

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

    // Initialize webcam by default
    await initializeWebcam();
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
        liveElements.btnStart.disabled = false;

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

function startRecording() {
    if (!state.stream) {
        alert('Webcam not initialized');
        return;
    }

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
    liveElements.btnStart.disabled = true;
    liveElements.btnStop.disabled = false;
    liveElements.btnConvert.disabled = true;
    liveElements.recordingIndicator.classList.add('active');
    updateStatus('Recording', 'recording');

    // Start timer
    state.recordingTimer = setInterval(updateRecordingTime, 100);
}

function stopRecording() {
    if (!state.mediaRecorder || !state.isRecording) return;

    state.mediaRecorder.stop();
    state.isRecording = false;

    // Update UI
    liveElements.btnStart.disabled = false;
    liveElements.btnStop.disabled = true;
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

    showOverlay('Converting video to pose keypoints...');
    liveElements.btnConvert.disabled = true;
    liveElements.convertHint.textContent = 'Converting...';

    try {
        const formData = new FormData();

        if (state.inputMode === 'webcam') {
            const blob = new Blob(state.recordedChunks, { type: 'video/webm' });
            formData.append('video', blob, 'recording.webm');
        } else {
            formData.append('video', state.uploadedFile);
        }

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

    showOverlay('Uploading pose file...');
    liveElements.btnContinuePose.disabled = true;

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

        card.innerHTML = `
            <div class="sample-thumbnail">
                <video src="/demo-data/samples/${sample.id}/pose_video.mp4" muted loop></video>
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

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    stopWebcam();
});
