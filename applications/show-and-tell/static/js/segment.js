/**
 * Phase 2: Segment Pose
 * Handles pose segmentation into individual signs
 * Supports both Live and Demo modes
 */

const state = {
    poseFile: null,
    segments: [],
    demoSample: null
};

const elements = {
    fullPoseVideo: document.getElementById('full-pose-video'),
    btnSegment: document.getElementById('btn-segment'),
    loadingState: document.getElementById('loading-state'),
    errorState: document.getElementById('error-state'),
    errorMessage: document.getElementById('error-message'),
    btnRetry: document.getElementById('btn-retry'),
    segmentsResult: document.getElementById('segments-result'),
    segmentsSummary: document.getElementById('segments-summary'),
    segmentsTimeline: document.getElementById('segments-timeline'),
    btnNext: document.getElementById('btn-next')
};

document.addEventListener('DOMContentLoaded', initialize);

function initialize() {
    // Get pose file from session storage
    state.poseFile = sessionStorage.getItem('poseFile');

    if (!state.poseFile) {
        showError('No pose file found. Please go back and convert a video first.');
        return;
    }

    // Check if demo mode
    if (state.poseFile.startsWith('demo:')) {
        initializeDemoMode();
    } else {
        initializeLiveMode();
    }

    // Button listeners
    elements.btnSegment.addEventListener('click', segmentPose);
    elements.btnRetry.addEventListener('click', segmentPose);
}

function initializeLiveMode() {
    // Load the pose video (add cache buster)
    const sessionId = state.poseFile.split(/[\\\/]/).slice(-2, -1)[0];
    elements.fullPoseVideo.src = `/temp/${sessionId}/pose_video.mp4?t=${Date.now()}`;
}

function initializeDemoMode() {
    // Get demo sample data
    state.demoSample = getSelectedSample();

    if (!state.demoSample) {
        showError('Demo sample data not found. Please go back and select a sample.');
        return;
    }

    // Load pre-computed pose video
    const sampleId = state.poseFile.replace('demo:', '');
    elements.fullPoseVideo.src = `/demo-data/samples/${sampleId}/pose_video.mp4?t=${Date.now()}`;
}

async function segmentPose() {
    // Check if demo mode with pre-computed segments
    if (state.poseFile.startsWith('demo:') && state.demoSample?.precomputed?.segments) {
        // Use pre-computed segments
        useDemoSegments();
        return;
    }

    // Live mode - run actual segmentation
    showLoading();

    try {
        const response = await fetch('/api/segment', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ pose_file: state.poseFile, fast_mode: isFastMode() })
        });

        const result = await response.json();

        if (result.success) {
            state.segments = result.segments;
            sessionStorage.setItem('segments', JSON.stringify(result.segments));
            displaySegments(result.segments);
        } else {
            throw new Error(result.error || 'Segmentation failed');
        }

    } catch (error) {
        console.error('Segmentation error:', error);
        showError(error.message);
    }
}

function useDemoSegments() {
    showLoading();

    // Small delay for visual feedback
    setTimeout(() => {
        const sampleId = state.poseFile.replace('demo:', '');
        const precomputedSegments = state.demoSample.precomputed.segments;

        // Transform segment data to include correct URLs
        const segments = precomputedSegments.map(seg => ({
            ...seg,
            video_url: `/demo-data/samples/${sampleId}/${seg.video_file}`
        }));

        state.segments = segments;
        sessionStorage.setItem('segments', JSON.stringify(segments));
        displaySegments(segments);
    }, 300);
}

function displaySegments(segments) {
    elements.segmentsSummary.textContent = `Detected ${segments.length} sign segment${segments.length !== 1 ? 's' : ''}`;

    elements.segmentsTimeline.innerHTML = '';

    const cacheBuster = Date.now();
    segments.forEach((segment, index) => {
        const item = document.createElement('div');
        item.className = 'segment-timeline-item';

        // Handle fast mode (no video preview)
        let previewContent;
        if (segment.video_url) {
            previewContent = `<video src="${segment.video_url}?t=${cacheBuster}" autoplay loop muted playsinline></video>`;
        } else {
            // Fast mode - show placeholder with segment number
            previewContent = `<div class="segment-placeholder">
                <span class="placeholder-icon">⚡</span>
                <span class="placeholder-text">Fast Mode</span>
            </div>`;
        }

        item.innerHTML = `
            <div class="segment-number-badge">${segment.segment_id}</div>
            <div class="segment-preview">
                ${previewContent}
            </div>
            <div class="segment-info-mini">
                <span class="segment-frames">Segment ${segment.segment_id}</span>
            </div>
        `;
        elements.segmentsTimeline.appendChild(item);
    });

    showResults();
}

function showLoading() {
    elements.btnSegment.style.display = 'none';
    elements.loadingState.style.display = 'flex';
    elements.errorState.style.display = 'none';
    elements.segmentsResult.style.display = 'none';
}

function showError(message) {
    elements.btnSegment.style.display = 'none';
    elements.loadingState.style.display = 'none';
    elements.errorState.style.display = 'flex';
    elements.errorMessage.textContent = message;
    elements.segmentsResult.style.display = 'none';
}

function showResults() {
    elements.btnSegment.style.display = 'none';
    elements.loadingState.style.display = 'none';
    elements.errorState.style.display = 'none';
    elements.segmentsResult.style.display = 'block';
}
