/**
 * Live Mode Learn - Tutorial Video Playback
 */

// State
let glosses = [];
let tutorials = [];
let currentIndex = 0;
let reference = '';

// DOM Elements
let referenceDisplay;
let progressFill, currentIndexSpan, totalSignsSpan;
let currentGlossEl, tutorialVideo, videoLoading, videoUnavailable;
let replayBtn, slowBtn, normalBtn;
let queueItems;
let prevBtn, nextBtn, skipBtn;

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
        // No data, go back to setup
        window.location.href = '/live-setup';
        return;
    }

    glosses = JSON.parse(glossesJson);

    initializeElements();
    await loadTutorials();
    setupEventListeners();
    renderQueue();
    showCurrentSign();
});

function initializeElements() {
    referenceDisplay = document.getElementById('reference-sentence-display');
    progressFill = document.getElementById('progress-fill');
    currentIndexSpan = document.getElementById('current-index');
    totalSignsSpan = document.getElementById('total-signs');
    currentGlossEl = document.getElementById('current-gloss');
    tutorialVideo = document.getElementById('tutorial-video');
    videoLoading = document.getElementById('video-loading');
    videoUnavailable = document.getElementById('video-unavailable');
    replayBtn = document.getElementById('replay-btn');
    slowBtn = document.getElementById('slow-btn');
    normalBtn = document.getElementById('normal-btn');
    queueItems = document.getElementById('queue-items');
    prevBtn = document.getElementById('prev-btn');
    nextBtn = document.getElementById('next-btn');
    skipBtn = document.getElementById('skip-btn');

    // Display reference sentence
    referenceDisplay.textContent = reference;
    totalSignsSpan.textContent = glosses.length;
}

async function loadTutorials() {
    try {
        const response = await fetch('/api/gloss-tutorials', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ glosses: glosses })
        });

        const data = await response.json();
        tutorials = data.tutorials || [];

    } catch (error) {
        console.error('Error loading tutorials:', error);
        tutorials = glosses.map(g => ({ gloss: g, available: false, reason: 'Load error' }));
    }
}

function setupEventListeners() {
    prevBtn.addEventListener('click', showPreviousSign);
    nextBtn.addEventListener('click', showNextSign);
    skipBtn.addEventListener('click', skipToRecord);

    replayBtn.addEventListener('click', () => {
        tutorialVideo.currentTime = 0;
        tutorialVideo.play();
    });

    slowBtn.addEventListener('click', () => {
        tutorialVideo.playbackRate = 0.5;
        updateSpeedButtons(0.5);
    });

    normalBtn.addEventListener('click', () => {
        tutorialVideo.playbackRate = 1.0;
        updateSpeedButtons(1.0);
    });

    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowLeft' && currentIndex > 0) {
            showPreviousSign();
        } else if (e.key === 'ArrowRight' && currentIndex < glosses.length - 1) {
            showNextSign();
        } else if (e.key === 'Enter' && currentIndex === glosses.length - 1) {
            skipToRecord();
        }
    });
}

function updateSpeedButtons(rate) {
    slowBtn.classList.toggle('active', rate === 0.5);
    normalBtn.classList.toggle('active', rate === 1.0);
}

function renderQueue() {
    queueItems.innerHTML = glosses.map((gloss, idx) => `
        <div class="queue-item ${idx === currentIndex ? 'active' : ''} ${idx < currentIndex ? 'completed' : ''}"
             data-index="${idx}">
            <span class="queue-number">${idx + 1}</span>
            <span class="queue-gloss">${gloss}</span>
        </div>
    `).join('');

    // Add click handlers
    document.querySelectorAll('.queue-item').forEach(item => {
        item.addEventListener('click', () => {
            const idx = parseInt(item.dataset.index);
            currentIndex = idx;
            showCurrentSign();
            renderQueue();
        });
    });
}

async function showCurrentSign() {
    const gloss = glosses[currentIndex];
    const tutorial = tutorials[currentIndex];

    // Update UI
    currentGlossEl.textContent = gloss.toUpperCase();
    currentIndexSpan.textContent = currentIndex + 1;
    updateProgress();
    updateNavButtons();

    // Show loading state
    videoLoading.style.display = 'flex';
    videoUnavailable.style.display = 'none';
    tutorialVideo.style.display = 'none';

    if (!tutorial || !tutorial.available) {
        // No tutorial available
        videoLoading.style.display = 'none';
        videoUnavailable.style.display = 'flex';
        return;
    }

    if (tutorial.needs_visualization) {
        // Need to generate visualization on-demand
        try {
            const response = await fetch('/api/visualize-gloss', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    gloss: tutorial.gloss,
                    pose_file: tutorial.pose_file
                })
            });

            const data = await response.json();

            if (data.success) {
                tutorial.video_url = data.video_url;
                tutorial.needs_visualization = false;
            } else {
                throw new Error(data.error || 'Visualization failed');
            }
        } catch (error) {
            console.error('Error visualizing gloss:', error);
            videoLoading.style.display = 'none';
            videoUnavailable.style.display = 'flex';
            return;
        }
    }

    // Load video
    tutorialVideo.src = tutorial.video_url;
    tutorialVideo.playbackRate = 1.0;
    updateSpeedButtons(1.0);

    tutorialVideo.onloadeddata = () => {
        videoLoading.style.display = 'none';
        tutorialVideo.style.display = 'block';
        tutorialVideo.play();
    };

    tutorialVideo.onerror = () => {
        videoLoading.style.display = 'none';
        videoUnavailable.style.display = 'flex';
    };
}

function updateProgress() {
    const progress = ((currentIndex + 1) / glosses.length) * 100;
    progressFill.style.width = `${progress}%`;
}

function updateNavButtons() {
    prevBtn.disabled = currentIndex === 0;

    if (currentIndex === glosses.length - 1) {
        nextBtn.innerHTML = 'Continue to Record <span class="arrow">&#8594;</span>';
        nextBtn.onclick = skipToRecord;
    } else {
        nextBtn.innerHTML = 'Next Sign <span class="arrow">&#8594;</span>';
        nextBtn.onclick = showNextSign;
    }
}

function showPreviousSign() {
    if (currentIndex > 0) {
        currentIndex--;
        showCurrentSign();
        renderQueue();
    }
}

function showNextSign() {
    if (currentIndex < glosses.length - 1) {
        currentIndex++;
        showCurrentSign();
        renderQueue();
    }
}

function skipToRecord() {
    // Navigate to record page
    window.location.href = '/live-record';
}
