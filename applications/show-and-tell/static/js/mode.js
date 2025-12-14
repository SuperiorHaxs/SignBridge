/**
 * Global Mode Management
 * Handles Live/Demo mode toggle across all phases
 */

const AppMode = {
    LIVE: 'live',
    DEMO: 'demo'
};

// Get current mode from sessionStorage
function getMode() {
    return sessionStorage.getItem('appMode') || AppMode.LIVE;
}

// Set mode and update UI
function setMode(mode) {
    const currentMode = getMode();

    if (mode !== currentMode) {
        // Confirm mode switch if there's progress
        const hasProgress = sessionStorage.getItem('poseFile') ||
                           sessionStorage.getItem('segments') ||
                           sessionStorage.getItem('referenceSentence');

        if (hasProgress) {
            const confirmSwitch = confirm(
                `Switch to ${mode === AppMode.DEMO ? 'Demo' : 'Live'} mode?\n\nThis will reset your current progress.`
            );
            if (!confirmSwitch) {
                // Revert toggle
                updateToggleUI(currentMode);
                return;
            }
        }

        // Clear session data
        sessionStorage.clear();
        sessionStorage.setItem('appMode', mode);

        // Redirect to Phase 1
        window.location.href = '/';
    }
}

// Update toggle UI to match mode
function updateToggleUI(mode) {
    const toggle = document.getElementById('mode-switch');
    if (toggle) {
        toggle.checked = (mode === AppMode.DEMO);
    }

    // Update body class for CSS styling
    document.body.classList.remove('mode-live', 'mode-demo');
    document.body.classList.add(`mode-${mode}`);

    // Update Play All button visibility
    const btnPlayAll = document.getElementById('btn-play-all');
    if (btnPlayAll) {
        btnPlayAll.style.display = (mode === AppMode.DEMO) ? 'flex' : 'none';
    }
}

// Initialize mode on page load
function initializeMode() {
    const mode = getMode();
    updateToggleUI(mode);

    // Set up toggle listener
    const toggle = document.getElementById('mode-switch');
    if (toggle) {
        toggle.addEventListener('change', (e) => {
            const newMode = e.target.checked ? AppMode.DEMO : AppMode.LIVE;
            setMode(newMode);
        });
    }

    // Dispatch event for page-specific handlers
    window.dispatchEvent(new CustomEvent('modeInitialized', { detail: { mode } }));
}

// Check if in demo mode
function isDemoMode() {
    return getMode() === AppMode.DEMO;
}

// Check if in live mode
function isLiveMode() {
    return getMode() === AppMode.LIVE;
}

// Get selected demo sample
function getSelectedSample() {
    const sampleJson = sessionStorage.getItem('demoSample');
    return sampleJson ? JSON.parse(sampleJson) : null;
}

// Set selected demo sample
function setSelectedSample(sample) {
    sessionStorage.setItem('demoSample', JSON.stringify(sample));
}

// Play All functionality for demo mode
let playAllActive = false;
let playAllAborted = false;

function initializePlayAll() {
    const btnPlayAll = document.getElementById('btn-play-all');
    if (!btnPlayAll) return;

    // Show Play All button only in demo mode
    updatePlayAllVisibility();

    btnPlayAll.addEventListener('click', togglePlayAll);
}

function updatePlayAllVisibility() {
    const btnPlayAll = document.getElementById('btn-play-all');
    if (!btnPlayAll) return;

    if (isDemoMode()) {
        btnPlayAll.style.display = 'flex';
    } else {
        btnPlayAll.style.display = 'none';
    }
}

function togglePlayAll() {
    const btnPlayAll = document.getElementById('btn-play-all');
    if (!btnPlayAll) return;

    if (playAllActive) {
        // Stop auto-play
        playAllAborted = true;
        playAllActive = false;
        btnPlayAll.classList.remove('playing');
        btnPlayAll.innerHTML = '&#9654; Play All';
    } else {
        // Start auto-play
        playAllActive = true;
        playAllAborted = false;
        btnPlayAll.classList.add('playing');
        btnPlayAll.innerHTML = '&#9632; Stop';
        startAutoPlay();
    }
}

async function startAutoPlay() {
    // Store flag indicating auto-play is active
    sessionStorage.setItem('autoPlayActive', 'true');

    // Get current phase from URL
    const path = window.location.pathname;

    // Determine what action to take based on current page
    await autoPlayCurrentPhase(path);
}

async function autoPlayCurrentPhase(path) {
    if (playAllAborted) {
        sessionStorage.removeItem('autoPlayActive');
        return;
    }

    // Small delay before auto-clicking
    await delay(500);

    if (path === '/' || path === '/convert') {
        // Phase 1: Click "Continue with Sample" if a sample is selected
        const btnContinue = document.getElementById('btn-continue-sample');
        if (btnContinue && !btnContinue.disabled) {
            btnContinue.click();
        }
    } else if (path === '/segment') {
        // Phase 2: Click "Segment Signs"
        const btnSegment = document.getElementById('btn-segment');
        if (btnSegment && btnSegment.style.display !== 'none') {
            btnSegment.click();
            // Wait for segmentation, then click Next
            await waitForElement('#btn-next', 5000);
            if (!playAllAborted) {
                await delay(1000);
                const btnNext = document.getElementById('btn-next');
                if (btnNext) btnNext.click();
            }
        } else {
            // Already segmented, click Next
            const btnNext = document.getElementById('btn-next');
            if (btnNext) btnNext.click();
        }
    } else if (path === '/predict') {
        // Phase 3: Wait for predictions to display, then click Next
        await delay(1000);
        const btnNext = document.querySelector('.action-bar .btn-success');
        if (btnNext && !playAllAborted) btnNext.click();
    } else if (path === '/construct') {
        // Phase 4: Click "Construct Sentence"
        const btnConstruct = document.getElementById('btn-construct');
        if (btnConstruct && btnConstruct.style.display !== 'none') {
            btnConstruct.click();
            // Wait for construction, then click Next
            await waitForElement('.construct-results .btn-success', 5000);
            if (!playAllAborted) {
                await delay(1000);
                const btnNext = document.querySelector('.construct-results .btn-success');
                if (btnNext) btnNext.click();
            }
        }
    } else if (path === '/evaluate') {
        // Phase 5: Click "Calculate Metrics"
        const btnEvaluate = document.getElementById('btn-evaluate');
        if (btnEvaluate) {
            btnEvaluate.click();
            // Auto-play complete after metrics are shown
            await waitForElement('#final-actions', 5000);
            stopAutoPlay();
        }
    }
}

function stopAutoPlay() {
    playAllActive = false;
    playAllAborted = false;
    sessionStorage.removeItem('autoPlayActive');

    const btnPlayAll = document.getElementById('btn-play-all');
    if (btnPlayAll) {
        btnPlayAll.classList.remove('playing');
        btnPlayAll.innerHTML = '&#9654; Play All';
    }
}

// Helper function for delays
function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// Helper function to wait for an element to appear
function waitForElement(selector, timeout = 5000) {
    return new Promise((resolve) => {
        const element = document.querySelector(selector);
        if (element) {
            resolve(element);
            return;
        }

        const observer = new MutationObserver(() => {
            const element = document.querySelector(selector);
            if (element) {
                observer.disconnect();
                resolve(element);
            }
        });

        observer.observe(document.body, { childList: true, subtree: true });

        setTimeout(() => {
            observer.disconnect();
            resolve(null);
        }, timeout);
    });
}

// Check if auto-play should continue after page navigation
function checkAutoPlayContinuation() {
    if (sessionStorage.getItem('autoPlayActive') === 'true' && isDemoMode()) {
        playAllActive = true;

        const btnPlayAll = document.getElementById('btn-play-all');
        if (btnPlayAll) {
            btnPlayAll.classList.add('playing');
            btnPlayAll.innerHTML = '&#9632; Stop';
        }

        // Continue auto-play after a short delay for page initialization
        setTimeout(() => {
            autoPlayCurrentPhase(window.location.pathname);
        }, 800);
    }
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    initializeMode();
    initializePlayAll();
    checkAutoPlayContinuation();
});

// Export for use in other scripts
window.AppMode = AppMode;
window.getMode = getMode;
window.setMode = setMode;
window.isDemoMode = isDemoMode;
window.isLiveMode = isLiveMode;
window.getSelectedSample = getSelectedSample;
window.setSelectedSample = setSelectedSample;
window.stopAutoPlay = stopAutoPlay;
