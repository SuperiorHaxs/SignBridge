/**
 * Global Mode Management
 * Handles Demo/Live/Live(Detailed) mode toggle across all phases
 */

const AppMode = {
    DEMO: 'demo',
    LIVE: 'live',
    LIVE_DETAILED: 'live_detailed'
};

// Mode display names
const ModeNames = {
    [AppMode.DEMO]: 'Preset Demo',
    [AppMode.LIVE]: 'Live',
    [AppMode.LIVE_DETAILED]: 'Upload'
};

// Get current mode from sessionStorage
function getMode() {
    return sessionStorage.getItem('appMode') || AppMode.DEMO;
}

// Check if fast mode (no visualizations) - Live mode is fast, Live Detailed is not
function isFastMode() {
    return getMode() === AppMode.LIVE;
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
                `Switch to ${ModeNames[mode]} mode?\n\nThis will reset your current progress.`
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

        // Redirect to Phase 1 (different start page for Live mode)
        if (mode === AppMode.LIVE) {
            window.location.href = '/live-setup';
        } else {
            window.location.href = '/';
        }
    }
}

// Update toggle UI to match mode
function updateToggleUI(mode) {
    // Update button states
    const buttons = document.querySelectorAll('.mode-buttons .mode-btn');
    buttons.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === mode);
    });

    // Update body class for CSS styling
    document.body.classList.remove('mode-live', 'mode-demo', 'mode-live_detailed');
    document.body.classList.add(`mode-${mode}`);

    // Update breadcrumb visibility based on mode
    const detailedBreadcrumb = document.getElementById('detailed-breadcrumb');
    const liveBreadcrumb = document.getElementById('live-breadcrumb');

    if (detailedBreadcrumb && liveBreadcrumb) {
        if (mode === AppMode.LIVE) {
            // Live mode - show 4-step breadcrumb (Setup → Learn → Record → Results)
            detailedBreadcrumb.style.display = 'none';
            liveBreadcrumb.style.display = 'flex';
        } else {
            // Demo or Detailed mode - show 5-step breadcrumb
            detailedBreadcrumb.style.display = 'flex';
            liveBreadcrumb.style.display = 'none';
        }
    }
}

// Initialize mode on page load
function initializeMode() {
    const mode = getMode();

    // Ensure mode is always saved to sessionStorage (for first visit)
    if (!sessionStorage.getItem('appMode')) {
        sessionStorage.setItem('appMode', mode);
    }

    console.log('Mode initialized:', mode, 'Fast mode:', mode === AppMode.LIVE);

    // If in LIVE mode and on the main convert page, redirect to live-setup
    // (Live mode uses the new 4-step workflow starting at /live-setup)
    if (mode === AppMode.LIVE && window.location.pathname === '/') {
        window.location.href = '/live-setup';
        return;
    }

    updateToggleUI(mode);

    // Set up button listeners
    const buttons = document.querySelectorAll('.mode-buttons .mode-btn');
    buttons.forEach(btn => {
        btn.addEventListener('click', () => {
            const newMode = btn.dataset.mode;
            setMode(newMode);
        });
    });

    // Dispatch event for page-specific handlers
    window.dispatchEvent(new CustomEvent('modeInitialized', { detail: { mode } }));
}

// Check if in demo mode
function isDemoMode() {
    return getMode() === AppMode.DEMO;
}

// Check if in live mode (either Live or Live Detailed)
function isLiveMode() {
    const mode = getMode();
    return mode === AppMode.LIVE || mode === AppMode.LIVE_DETAILED;
}

// Check if in live detailed mode
function isLiveDetailedMode() {
    return getMode() === AppMode.LIVE_DETAILED;
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

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    initializeMode();
});

// Export for use in other scripts
window.AppMode = AppMode;
window.ModeNames = ModeNames;
window.getMode = getMode;
window.setMode = setMode;
window.isDemoMode = isDemoMode;
window.isLiveMode = isLiveMode;
window.isLiveDetailedMode = isLiveDetailedMode;
window.isFastMode = isFastMode;
window.getSelectedSample = getSelectedSample;
window.setSelectedSample = setSelectedSample;
