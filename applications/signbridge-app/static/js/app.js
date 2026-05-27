// ══════════════════════════════════════════════════════════════
// SignBridge — Main Application JavaScript
// ══════════════════════════════════════════════════════════════

// ── State (must be first — other modules reference these) ────
let MODE = document.body.dataset.mode || 'demo';
let METHOD = document.body.dataset.method || 'speak';
let conversation = null;
let currentTurn = 0;
let isPlaying = false;
let history = [];
let cameraStream = null;
let mediaRecorder = null;
let recordedChunks = [];
let selectedDomain = 'doctor_visit';
let selectedScenario = 'Doctor Visit';
let interactionMode = 'in-person';
let currentTrainingSubTab = 'scripted';
let currentLiveSubTab = 'scenarios';
let currentNav = 'live';

// SignBridge Live mode (Kiosk / Lanyard / Conf Call). Declared up here
// so that applySignBridgeMode() / syncTtsToggleUI() called from the
// page-load init block (~line 920) don't hit a temporal-dead-zone
// ReferenceError -- the function definitions live further down and
// reference these via closure. Defaults: Kiosk, TTS ON.
let SIGNBRIDGE_MODE = localStorage.getItem('signbridge.mode') || 'kiosk';
// TTS defaults to ON unless the user has explicitly toggled it off
// (stored as '0'). Earlier the default was off which was confusing --
// users expected the signed sentence to be spoken aloud after Send.
let TTS_ENABLED = localStorage.getItem('signbridge.ttsEnabled') !== '0';

// Audible chime when CTQI just crossed below the low threshold.
// Hoisted up here so the page-load init block (~line 920) can read it
// without hitting a temporal-dead-zone ReferenceError. Default off
// (chimes are easy to find annoying).
let CHIME_ON_LOW_CTQI = localStorage.getItem('signbridge.lowCtqiChime') === '1';
let _prevCtqiWasHigh = true;     // start optimistic so the first low one chimes
let _chimeAudioCtx = null;

// CTQI low-alert threshold (0-100). Captions with CTQI below this get
// the pulsing red border + chime (if enabled). User-tunable in
// Settings -> Advanced. Default 80.
let CTQI_LOW_THRESHOLD = (() => {
    const stored = parseInt(localStorage.getItem('signbridge.ctqiThreshold'), 10);
    return (isFinite(stored) && stored >= 0 && stored <= 100) ? stored : 80;
})();

// CTQI hard floor (0-100). When the score drops below this, the model
// is essentially guessing -- we hide the caption text and show
// "[unclear]" instead so we don't confidently present a wrong sentence.
// CTQI badge + Regenerate/Edit chips still appear (user can manually
// rescue the message). User-tunable. Default 40.
let CTQI_HARD_FLOOR = (() => {
    const stored = parseInt(localStorage.getItem('signbridge.ctqiHardFloor'), 10);
    return (isFinite(stored) && stored >= 0 && stored <= 100) ? stored : 40;
})();

// Settings-page state, hoisted up here so the page-load init block
// (~line 920) can call syncSettingsPageUI() without hitting TDZ
// ReferenceErrors. (Same pattern as CHIME_ON_LOW_CTQI / CTQI_LOW_THRESHOLD
// up above. Any future Settings-page state with let/const that the
// sync function transitively reads must also be hoisted here.)
let _settingsEditing = false;
let _settingsSnapshot = null;
const _SETTINGS_INPUTS = [
    'setDisplayName',
    'setTtsEnabled',
    'setChimeEnabled',
    'setWakeWord',
    'setCtqiThreshold',
    'setCtqiHardFloor',
];

// Hoisted up from the kiosk-mode section further below. selectCameraSource
// (called indirectly from applySignBridgeMode at init time) reads from
// KIOSK_PARAMS; without this earlier declaration, a Lanyard-mode default
// would hit a TDZ ReferenceError on page load and break the whole script.
const KIOSK_PARAMS = new URLSearchParams(window.location.search);
const IS_KIOSK = KIOSK_PARAMS.get('kiosk') === 'true';

// ── Feature flags ──
// Set to true to convert spoken English to ASL glosses + play sign bank videos.
// Set to false for speech-to-text only (deaf users read English directly).
const SPEAK_MODE_TRANSLATE_TO_ASL = false;

// ── External camera state ──
let externalCameraUrl = null;       // MJPEG URL (null = use device camera)
let externalCameraImg = null;       // <img> element loading MJPEG stream
let externalCameraCanvas = null;    // hidden canvas for frame capture
let externalCameraInterval = null;  // setInterval handle for canvas draw loop

// Generic labels regardless of demo/live mode so transcript reads naturally
// across any scenario (medical, emergency, banking, etc.).
const SPEAKER_LABEL = 'Speaker';
const SIGNER_LABEL  = 'Signer (ASL \u2192 English)';

// ── Toast Notification System (1.6) ──────────────────────────
const toastContainer = document.createElement('div');
toastContainer.className = 'toast-container';
document.body.appendChild(toastContainer);

function showToast(message, type = 'info', duration = 4000) {
    const icons = { error: '\u26A0\uFE0F', success: '\u2705', info: '\u2139\uFE0F', warn: '\u26A0\uFE0F' };
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `
        <span class="toast-icon">${icons[type] || icons.info}</span>
        <span class="toast-msg">${message}</span>
        <button class="toast-close" onclick="this.parentElement.remove()">\u2715</button>
    `;
    toastContainer.appendChild(toast);
    setTimeout(() => {
        toast.classList.add('toast-exit');
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

// ── Theme Toggle (1.9) ──────────────────────────────────────
function initTheme() {
    const saved = localStorage.getItem('signbridge-theme');
    if (saved) {
        document.documentElement.setAttribute('data-theme', saved);
    }
    updateThemeIcon();
}

function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    if (next === 'light') {
        document.documentElement.removeAttribute('data-theme');
    } else {
        document.documentElement.setAttribute('data-theme', 'dark');
    }
    localStorage.setItem('signbridge-theme', next);
    updateThemeIcon();
}

function updateThemeIcon() {
    const btn = document.getElementById('themeToggle');
    if (!btn) return;
    const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
    btn.textContent = isDark ? '\u2600\uFE0F' : '\uD83C\uDF19';
    btn.title = isDark ? 'Switch to light mode' : 'Switch to dark mode';
}

initTheme();

// ── Hash-based routing (1.11) ────────────────────────────────
function pushRoute(route) {
    if (window.location.hash !== '#' + route) {
        window.history.pushState(null, '', '#' + route);
    }
}

window.addEventListener('popstate', () => {
    const hash = window.location.hash.replace('#', '');
    if (!hash || hash === 'home') {
        showHomeScreen(true); // skipPush = true to avoid loop
    }
});

// ── Unified settings panel (camera source + signing zone) ────────────────
function toggleSettings() {
    const panel = document.getElementById('settingsPanel');
    if (!panel) return;
    const showing = panel.style.display === 'none' || !panel.style.display;
    panel.style.display = showing ? 'block' : 'none';

    // If external camera is active, reflect that in the radio buttons when
    // opening (so the user lands on the right state).
    if (showing && externalCameraUrl) {
        const wifiRadio = document.querySelector('input[name="cameraSource"][value="wifi"]');
        if (wifiRadio) wifiRadio.checked = true;
        selectCameraSource('wifi');
    }
}

// Legacy names — kept as thin aliases so any other code that still references
// them keeps working. Both forward to the unified panel.
function toggleCameraSourcePanel() { toggleSettings(); }
function toggleStreamSettings()     { toggleSettings(); }

function selectCameraSource(type) {
    const wifiRow = document.getElementById('wifiCameraInputRow');
    if (type === 'wifi') {
        wifiRow.style.display = 'flex';
        // Pre-fill: kiosk param > localStorage > current external URL > empty
        const urlInput = document.getElementById('wifiCameraUrl');
        if (!urlInput.value) {
            const defaultUrl = KIOSK_PARAMS.get('camera')
                || localStorage.getItem('signbridge-wifi-camera-url')
                || (externalCameraUrl || '');
            urlInput.value = defaultUrl;
        }
    } else {
        wifiRow.style.display = 'none';
        // Switch back to device camera
        if (externalCameraUrl) {
            externalCameraUrl = null;
            stopExternalCamera();
            document.getElementById('wifiCameraStatus').style.display = 'none';
            document.querySelector('.camera-title').textContent = 'Your Camera';
            // Restore device-camera strict defaults (tighter motion thresholds,
            // narrower zone that excludes desk/lap noise).
            if (signSegmenter && typeof signSegmenter.tune === 'function') {
                signSegmenter.tune({
                    motion_threshold: 325000,
                    motion_threshold_continue: 120000,
                    zone_top: 0.0,
                    zone_bottom: 0.9,
                });
            }
            // Reset digital zoom; device cam has narrower FOV natively.
            if (signSegmenter && typeof signSegmenter.setZoom === 'function') {
                signSegmenter.setZoom(1.0);
            }
            // Device cam frames already have correct handedness for inference.
            if (signSegmenter && typeof signSegmenter.setMirror === 'function') {
                signSegmenter.setMirror(false);
            }
            // Restart with device camera if live
            if (liveActive) {
                stopCamera();
                startCamera();
            }
        }
    }
}

async function connectWifiCamera() {
    const urlInput = document.getElementById('wifiCameraUrl');
    const statusEl = document.getElementById('wifiCameraStatus');
    const connectBtn = document.getElementById('wifiConnectBtn');
    const url = urlInput.value.trim();

    if (!url) {
        statusEl.style.display = 'block';
        statusEl.style.color = '#ff6666';
        statusEl.textContent = 'Enter a camera URL';
        return;
    }

    connectBtn.disabled = true;
    connectBtn.textContent = 'Connecting...';
    statusEl.style.display = 'block';
    statusEl.style.color = 'var(--text-muted)';
    statusEl.textContent = 'Connecting to camera...';

    try {
        // Swap the underlying camera, but DON'T tear down signSegmenter --
        // the WS streamer reads from cameraFeed.videoEl, which keeps its
        // identity across the swap; only its srcObject changes. Calling
        // signSegmenter.stop() here closes the WS and kills the frame loop,
        // and StreamingLiveMode.start() is a no-op (compat shim), so the
        // streamer never recovers and the server never sees a single frame.
        stopCamera();

        externalCameraUrl = url;
        localStorage.setItem('signbridge-wifi-camera-url', url);

        await startCamera();

        // Clear server-side segmenter state so any half-open segment from
        // the previous camera doesn't bleed into the WiFi feed. Also turn
        // OFF the hand-presence gate: at QVGA + quality=25 the ESP32 stream
        // is too blocky for MediaPipe Hands to find a hand, so the gate
        // silently suppresses every motion event. The motion + zone gate
        // alone is sufficient for WiFi-cam framing (camera is mounted, not
        // handheld, so chair-scrape false positives are rare).
        // Clear server-side segmenter state so any half-open segment from
        // the previous camera doesn't bleed into the WiFi feed.
        if (signSegmenter && signSegmenter._ws &&
            signSegmenter._ws.readyState === WebSocket.OPEN) {
            try { signSegmenter._ws.send(JSON.stringify({ type: 'reset' })); } catch (_) {}
        }
        // WiFi-cam tuning for the VGA/lanyard setup:
        //
        //   motion_threshold 325k -> 220k, continue 120k -> 80k
        //     Halfway between the QVGA-low (150k/50k) and device-cam strict
        //     (325k/120k) defaults. VGA-q=15 pumps more pixel-noise than the
        //     old QVGA-q=12, so we don't need to drop as low to clear real
        //     signing, but still below device-camera baseline since hands at
        //     3-5 ft are smaller than at desk-camera distance.
        //
        //   require_hand_present stays true (server default)
        //     With VGA + 480x360 client capture, MediaPipe Hands sees ~80px
        //     hands at conversational distance -- enough for reliable detection.
        //     The gate is back to being useful (chair-scrape, face-turn).
        //     If it ever starts suppressing real signing again, drop confidence
        //     server-side or set require_hand_present: false here.
        if (signSegmenter && typeof signSegmenter.tune === 'function') {
            signSegmenter.tune({
                motion_threshold: 220000,
                motion_threshold_continue: 80000,
                // Open the zone for lanyard framing. The default zone_bottom=0.9
                // excludes the bottom 10% (desk/lap noise for laptop cams), but
                // on a lanyard pointing up, the signer's hands at chest height
                // sit in that bottom strip and never register motion. Push
                // zone_bottom to 1.0 and keep zone_top at 0; the user can
                // narrow either side via the UI sliders if face-triggers or
                // sky-triggers become a problem.
                zone_top: 0.0,
                zone_bottom: 1.0,
            });
        }
        // Match laptop-cam framing. ESP32 OV2640 has ~75deg horizontal FOV
        // vs the laptop cam's ~60deg, so at equal distance the signer
        // looks ~30% smaller in frame -- which means smaller hands for
        // MediaPipe even with VGA source. 1.3x digital zoom recovers the
        // tighter framing without losing resolution (still downsampling
        // into 480x360 target). Tune live via signSegmenter.setZoom(N).
        if (signSegmenter && typeof signSegmenter.setZoom === 'function') {
            signSegmenter.setZoom(1.3);
        }
        // Two mirrors, by design:
        //   1) Canvas-bridge mirror in startExternalCamera flips the MJPEG
        //      so the preview looks like the device cam (selfie-feel).
        //   2) setMirror(true) here flips the inference frame in _sendFrame
        //      so it lands back at the ESP32's original orientation, which
        //      empirically matches OpenHands' training convention (signer's
        //      right hand on viewer's RIGHT -- the model was trained on
        //      pre-mirrored video, not standard un-mirrored).
        // Both mirrors compound to give: preview = selfie, inference =
        // mirrored. Validated live by single-handed-sign accuracy.
        if (signSegmenter && typeof signSegmenter.setMirror === 'function') {
            signSegmenter.setMirror(true);
        }

        statusEl.style.color = '#4caf80';
        statusEl.textContent = 'Connected';
        // The settings popup that contains the camera-source + zone controls
        // used to be id="cameraSourcePanel". Was renamed to settingsPanel when
        // we merged the two panels into one. Tolerate either for safety.
        const _settingsPanel = document.getElementById('settingsPanel')
                            || document.getElementById('cameraSourcePanel');
        if (_settingsPanel) _settingsPanel.style.display = 'none';

        if (liveActive) {
            continuousRecording = true;
            isRecordingSign = true;
            statusBar.innerHTML = '<span class="processing">Sign when ready \u2014 your signs will appear here</span>';
        }
    } catch (e) {
        statusEl.style.color = '#ff6666';
        statusEl.textContent = 'Failed: ' + e.message;
        externalCameraUrl = null;
    }

    connectBtn.disabled = false;
    connectBtn.textContent = 'Connect';
}

// ── Kiosk Mode (WI-3) ───────────────────────────────────────
// KIOSK_PARAMS / IS_KIOSK hoisted to top-of-file (above) to fix a TDZ
// that bit at page-load init time.
let kioskIdleTimer = null;
const KIOSK_IDLE_TIMEOUT = 30000; // 30 seconds

if (IS_KIOSK) {
    document.body.classList.add('kiosk-mode');
    // Pre-configure from URL params
    const kioskCamera = KIOSK_PARAMS.get('camera');
    const kioskDomain = KIOSK_PARAMS.get('domain');
    if (kioskCamera) externalCameraUrl = kioskCamera;
    if (kioskDomain) selectedDomain = kioskDomain;
}

function kioskResetIdleTimer() {
    if (!IS_KIOSK) return;
    if (kioskIdleTimer) clearTimeout(kioskIdleTimer);
    kioskIdleTimer = setTimeout(() => {
        kioskShowTapToStart();
    }, KIOSK_IDLE_TIMEOUT);
}

function kioskShowTapToStart() {
    // Stop everything
    stopLiveMode();
    stopCamera();
    history = [];
    const msgs = document.getElementById('transcriptMsgs');
    if (msgs) msgs.innerHTML = '';

    // Show overlay
    let overlay = document.getElementById('kioskTapOverlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.id = 'kioskTapOverlay';
        overlay.className = 'kiosk-tap-overlay';
        overlay.innerHTML = '<h1>SignBridge</h1><p>Tap anywhere to start a conversation</p>';
        overlay.onclick = () => kioskStart();
        document.body.appendChild(overlay);
    }
    overlay.style.display = 'flex';
}

async function kioskStart() {
    const overlay = document.getElementById('kioskTapOverlay');
    if (overlay) overlay.style.display = 'none';

    // Jump straight into conversation
    navigateTo('live');
    // Give DOM a moment to update
    await sleep(100);
    const btnStart = document.getElementById('btnStart');
    if (btnStart) btnStart.click();
    kioskResetIdleTimer();
}

// ── Mode Switch Float Button (WI-4) ─��───────────────────────
function createModeSwitchButton() {
    const existing = document.getElementById('modeSwitchFloat');
    if (existing) existing.remove();

    const btn = document.createElement('button');
    btn.id = 'modeSwitchFloat';
    btn.className = 'mode-switch-float';
    updateModeSwitchButton(btn);
    btn.onclick = () => switchSignSpeakMode();

    const cameraBox = document.querySelector('.camera-box');
    if (cameraBox) cameraBox.appendChild(btn);
}

function updateModeSwitchButton(btn) {
    btn = btn || document.getElementById('modeSwitchFloat');
    if (!btn) return;
    if (METHOD === 'sign') {
        btn.textContent = 'Switch to Speak';
        btn.style.background = '#7cb3f0';
    } else {
        btn.textContent = 'Switch to Sign';
        btn.style.background = '#b07cf0';
    }
}

async function switchSignSpeakMode() {
    const newMethod = METHOD === 'sign' ? 'speak' : 'sign';

    // Stop current mode
    if (signSegmenter) signSegmenter.stop();
    continuousRecording = false;
    isRecordingSign = false;
    if (liveRecognition) {
        try { liveRecognition.stop(); } catch(e) {}
        liveRecognition = null;
    }

    METHOD = newMethod;

    // Update header toggles
    const htogSign = document.getElementById('htogSign');
    const htogSpeak = document.getElementById('htogSpeak');
    if (htogSign) htogSign.classList.toggle('active', METHOD === 'sign');
    if (htogSpeak) htogSpeak.classList.toggle('active', METHOD === 'speak');

    updateModeSwitchButton();

    // Restart the appropriate mode
    if (METHOD === 'sign') {
        const seg = await ensureSegmenter();
        seg.clearGlosses();
        liveCollectedGlosses = [];
        continuousRecording = true;
        isRecordingSign = true;
        if (cameraStream) seg.start(cameraStream);
        statusBar.innerHTML = '<span class="processing">Sign when ready \u2014 your signs will appear here</span>';
        // No speech recognition in sign mode — avoids background noise interruptions
    } else {
        const speakHint = SPEAK_MODE_TRANSLATE_TO_ASL
            ? 'Speak naturally \u2014 your words will be converted to ASL glosses'
            : 'Speak naturally \u2014 your words will appear as text for the signer';
        statusBar.innerHTML = `<span class="prompt"><span class="listening-indicator"></span>${speakHint}</span>`;
        startContinuousListeningForSpeaker();
    }

    // Hide/show relevant buttons
    const btnRecordSign = document.getElementById('btnRecordSign');
    const btnSendMessage = document.getElementById('btnSendMessage');
    const btnSpeakSend = document.getElementById('btnSpeakSend');
    if (btnRecordSign) btnRecordSign.style.display = 'none';
    if (btnSendMessage) btnSendMessage.style.display = 'none';
    if (btnSpeakSend) btnSpeakSend.style.display = METHOD === 'speak' ? 'inline-block' : 'none';

    showToast(`Switched to ${METHOD === 'sign' ? 'Sign' : 'Speak'} mode`, 'info', 2000);
    kioskResetIdleTimer();
}

// ── Debug Panel ──────────────────────────────────────────────
const DEBUG_ENABLED = new URLSearchParams(window.location.search).get('debug') === 'true';

function toggleDebugPanel() {
    const panel = document.getElementById('debugPanel');
    if (!panel) return;
    panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
}

// Old wrist-velocity diagnostics removed -- the streaming pipeline doesn't
// produce those signals. Kept as a no-op so existing call sites (in the
// retired SignSegmenter path) don't throw if they ever execute.
function updateDebugPanel(_data) { /* no-op in streaming mode */ }

// Append a gloss row to the scrollable diagnostics list. Newest first.
// Top-1 prominent (gloss + confidence); alts smaller and dimmer.
// Low-confidence rows (below the model's confidence gate) are rendered
// dimmed + flagged so the user can see what was detected but knows it
// didn't make it into the LLM buffer.
function _appendGlossRow(pred) {
    const panel = document.getElementById('glossListPanel');
    if (!panel) return;
    const empty = panel.querySelector('.gloss-list-empty');
    if (empty) empty.remove();

    const isConfident = pred.confident !== false;
    const row = document.createElement('div');
    row.className = 'gloss-row' + (isConfident ? '' : ' unconfident');
    const conf = Math.round((pred.confidence || 0) * 100);
    const alts = (pred.top_k || []).slice(1, 3)
        .map(t => (t.gloss || '').toUpperCase())
        .filter(Boolean);
    const lowTag = isConfident ? '' : '<span class="gloss-low-tag">below threshold — not sent</span>';
    row.innerHTML =
        `<span class="gloss-main">${(pred.gloss || '').toUpperCase()}</span>` +
        `<span class="gloss-conf">${conf}%</span>` +
        lowTag +
        (alts.length ? `<span class="gloss-alts">${alts.join(' · ')}</span>` : '');
    panel.insertBefore(row, panel.firstChild);
}

function _clearGlossRows() {
    const panel = document.getElementById('glossListPanel');
    if (!panel) return;
    panel.innerHTML = '<div class="gloss-list-empty">No glosses detected yet</div>';
}

// Set threshold marker position
function initDebugThreshold() {
    const thresh = document.getElementById('dbgVelocityThresh');
    if (thresh) {
        // 0.012 threshold on 0-0.05 scale = 24%
        thresh.style.left = '24%';
    }
}

// Test single sign — bypasses segmenter, records 2.5s, sends to backend
async function testSingleSign() {
    const btn = document.getElementById('btnTestSign');
    const resultEl = document.getElementById('dbgTestResult');
    if (!cameraStream) {
        resultEl.style.display = 'block';
        resultEl.innerHTML = '<span style="color:#ff6666">No camera stream</span>';
        return;
    }

    btn.disabled = true;
    btn.textContent = 'Recording 2.5s...';
    resultEl.style.display = 'block';
    resultEl.innerHTML = '<span style="color:var(--text-muted)">Recording...</span>';

    // Record 2.5 seconds
    const blob = await recordFromCamera(2500);

    if (!blob) {
        resultEl.innerHTML = '<span style="color:#ff6666">No video captured</span>';
        btn.disabled = false;
        btn.textContent = 'Test Sign (2.5s timed record)';
        return;
    }

    btn.textContent = 'Analyzing...';
    resultEl.innerHTML = `<span style="color:var(--text-muted)">Sending ${(blob.size / 1024).toFixed(0)}KB to backend...</span>`;

    try {
        const formData = new FormData();
        formData.append('video', blob, 'test.webm');
        formData.append('domain', selectedDomain);

        const resp = await fetch('/api/process-sign', { method: 'POST', body: formData });
        const result = await resp.json();

        if (result.success) {
            const topK = (result.top_k || []).slice(0, 3).map(k =>
                `${k.gloss} (${(k.confidence * 100).toFixed(1)}%)`
            ).join(', ');
            resultEl.innerHTML = `<span style="color:#4caf80;font-weight:600">${result.gloss}</span> ` +
                `<span style="color:var(--text-muted)">${(result.confidence * 100).toFixed(1)}%</span>` +
                (topK ? `<br><span style="color:var(--text-muted)">Top 3: ${topK}</span>` : '');
        } else {
            resultEl.innerHTML = `<span style="color:#ff6666">Error: ${result.error}</span>`;
        }
    } catch (e) {
        resultEl.innerHTML = `<span style="color:#ff6666">Failed: ${e.message}</span>`;
    }

    btn.disabled = false;
    btn.textContent = 'Test Sign (2.5s timed record)';
}

if (DEBUG_ENABLED) {
    setTimeout(() => {
        const panel = document.getElementById('debugPanel');
        if (panel) panel.style.display = 'block';
        initDebugThreshold();
    }, 500);
}

// ── Camera Permission Modal (1.5) ────────────────────────────
async function requestCameraWithPrompt(needsAudio) {
    const overlay = document.getElementById('cameraPermModal');
    return new Promise((resolve) => {
        overlay.classList.add('visible');
        const allowBtn = document.getElementById('cameraPermAllow');
        const skipBtn = document.getElementById('cameraPermSkip');

        function cleanup() {
            overlay.classList.remove('visible');
            allowBtn.removeEventListener('click', onAllow);
            skipBtn.removeEventListener('click', onSkip);
        }

        async function onAllow() {
            cleanup();
            try {
                const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
                const isPortrait = window.innerHeight > window.innerWidth;
                // Desktop: 640x480 matches /ws-test exactly so the camera framing/zoom
                // the user verified in testing carries forward unchanged into the live
                // streaming mode. Mobile portrait keeps its vertical aspect.
                const video = (isMobile && isPortrait)
                    ? { facingMode: 'user', width: { ideal: 720 }, height: { ideal: 1280 }, aspectRatio: { ideal: 9/16 } }
                    : { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } };
                const constraints = { video, audio: !!needsAudio };
                const stream = await navigator.mediaDevices.getUserMedia(constraints);
                resolve(stream);
            } catch (e) {
                showToast('Camera access denied. Check browser permissions (lock icon in address bar).', 'error', 6000);
                resolve(null);
            }
        }

        function onSkip() {
            cleanup();
            resolve(null);
        }

        allowBtn.addEventListener('click', onAllow);
        skipBtn.addEventListener('click', onSkip);
    });
}

// ══════════════════════════════════════════════════════════════
// USER PROFILE (2.12) — localStorage-based personalization
// ══════════════════════════════════════════════════════════════
const PROFILE_KEY = 'signbridge-profile';

function loadProfile() {
    try {
        const raw = localStorage.getItem(PROFILE_KEY);
        if (raw) return JSON.parse(raw);
    } catch (e) {}
    return {
        name: '',
        method: 'speak',
        saveHistory: true,
        joinedDate: new Date().toISOString().split('T')[0],
        practiceStats: {},   // { gloss: { attempts, correct } }
        recentScenarios: [], // [{ domain, label, timestamp }]
        conversations: [],   // saved live conversation transcripts
    };
}

function saveProfile(profile) {
    try {
        localStorage.setItem(PROFILE_KEY, JSON.stringify(profile));
    } catch (e) {
        console.warn('[Profile] Could not save:', e);
    }
}

function initProfile() {
    const profile = loadProfile();

    // (Header profile button removed -- panel folded into Settings tab.)

    // Apply saved method preference (still respected for non-Kiosk/Lanyard
    // modes; setSignBridgeMode pins METHOD='sign' in those two modes).
    if (profile.method) {
        METHOD = profile.method;
        const htogSign = document.getElementById('htogSign');
        const htogSpeak = document.getElementById('htogSpeak');
        if (htogSign) htogSign.classList.toggle('active', profile.method === 'sign');
        if (htogSpeak) htogSpeak.classList.toggle('active', profile.method === 'speak');
    }

    // Show welcome message
    const welcomeEl = document.getElementById('heroWelcome');
    if (welcomeEl && profile.name) {
        welcomeEl.textContent = `Welcome back, ${profile.name}`;
        welcomeEl.style.display = 'inline';
    }
}

function openProfile() {
    // Profile panel was removed in favor of the Settings tab.
    // Redirect any legacy openProfile() calls to navigate there instead.
    if (typeof navigateTo === 'function') navigateTo('settings');
}

function closeProfile() {
    // No-op now that the profile overlay is gone. Kept for any legacy
    // callers (e.g., inline onclicks) that might still reference it.
}

function saveProfileName() {
    // Legacy entry point -- the new flow goes through the Settings
    // tab's Edit/Save buttons. Keep as a defensive no-op so any stale
    // call site doesn't throw.
}

function saveProfilePrefs() {
    // Same as above -- handled by onSettingsSaveClick now.
}

function recordRecentScenario(domain, label) {
    const profile = loadProfile();
    // Remove duplicate if exists
    profile.recentScenarios = (profile.recentScenarios || []).filter(s => s.domain !== domain);
    // Add to front
    profile.recentScenarios.unshift({
        domain,
        label,
        timestamp: new Date().toISOString(),
    });
    // Keep only last 5
    profile.recentScenarios = profile.recentScenarios.slice(0, 5);
    saveProfile(profile);
}

function renderRecentScenarios(profile) {
    const list = document.getElementById('profileRecentList');
    if (!list) return;   // Profile panel was removed; no DOM target.
    const recents = profile.recentScenarios || [];
    if (recents.length === 0) {
        list.innerHTML = '<div style="font-size:0.82rem;color:var(--text-muted)">No recent activity</div>';
        return;
    }

    list.innerHTML = '';
    recents.forEach(item => {
        // Get icon from registry if loaded
        let icon = '?';
        if (registryData && registryData.domains && registryData.domains[item.domain]) {
            icon = registryData.domains[item.domain].icon;
        }
        const timeAgo = formatTimeAgo(item.timestamp);
        const el = document.createElement('div');
        el.className = 'profile-recent-item';
        el.innerHTML = `
            <div class="profile-recent-icon">${icon}</div>
            <span class="profile-recent-label">${item.label}</span>
            <span class="profile-recent-time">${timeAgo}</span>
        `;
        el.onclick = () => {
            closeProfile();
            selectScenario(item.domain, item.label);
        };
        list.appendChild(el);
    });
}

function formatTimeAgo(isoStr) {
    const diff = Date.now() - new Date(isoStr).getTime();
    const mins = Math.floor(diff / 60000);
    if (mins < 1) return 'just now';
    if (mins < 60) return mins + 'm ago';
    const hrs = Math.floor(mins / 60);
    if (hrs < 24) return hrs + 'h ago';
    const days = Math.floor(hrs / 24);
    return days + 'd ago';
}

function renderProfileHistory(profile) {
    const summary = document.getElementById('profileHistorySummary');
    const link = document.getElementById('profileHistoryLink');
    if (!summary && !link) return;   // Profile panel removed; nothing to render.
    const convs = profile.conversations || [];
    if (convs.length === 0) {
        if (summary) summary.textContent = 'No conversations saved yet';
        if (link) link.style.display = 'none';
    } else {
        if (summary) summary.textContent = `${convs.length} conversation${convs.length !== 1 ? 's' : ''} saved`;
        if (link) link.style.display = 'inline-block';
    }
}

function showHistoryPage() {
    hideHomeContent();
    document.getElementById('historyScreen').style.display = 'block';
    lockHeaderToggles();

    // Show list view, hide viewer
    document.getElementById('historyListView').style.display = 'block';
    document.getElementById('histViewerView').style.display = 'none';

    renderHistoryList();
}

function renderHistoryList() {
    const profile = loadProfile();
    const convs = profile.conversations || [];
    const container = document.getElementById('histListContainer');

    if (convs.length === 0) {
        container.innerHTML = '<div class="hist-empty">No conversations saved yet. Live conversations are saved automatically.</div>';
        return;
    }

    container.innerHTML = '';
    convs.forEach((conv, idx) => {
        let icon = '?';
        if (registryData && registryData.domains && registryData.domains[conv.domain]) {
            icon = registryData.domains[conv.domain].icon;
        }

        const date = new Date(conv.timestamp);
        const dateStr = date.toLocaleDateString([], { month: 'short', day: 'numeric' });
        const timeStr = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        const msgCount = conv.messages.length;
        const preview = conv.messages.find(m => m.speaker === 'patient' || m.speaker === 'doctor');
        const previewText = preview ? preview.text : '';

        const card = document.createElement('div');
        card.className = 'hist-card';
        card.innerHTML = `
            <div class="hist-card-icon">${icon}</div>
            <div class="hist-card-info">
                <div class="hist-card-title">${conv.scenario}</div>
                <div class="hist-card-meta">
                    <span>${dateStr} ${timeStr}</span>
                    <span>${msgCount} messages</span>
                </div>
                <div class="hist-card-preview">${previewText}</div>
            </div>
            <div class="hist-card-actions">
                <button class="hist-delete-btn" title="Delete" onclick="event.stopPropagation(); deleteConversation(${idx})">&#x1F5D1;</button>
            </div>
        `;
        card.addEventListener('click', () => viewConversation(idx));
        container.appendChild(card);
    });
}

function viewConversation(idx) {
    const profile = loadProfile();
    const conv = (profile.conversations || [])[idx];
    if (!conv) return;

    document.getElementById('historyListView').style.display = 'none';
    document.getElementById('histViewerView').style.display = 'block';

    const date = new Date(conv.timestamp);
    const dateStr = date.toLocaleDateString([], { weekday: 'short', month: 'short', day: 'numeric', year: 'numeric' });
    const timeStr = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const methodLabel = conv.method === 'sign' ? 'I Sign' : 'I Speak';

    document.getElementById('histViewerTitle').textContent = conv.scenario;
    document.getElementById('histViewerMeta').textContent = `${methodLabel} \u00B7 ${dateStr} at ${timeStr} \u00B7 ${conv.messages.length} messages`;

    const msgsEl = document.getElementById('histViewerMsgs');
    msgsEl.innerHTML = '';

    conv.messages.forEach(msg => {
        const type = (msg.speaker === 'doctor' || msg.speaker === 'Speaker') ? 'doctor' : 'patient';
        const speakerLabel = type === 'doctor' ? 'Speaker' : 'Signer';
        const d = document.createElement('div');
        d.className = 'msg ' + type;
        d.innerHTML = `<div class="speaker">${speakerLabel}</div><div>${msg.text}</div>`;
        msgsEl.appendChild(d);
    });
}

function backToHistoryList() {
    document.getElementById('histViewerView').style.display = 'none';
    document.getElementById('historyListView').style.display = 'block';
}

function deleteConversation(idx) {
    const profile = loadProfile();
    if (!profile.conversations) return;
    profile.conversations.splice(idx, 1);
    saveProfile(profile);
    renderHistoryList();
    showToast('Conversation deleted', 'info');
}

// Practice stats persistence
function recordPracticeResult(gloss, correct) {
    const profile = loadProfile();
    if (!profile.practiceStats) profile.practiceStats = {};
    if (!profile.practiceStats[gloss]) {
        profile.practiceStats[gloss] = { attempts: 0, correct: 0 };
    }
    profile.practiceStats[gloss].attempts++;
    if (correct) profile.practiceStats[gloss].correct++;
    saveProfile(profile);
}

function clearProfileData() {
    if (!confirm('Clear all your SignBridge data? This will reset your name, practice stats, and recent scenarios.')) return;
    localStorage.removeItem(PROFILE_KEY);
    const welcomeEl = document.getElementById('heroWelcome');
    if (welcomeEl) welcomeEl.style.display = 'none';
    showToast('Profile data cleared', 'info');
}

// Init on page load
initProfile();
applySignBridgeMode();
syncTtsToggleUI();
syncWakeWordUI();
syncLowCtqiChimeUI();
syncSettingsPageUI();
_wireNavLiveSubmenu();

// ══════════════════════════════════════════════════════════════
// CONVERSATION HISTORY (2.3)
// ══════════════════════════════════════════════════════════════
const MAX_SAVED_CONVERSATIONS = 20;

function saveConversation() {
    // Conversation history feature was removed; this is now a no-op.
    // Existing call sites (stopLiveMode, etc.) can keep invoking it
    // harmlessly.
}

function closeHistoryViewer() {
    const histScreen = document.getElementById('historyScreen');
    if (histScreen) histScreen.style.display = 'none';
}

// ══════════════════════════════════════════════════════════════
// REGISTRY — Single source of truth for domains
// ══════════════════════════════════════════════════════════════
let registryData = null;

async function loadRegistry() {
    try {
        console.log('[Registry] Fetching /api/registry...');
        const resp = await fetch('/api/registry');
        if (!resp.ok) throw new Error('HTTP ' + resp.status);
        registryData = await resp.json();
        console.log('[Registry] Loaded', Object.keys(registryData.domains || {}).length, 'domains');
        renderAllScenarioGrids();
        // Re-apply current nav state so the right section is visible
        navigateTo(currentNav);
        console.log('[Registry] Scenarios rendered');
    } catch (e) {
        console.error('[Registry] Failed to load:', e);
        showToast('Could not load domain registry', 'error');
        // Render a fallback so the page isn't empty
        const demoEl = document.getElementById('demoScenarios');
        if (demoEl) demoEl.innerHTML = '<p style="color:var(--text-muted);text-align:center;padding:20px;">Could not load scenarios. Please refresh the page.</p>';
    }
}

const STATUS_BADGES = {
    ready:    { label: 'Ready', class: 'badge-healthcare' },
    training: { label: 'Training', class: 'badge-training' },
    upcoming: { label: 'Coming soon', class: 'badge-coming' },
};

function buildScenarioCard(domainKey, entry, { onclick, isFirst = false, captionMode = false, compact = false } = {}) {
    const isReady = entry.status === 'ready';
    const badge = STATUS_BADGES[entry.status] || STATUS_BADGES.upcoming;
    const description = captionMode
        ? 'Upload a signing video \u2014 get it back with English captions burned in'
        : entry.description;

    const card = document.createElement('div');
    card.className = 'scenario-card' + (isReady ? (compact ? '' : ' featured') : ' coming-soon');
    if (isReady && onclick) {
        card.onclick = onclick;
    }
    card.innerHTML = `
        <div class="scenario-icon"><span class="ico">${entry.icon}</span></div>
        <div class="scenario-info">
            <h3>${entry.label}</h3>
            <p>${description}</p>
            <span class="scenario-badge ${badge.class}">${isReady ? entry.label + ' model' : badge.label}</span>
            ${isFirst && isReady ? '<span class="scenario-badge badge-start-here">Start here</span>' : ''}
        </div>
    `;
    return card;
}

function renderAllScenarioGrids() {
    if (!registryData || !registryData.domains) return;
    const domains = registryData.domains;
    const entries = Object.entries(domains);

    const readyDomains = entries.filter(([, e]) => e.status === 'ready');
    const otherDomains = entries.filter(([, e]) => e.status !== 'ready');

    // ── Demo scenarios (only those with demo_available) ──
    const demoEl = document.getElementById('demoScenarios');
    demoEl.innerHTML = '';
    demoEl.classList.add('live-card-style');
    const demoReady = entries.filter(([, e]) => e.demo_available);
    const demoOther = entries.filter(([, e]) => !e.demo_available);

    if (demoReady.length > 0) {
        demoEl.appendChild(makeLabel('Scripted Demos'));
        const readyGrid = makeGrid();
        demoReady.forEach(([key, entry], i) => {
            readyGrid.appendChild(buildScenarioCard(key, entry, {
                onclick: () => selectScenario(key, entry.label),
                isFirst: i === 0,
                compact: true,
            }));
        });
        demoEl.appendChild(readyGrid);
    }
    if (demoOther.length > 0) {
        demoEl.appendChild(makeLabel('More Demos Coming Soon'));
        const otherGrid = makeGrid();
        demoOther.forEach(([key, entry]) => {
            otherGrid.appendChild(buildScenarioCard(key, entry, { compact: true }));
        });
        demoEl.appendChild(otherGrid);
    }

    // ── Live scenarios (compact card style) ──
    const liveEl = document.getElementById('liveScenarios');
    liveEl.classList.add('live-card-style');
    const existingGrids = liveEl.querySelectorAll('.home-section-label, .scenario-grid');
    existingGrids.forEach(el => el.remove());

    if (readyDomains.length > 0) {
        liveEl.appendChild(makeLabel('Specialized'));
        const liveReadyGrid = makeGrid();
        readyDomains.forEach(([key, entry]) => {
            liveReadyGrid.appendChild(buildScenarioCard(key, entry, {
                onclick: () => selectScenario(key, entry.label),
                compact: true,
            }));
        });
        liveEl.appendChild(liveReadyGrid);
    }
    if (otherDomains.length > 0) {
        liveEl.appendChild(makeLabel('More Scenarios'));
        const liveOtherGrid = makeGrid();
        otherDomains.forEach(([key, entry]) => {
            liveOtherGrid.appendChild(buildScenarioCard(key, entry, { compact: true }));
        });
        liveEl.appendChild(liveOtherGrid);
    }

    // ── Upload domain dropdown ──
    const uploadSelect = document.getElementById('uploadDomainSelect');
    if (uploadSelect) {
        uploadSelect.innerHTML = '';
        readyDomains.forEach(([key, entry]) => {
            const opt = document.createElement('option');
            opt.value = key;
            opt.textContent = entry.label;
            uploadSelect.appendChild(opt);
        });
    }
}

function makeLabel(text) {
    const el = document.createElement('div');
    el.className = 'home-section-label';
    el.textContent = text;
    return el;
}

function makeGrid() {
    const el = document.createElement('div');
    el.className = 'scenario-grid';
    return el;
}

// Load registry on page load
loadRegistry();

// ══════════════════════════════════════════════════════════════
// INTERACTION MODE / HEADER TOGGLES
// ══════════════════════════════════════════════════════════════

// ══════════════════════════════════════════════════════════════
// INTERACTION MODE (In-Person vs Video Call)
// ══════════════════════════════════════════════════════════════
function setInteractionMode(mode) {
    interactionMode = mode;
    // Update all toggle buttons with this mode
    document.querySelectorAll('#liveModeToggle .mode-btn').forEach(btn => {
        btn.classList.toggle('active', btn.textContent.toLowerCase().includes(
            mode === 'in-person' ? 'in-person' : 'video'
        ));
    });
}

// ══════════════════════════════════════════════════════════════
// SIGNBRIDGE MODE (Kiosk / Lanyard / Conf Call)
// ══════════════════════════════════════════════════════════════
// Hover sub-menu off the Live nav button. Both Kiosk and Lanyard are
// single-device, same-physical-space setups, so they keep
// interactionMode='in-person' under the hood -- the existing camera /
// streaming / inference pipeline is unchanged. Mode only flips a few UI
// defaults: hides the I Sign / I Speak header toggle (roles are fixed
// within each mode), sets the default camera source, updates the chip
// label on the Live nav button, and (for Lanyard) auto-opens the WiFi
// camera URL prompt on user-initiated entry. Conf Call is P5/deferred.

function setSignBridgeMode(mode, userInitiated) {
    if (mode !== 'kiosk' && mode !== 'lanyard') return;
    if (userInitiated === undefined) userInitiated = true;
    SIGNBRIDGE_MODE = mode;
    localStorage.setItem('signbridge.mode', mode);

    // Active highlight in the sub-menu.
    const btnK = document.getElementById('modeBtnKiosk');
    const btnL = document.getElementById('modeBtnLanyard');
    if (btnK) btnK.classList.toggle('active', mode === 'kiosk');
    if (btnL) btnL.classList.toggle('active', mode === 'lanyard');

    // Chip on the Live nav button so the active mode is visible without
    // hovering the menu.
    const chip = document.getElementById('navLiveModeChip');
    if (chip) chip.textContent = mode === 'lanyard' ? 'Lanyard' : 'Kiosk';

    // Both modes are single-device, same-space. Pin interactionMode so
    // existing code paths that branch on it (camera placement, transcript
    // layout) continue to take the in-person branch.
    interactionMode = 'in-person';

    // Pin METHOD = 'sign' in both modes. The signer's streaming pipeline
    // is the always-on side; the speaker's turn is on-demand via the
    // Tap-to-Reply button (avoids the continuous-listener-for-speaker
    // path that would fight Tap-to-Reply for the microphone).
    METHOD = 'sign';
    try { document.body.dataset.method = 'sign'; } catch (_) {}

    // Roles are fixed within each mode (signer in front, speaker behind
    // the iPad); the global I Sign / I Speak header toggle is meaningless.
    const headerToggle = document.getElementById('headerMethodToggle');
    if (headerToggle) headerToggle.style.display = 'none';

    // Per-mode camera default.
    const deviceRadio = document.querySelector('input[name="cameraSource"][value="device"]');
    const wifiRadio   = document.querySelector('input[name="cameraSource"][value="wifi"]');
    if (mode === 'kiosk') {
        if (deviceRadio) deviceRadio.checked = true;
        // Only flip the actual source if a WiFi cam isn't currently in use --
        // a user mid-session shouldn't have their feed yanked.
        if (!externalCameraUrl) selectCameraSource('device');
    } else if (mode === 'lanyard') {
        if (wifiRadio) wifiRadio.checked = true;
        selectCameraSource('wifi');
        // Auto-open the WiFi camera prompt only on user action, only if
        // no cam is connected yet. Don't badger on every page load.
        if (userInitiated && !externalCameraUrl) {
            const panel = document.getElementById('settingsPanel');
            if (panel) panel.style.display = 'block';
            const urlInput = document.getElementById('wifiCameraUrl');
            if (urlInput) setTimeout(() => { try { urlInput.focus(); } catch (_) {} }, 50);
        }
    }
}

// Apply the saved mode at page-load time -- no auto-prompt, no focus
// grab, just sync the UI state (chip label, active highlight, hide
// header toggle).
function applySignBridgeMode() {
    setSignBridgeMode(SIGNBRIDGE_MODE, false);
}

// ── Live nav sub-menu: JS-driven hover popover ──────────────────
// Sidebar has overflow-y:auto, which clips absolute-positioned children.
// Solution: append the sub-menu to document.body (so no clipping
// ancestor), keep it position:fixed, and reposition it on every show
// to track the nav button's bounding rect (handles window resize and
// sticky-scroll).
let _navSubmenuHideTimer = null;

function _positionNavLiveSubmenu() {
    const wrap = document.getElementById('navLive');
    const menu = document.getElementById('navLiveSubmenu');
    if (!wrap || !menu) return;
    const r = wrap.getBoundingClientRect();
    // 4px gap to the right of the nav button.
    menu.style.left = (r.right + 4) + 'px';
    menu.style.top  = r.top + 'px';
}

function showNavLiveSubmenu() {
    if (_navSubmenuHideTimer) {
        clearTimeout(_navSubmenuHideTimer);
        _navSubmenuHideTimer = null;
    }
    const menu = document.getElementById('navLiveSubmenu');
    if (!menu) return;
    _positionNavLiveSubmenu();
    menu.classList.add('is-open');
}

function hideNavLiveSubmenu() {
    // Slight delay so moving the mouse from the nav button across the
    // 4px gap into the menu doesn't dismiss it mid-transit.
    if (_navSubmenuHideTimer) clearTimeout(_navSubmenuHideTimer);
    _navSubmenuHideTimer = setTimeout(() => {
        const menu = document.getElementById('navLiveSubmenu');
        if (menu) menu.classList.remove('is-open');
    }, 200);
}

function _wireNavLiveSubmenu() {
    const wrap = document.getElementById('navLive') &&
                 document.getElementById('navLive').closest('.sidebar-item-wrap');
    const menu = document.getElementById('navLiveSubmenu');
    if (!wrap || !menu) return;
    // Move the menu out of the sidebar's overflow:auto so it can't be
    // clipped. Keep the reference fields by ID either way.
    if (menu.parentNode !== document.body) {
        document.body.appendChild(menu);
    }
    wrap.addEventListener('mouseenter', showNavLiveSubmenu);
    wrap.addEventListener('mouseleave', hideNavLiveSubmenu);
    menu.addEventListener('mouseenter', showNavLiveSubmenu);
    menu.addEventListener('mouseleave', hideNavLiveSubmenu);
    // Reposition on window resize and scroll so the menu stays anchored
    // to the (sticky) nav button.
    window.addEventListener('resize', () => {
        const m = document.getElementById('navLiveSubmenu');
        if (m && m.classList.contains('is-open')) _positionNavLiveSubmenu();
    });
}

// ══════════════════════════════════════════════════════════════
// TAP TO SPEAK (speaker turn in Kiosk / Lanyard modes)
// ══════════════════════════════════════════════════════════════
// Button-triggered continuous Web Speech API session. Tap to start,
// tap again to stop early, OR 5s of silence auto-finalizes. The
// signer's streaming pipeline is paused while listening so the
// speaker's mouth motion doesn't open phantom segments on the camera
// side. Result is appended to the transcript as a Speaker message.
//
// NOTE on wake-word approach (reverted): an always-on continuous
// SpeechRecognition session that starts WITHOUT a user gesture errors
// out with `network` instantly on Edge (and inconsistently on Chrome).
// The button-triggered start path works reliably because the click
// satisfies the user-activation requirement Chromium puts on the API.

const TAP_REPLY_SILENCE_MS = 5000;   // 5s silence -> auto-finalize
let _tapReplyRecognition = null;
let _tapReplyListening = false;
let _tapReplySilenceTimer = null;

function showTapReplyButton() {
    const inMode = (SIGNBRIDGE_MODE === 'kiosk' || SIGNBRIDGE_MODE === 'lanyard');
    const live   = (typeof liveActive !== 'undefined' && liveActive);
    // Tap to Speak is intentionally hidden in Kiosk/Lanyard now -- the
    // wake-word listener handles the speaker's turn (armed by Start /
    // Send / New Conversation clicks). Having both visible was confusing
    // the turn-taking flow. Code path stays alive in case we re-enable
    // it as a fallback later.
    const btn = document.getElementById('btnTapReply');
    if (btn) btn.style.display = 'none';
    _setTapReplyButtonIdle();
    // New Conversation button stays Kiosk/Lanyard-only.
    const btnNew = document.getElementById('btnNewConvo');
    if (btnNew) {
        btnNew.style.display = (inMode && live) ? 'inline-block' : 'none';
    }
}

function hideTapReplyButton() {
    const btn = document.getElementById('btnTapReply');
    if (btn) btn.style.display = 'none';
    const btnNew = document.getElementById('btnNewConvo');
    if (btnNew) btnNew.style.display = 'none';
    _stopTapReplyListening('hidden');
    stopWakeWordListener();
}

function _setTapReplyButtonIdle() {
    const btn = document.getElementById('btnTapReply');
    if (!btn) return;
    btn.innerHTML = '\u{1F399} Tap to Speak';
    btn.disabled = false;
}

function _setTapReplyButtonListening() {
    const btn = document.getElementById('btnTapReply');
    if (!btn) return;
    btn.innerHTML = '\u{1F399} Listening... (tap to stop)';
    btn.disabled = false;
}

function onTapReplyClick() {
    // Mid-listen tap = early stop; accumulated text gets emitted via onend.
    if (_tapReplyListening) {
        _stopTapReplyListening('user-stop');
        return;
    }
    _startTapReplyListening();
}

function _startTapReplyListening() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        showToast('Speech recognition not available in this browser', 'warn');
        return;
    }
    if (_tapReplyListening) return;

    // Pause the signing pipeline so the speaker's mouth/head motion
    // doesn't open phantom segments on the WS side.
    if (signSegmenter && typeof signSegmenter.pause === 'function') {
        try { signSegmenter.pause(); } catch (_) {}
    }
    // Cancel any in-progress TTS so it doesn't bleed into the mic.
    if ('speechSynthesis' in window) {
        try { speechSynthesis.cancel(); } catch (_) {}
    }

    _tapReplyRecognition = new SpeechRecognition();
    _tapReplyRecognition.lang = 'en-US';
    _tapReplyRecognition.interimResults = true;
    _tapReplyRecognition.continuous = true;
    _tapReplyRecognition.maxAlternatives = 1;

    let finalText = '';

    _tapReplyRecognition.onresult = (event) => {
        // Any result resets the silence timer so a mid-sentence pause
        // doesn't drop the second half.
        _armTapReplySilenceTimer();
        for (let i = event.resultIndex; i < event.results.length; i++) {
            if (event.results[i].isFinal) {
                finalText += event.results[i][0].transcript + ' ';
            }
        }
    };
    _tapReplyRecognition.onerror = (event) => {
        console.warn('[TapReply] recognition error:', event.error);
    };
    _tapReplyRecognition.onend = () => {
        _tapReplyListening = false;
        _clearTapReplySilenceTimer();
        _setTapReplyButtonIdle();
        if (signSegmenter && typeof signSegmenter.resume === 'function') {
            try { signSegmenter.resume(); } catch (_) {}
        }
        const text = (finalText || '').trim();
        if (!text) {
            showToast('Didn’t catch that — try again', 'info', 2200);
            return;
        }
        _appendSpeakerTurnToTranscript(text);
    };

    try {
        _tapReplyRecognition.start();
        _tapReplyListening = true;
        _setTapReplyButtonListening();
        _armTapReplySilenceTimer();
    } catch (e) {
        console.warn('[TapReply] start failed:', e);
        _tapReplyListening = false;
        _setTapReplyButtonIdle();
        if (signSegmenter && typeof signSegmenter.resume === 'function') {
            try { signSegmenter.resume(); } catch (_) {}
        }
    }
}

function _armTapReplySilenceTimer() {
    _clearTapReplySilenceTimer();
    _tapReplySilenceTimer = setTimeout(() => {
        if (_tapReplyListening && _tapReplyRecognition) {
            try { _tapReplyRecognition.stop(); } catch (_) {}
        }
    }, TAP_REPLY_SILENCE_MS);
}

function _clearTapReplySilenceTimer() {
    if (_tapReplySilenceTimer) {
        clearTimeout(_tapReplySilenceTimer);
        _tapReplySilenceTimer = null;
    }
}

function _stopTapReplyListening(reason) {
    _clearTapReplySilenceTimer();
    if (_tapReplyRecognition && _tapReplyListening) {
        // user-stop -> stop() (emits accumulated text via onend).
        // hidden / restart -> abort() (drops pending result; we don't
        // want a half-finished reply landing in the next turn).
        try {
            if (reason === 'user-stop') _tapReplyRecognition.stop();
            else _tapReplyRecognition.abort();
        } catch (_) {}
    }
    if (reason === 'hidden' || reason === 'restart') {
        _tapReplyListening = false;
        _tapReplyRecognition = null;
        _setTapReplyButtonIdle();
        if (signSegmenter && typeof signSegmenter.resume === 'function') {
            try { signSegmenter.resume(); } catch (_) {}
        }
    }
}

function _appendSpeakerTurnToTranscript(text) {
    try {
        addMsg('doctor', SPEAKER_LABEL, text);
    } catch (e) {
        console.warn('[TapReply] addMsg failed:', e);
    }
    try {
        history.push({ speaker: 'doctor', text });
    } catch (_) {}
}

// ══════════════════════════════════════════════════════════════
// WAKE WORD LISTENER (started by New Conversation click)
// ══════════════════════════════════════════════════════════════
// Continuous SpeechRecognition that scans for a configurable wake
// word (default "Signer"). Triggered by the New Conversation button's
// click handler, NOT auto-started, because SpeechRecognition.start()
// without a user gesture errors with `network` on Edge / modern Chrome.
// After the gesture-blessed initial start, the listener auto-recycles
// every 25s to dodge Chromium's ~60s continuous-session failure mode.

const WAKE_WORD_RECYCLE_MS = 25000;
const WAKE_WORD_SILENCE_MS = 5000;
// Short post-speech delay before flipping the status bar from
// "Listening to speaker..." to "Processing speech...". Gives the
// signer a clear visual cue that the speaker has paused and STT is
// finalizing, instead of staring at an unchanged screen.
const WAKE_WORD_PROCESSING_DELAY_MS = 1500;
let _wakeRecognition = null;
let _wakeListening   = false;
let _wakeCapturing   = false;        // wake word heard, accumulating reply
let _wakeFinalText   = '';
let _wakeLatestInterim = '';         // most recent interim transcript (fallback)
let _wakeSilenceTimer = null;
let _wakeRestartTimer = null;
let _wakeRecycleTimer = null;
let _wakeProcessingTimer = null;

function _getWakeWord() {
    let w = (localStorage.getItem('signbridge.wakeWord') || 'Signer').trim();
    w = w.split(/\s+/)[0] || 'Signer';
    return w;
}

function updateWakeWord() {
    const el = document.getElementById('wakeWordInput');
    if (!el) return;
    let val = (el.value || '').trim().split(/\s+/)[0] || 'Signer';
    if (val.length > 20) val = val.slice(0, 20);
    el.value = val;
    localStorage.setItem('signbridge.wakeWord', val);
    const mirror = document.getElementById('setWakeWord');
    if (mirror) mirror.value = val;
}

function syncWakeWordUI() {
    const el = document.getElementById('wakeWordInput');
    if (el) el.value = _getWakeWord();
}

function _wakeWordRegex() {
    const w = _getWakeWord();
    const esc = w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    return new RegExp('\\b' + esc + '\\b[,.!?:;]*\\s*', 'i');
}

// Critical: must be invoked SYNCHRONOUSLY from a user-gesture click
// handler. Calling .start() outside one will silently fail on Edge.
function startWakeWordListener() {
    const inMode = (SIGNBRIDGE_MODE === 'kiosk' || SIGNBRIDGE_MODE === 'lanyard');
    if (!inMode) return;
    if (typeof liveActive === 'undefined' || !liveActive) return;
    if (_wakeListening) return;

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        console.warn('[WakeWord] SpeechRecognition not available in this browser');
        return;
    }

    // Capture the instance in a closure so onend / onerror can tell
    // whether THIS instance is still the "current" one. Without this,
    // an old instance's late-firing onend (triggered by abort() during
    // a stop-then-start race) would set _wakeListening = false on the
    // NEW listener that's still alive -- causing wake-word to silently
    // stop working after the first Send.
    const thisRec = new SpeechRecognition();
    thisRec.lang = 'en-US';
    thisRec.interimResults = true;
    thisRec.continuous = true;
    thisRec.maxAlternatives = 1;
    thisRec.onresult = _onWakeResult;
    thisRec.onerror = (event) => {
        if (thisRec !== _wakeRecognition) return;   // stale instance
        console.warn('[WakeWord] error:', event.error);
        if (event.error === 'network') {
            try { thisRec.abort(); } catch (_) {}
        }
    };
    thisRec.onend = () => {
        if (thisRec !== _wakeRecognition) {
            console.log('[WakeWord] (stale instance onend ignored)');
            return;
        }
        _wakeListening = false;
        _wakeRecognition = null;
        console.log('[WakeWord] listener ended');
        // Auto-restart on natural end (Chrome closes the SR session
        // after extended silence, ~5-10s on typical settings).
        // Without this, the listener silently dies and the user has
        // to click Send / New Conversation to revive it -- exactly
        // the "worked once then stopped" symptom.
        // Use _armWakeWordDeferred (300ms setTimeout) -- a stale
        // abort cascade is fine because we null-checked the instance
        // above, and Chrome's user-activation seems to persist long
        // enough through setTimeout for the new start() to succeed.
        if ((SIGNBRIDGE_MODE === 'kiosk' || SIGNBRIDGE_MODE === 'lanyard')
            && typeof liveActive !== 'undefined' && liveActive) {
            _armWakeWordDeferred('natural-end');
        }
    };

    _wakeRecognition = thisRec;
    try {
        thisRec.start();
        _wakeListening = true;
        console.log('[WakeWord] listener started; wake word =', _getWakeWord());
    } catch (e) {
        console.warn('[WakeWord] start failed:', e);
        _wakeListening = false;
        if (_wakeRecognition === thisRec) _wakeRecognition = null;
    }
}

// Schedule a wake-word start() with a small delay. Chrome reliably
// silent-fails a fresh SR session if started synchronously right after
// aborting/ending the previous one (new session reports started but
// receives no audio). A 300ms gap gives the browser enough time to
// release the audio resource before the new session grabs it. Use
// this everywhere we re-arm; only the very first initial start in
// btnStart's click handler is synchronous (no prior session).
function _armWakeWordDeferred(why) {
    if (_wakeRestartTimer) clearTimeout(_wakeRestartTimer);
    _wakeRestartTimer = setTimeout(() => {
        _wakeRestartTimer = null;
        console.log('[WakeWord] deferred re-arm fires (' + (why || 'manual') + ')');
        startWakeWordListener();
    }, 300);
}

function stopWakeWordListener() {
    // Cancel any pending deferred re-arm first. If we don't, a
    // previously-scheduled _armWakeWordDeferred would still fire
    // after we've explicitly stopped (e.g., mode switch, end live).
    if (_wakeRestartTimer) { clearTimeout(_wakeRestartTimer); _wakeRestartTimer = null; }
    _clearWakeSilenceTimer();
    // Only abort if we have a recognition AND it's still listening.
    // After onend, _wakeRecognition is nulled out (so this won't fire).
    // Calling .abort() on a recently-ended instance appears to confuse
    // Chrome into silently failing the very next .start().
    if (_wakeRecognition && _wakeListening) {
        try { _wakeRecognition.abort(); } catch (_) {}
    }
    const wasCapturing = _wakeCapturing;
    _wakeRecognition = null;
    _wakeListening = false;
    _wakeCapturing = false;
    _wakeFinalText = '';
    _wakeLatestInterim = '';
    // If we were mid-capture, re-enable the buttons + clear the
    // "Listening / Processing" status so the signer isn't stuck with a
    // disabled UI after a New Conversation / mode change interruption.
    if (wasCapturing) _exitSpeakerTurnUI();
}

// ── Speaker-turn UI helpers ────────────────────────────────────
// During the wake-word capture window the signer can't usefully click
// Start Conversation / Restart / Resume Signing / New Conversation
// without interrupting the speaker. Disable them, and show a clear
// status badge so the signer sees something is happening.
function _showSpeakerOverlay(text, processing) {
    const el  = document.getElementById('speakerTurnOverlay');
    const txt = document.getElementById('speakerTurnText');
    const ico = document.getElementById('speakerTurnIcon');
    if (!el || !txt) return;
    txt.textContent = text;
    if (ico) ico.textContent = processing ? '✨' : '\u{1F399}\u{FE0F}';
    el.classList.toggle('processing', !!processing);
    el.style.display = 'inline-flex';
}

function _hideSpeakerOverlay() {
    const el = document.getElementById('speakerTurnOverlay');
    if (el) el.style.display = 'none';
}

function _enterSpeakerTurnUI() {
    console.log('[SpeakerUI] enter -> Listening to speaker');
    const btnStart = document.getElementById('btnStart');
    const btnSend  = document.getElementById('btnSendMessage');
    const btnNew   = document.getElementById('btnNewConvo');
    if (btnStart) btnStart.disabled = true;
    if (btnSend)  btnSend.disabled  = true;
    if (btnNew)   btnNew.disabled   = true;
    if (typeof statusBar !== 'undefined' && statusBar) {
        statusBar.innerHTML = '<span class="processing">\u{1F399}\u{FE0F} Listening to speaker…</span>';
    }
    _showSpeakerOverlay('Listening to speaker…', false);
}

function _setSpeakerProcessingUI() {
    console.log('[SpeakerUI] -> Processing speech');
    if (typeof statusBar !== 'undefined' && statusBar) {
        statusBar.innerHTML = '<span class="processing ai"><span class="ai-sparkle">✨</span>Processing speech… <span class="spinner"></span></span>';
    }
    _showSpeakerOverlay('Processing speech…', true);
}

function _exitSpeakerTurnUI() {
    _hideSpeakerOverlay();
    const btnStart = document.getElementById('btnStart');
    const btnSend  = document.getElementById('btnSendMessage');
    const btnNew   = document.getElementById('btnNewConvo');
    if (btnStart) btnStart.disabled = false;
    if (btnSend)  btnSend.disabled  = false;
    if (btnNew)   btnNew.disabled   = false;
    // Status bar restoration: if we're still in the post-Send "Resume
    // Signing" state, prefer that prompt; otherwise revert to the
    // idle "Sign when ready" prompt.
    const sendBtn = document.getElementById('btnSendMessage');
    if (sendBtn && sendBtn.textContent === 'Resume Signing'
        && typeof statusBar !== 'undefined' && statusBar) {
        statusBar.innerHTML = '<span class="prompt">Speaker replied. Click <b>Resume Signing</b> to continue.</span>';
    } else if (typeof statusBar !== 'undefined' && statusBar) {
        statusBar.innerHTML = '<span class="processing">Sign when ready — your signs will appear here</span>';
    }
}

function _onWakeResult(event) {
    // Track the most recent interim transcript so we always have the
    // freshest "in-flight" reply text. Prior versions only accumulated
    // FINAL transcripts after the wake word fired, which lost any words
    // that arrived in an interim ("Signer hello" came as interim ->
    // wake-word triggered on "Signer" -> "hello" was never captured
    // because final never came before listener ended).
    let latestInterim = null;
    for (let i = event.resultIndex; i < event.results.length; i++) {
        const res = event.results[i];
        const text = res[0].transcript;
        const isFinal = res.isFinal;

        if (!_wakeCapturing) {
            const re = _wakeWordRegex();
            const match = re.exec(text);
            if (match) {
                _wakeCapturing = true;
                _wakeFinalText = text.slice(match.index + match[0].length).trim();
                _armWakeSilenceTimer();
                if ('speechSynthesis' in window) {
                    try { speechSynthesis.cancel(); } catch (_) {}
                }
                _enterSpeakerTurnUI();
                console.log('[WakeWord] HEARD; initial text after wake:', JSON.stringify(_wakeFinalText));
            }
        } else if (isFinal) {
            _wakeFinalText += ' ' + text;
            _armWakeSilenceTimer();
        } else {
            // Interim while capturing -- remember the latest so
            // _finalizeWakeCapture has something to fall back to if no
            // final ever arrives before the silence timer fires.
            latestInterim = text;
            _armWakeSilenceTimer();
        }
    }
    // Stash the latest interim as a fallback for finalize. Don't append
    // to _wakeFinalText directly (interims supersede prior interims, and
    // appending would create duplicates as the same phrase repeats).
    if (_wakeCapturing && latestInterim != null) {
        _wakeLatestInterim = latestInterim;
    }
    if (_wakeCapturing) _armWakeSilenceTimer();
}

function _armWakeSilenceTimer() {
    // Two timers run in tandem while capturing:
    //   _wakeProcessingTimer (1.5s) -> flip status to "Processing speech..."
    //   _wakeSilenceTimer    (5.0s) -> finalize + append to transcript
    // Both reset on every new result, so they only fire when the
    // speaker truly stops talking.
    _clearWakeSilenceTimer();
    _wakeSilenceTimer = setTimeout(_finalizeWakeCapture, WAKE_WORD_SILENCE_MS);
    if (_wakeProcessingTimer) clearTimeout(_wakeProcessingTimer);
    _wakeProcessingTimer = setTimeout(() => {
        _wakeProcessingTimer = null;
        if (_wakeCapturing) _setSpeakerProcessingUI();
    }, WAKE_WORD_PROCESSING_DELAY_MS);
}

function _clearWakeSilenceTimer() {
    if (_wakeSilenceTimer) {
        clearTimeout(_wakeSilenceTimer);
        _wakeSilenceTimer = null;
    }
    if (_wakeProcessingTimer) {
        clearTimeout(_wakeProcessingTimer);
        _wakeProcessingTimer = null;
    }
}

function _finalizeWakeCapture() {
    _clearWakeSilenceTimer();
    const wasCapturing = _wakeCapturing;
    if (!wasCapturing) return;
    // Prefer accumulated final text; fall back to the latest interim
    // if no final ever arrived (Chrome sometimes ends the session
    // before flushing a final transcript). Strip the wake word from
    // either source, because interim sometimes includes it.
    let text = (_wakeFinalText || '').trim();
    if (!text && _wakeLatestInterim) text = _wakeLatestInterim.trim();
    _wakeCapturing = false;
    _wakeFinalText = '';
    _wakeLatestInterim = '';
    console.log('[WakeWord] finalize; raw text =', JSON.stringify(text));
    if (text) {
        const re = _wakeWordRegex();
        let clean = text;
        const m = re.exec(clean);
        if (m) clean = clean.slice(m.index + m[0].length);
        clean = clean.trim();
        if (clean) {
            console.log('[WakeWord] appending to transcript:', JSON.stringify(clean));
            _appendSpeakerTurnToTranscript(clean);
        } else {
            console.log('[WakeWord] post-strip text empty -- nothing to append');
        }
    } else {
        console.log('[WakeWord] no text captured -- nothing to append');
    }
    _exitSpeakerTurnUI();
}

// ══════════════════════════════════════════════════════════════
// NEW CONVERSATION button (Kiosk / Lanyard)
// ══════════════════════════════════════════════════════════════
// Full transcript wipe + fresh segmenter state + re-arms a single-shot
// wake-word session in this click's gesture context. Distinct from
// Restart, which only soft-resets the current in-flight utterance.
function onNewConversationClick() {
    const msgs = document.getElementById('transcriptMsgs');
    if (msgs) msgs.innerHTML = '';
    history = [];
    try { btnRestart.click(); } catch (_) {}
    // Use deferred re-arm for the same reason as sendNowAndPause:
    // synchronous restart-after-abort silently fails on Chrome.
    stopWakeWordListener();
    _armWakeWordDeferred('new-conversation');
    showToast('New conversation — say "' + _getWakeWord() + '" to reply', 'info', 3000);
}

// ── Text-to-speech enabled toggle (per-device settings) ─────────
// Off by default in both modes (staff reads from screen). Opt-in via
// the settings popup when an audible reply makes sense.
function updateTtsEnabled() {
    const cb = document.getElementById('ttsEnabledToggle');
    if (!cb) return;
    TTS_ENABLED = !!cb.checked;
    localStorage.setItem('signbridge.ttsEnabled', TTS_ENABLED ? '1' : '0');
    // Mirror to Settings page if its DOM is built.
    const mirror = document.getElementById('setTtsEnabled');
    if (mirror) mirror.checked = TTS_ENABLED;
}

function syncTtsToggleUI() {
    const cb = document.getElementById('ttsEnabledToggle');
    if (cb) cb.checked = TTS_ENABLED;
}

// ══════════════════════════════════════════════════════════════
// SETTINGS PAGE -- edit/save workflow + bidirectional sync with cog
// ══════════════════════════════════════════════════════════════
// Inputs are disabled by default. Click Edit -> capture current
// values as a snapshot and enable inputs. On Save, persist the
// edited values to localStorage + globals + sync the cog popup. On
// Cancel, restore the snapshot. The cog popup keeps its own
// auto-persist-on-change behavior (since it's used mid-session and
// changes should apply NOW) -- the two stay in sync because both
// read/write the same localStorage keys.

// _settingsEditing / _settingsSnapshot are hoisted to top-of-file
// globals (TDZ avoidance, since syncSettingsPageUI is called from the
// page-load init block before this line in source order would run).

// ── Editable-input IDs (collected for enable/disable + snapshot) ──
// _SETTINGS_INPUTS hoisted to top-of-file globals.

function _settingsInputValue(el) {
    if (!el) return null;
    return el.type === 'checkbox' ? !!el.checked : el.value;
}
function _settingsSetInputValue(el, val) {
    if (!el) return;
    if (el.type === 'checkbox') el.checked = !!val;
    else el.value = val == null ? '' : val;
}

function _settingsToggleInputs(enabled) {
    _SETTINGS_INPUTS.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.disabled = !enabled;
    });
    const page = document.querySelector('.settings-page');
    if (page) page.classList.toggle('editing', enabled);
    const editBtn   = document.getElementById('settingsEditBtn');
    const saveBtn   = document.getElementById('settingsSaveBtn');
    const cancelBtn = document.getElementById('settingsCancelBtn');
    if (editBtn)   editBtn.style.display   = enabled ? 'none' : 'inline-block';
    if (saveBtn)   saveBtn.style.display   = enabled ? 'inline-block' : 'none';
    if (cancelBtn) cancelBtn.style.display = enabled ? 'inline-block' : 'none';
}

function onSettingsEditClick() {
    if (_settingsEditing) return;
    // Snapshot current values so Cancel can restore.
    _settingsSnapshot = {};
    _SETTINGS_INPUTS.forEach(id => {
        _settingsSnapshot[id] = _settingsInputValue(document.getElementById(id));
    });
    _settingsEditing = true;
    _settingsToggleInputs(true);
    // Focus the first editable input for immediate keyboard editing.
    const firstEl = document.getElementById(_SETTINGS_INPUTS[0]);
    if (firstEl) try { firstEl.focus(); } catch (_) {}
}

function onSettingsCancelClick() {
    if (!_settingsEditing) return;
    if (_settingsSnapshot) {
        _SETTINGS_INPUTS.forEach(id => {
            _settingsSetInputValue(document.getElementById(id), _settingsSnapshot[id]);
        });
    }
    _settingsSnapshot = null;
    _settingsEditing = false;
    _settingsToggleInputs(false);
}

function onSettingsSaveClick() {
    if (!_settingsEditing) return;

    // --- Display name (Profile section) ---
    const nameEl = document.getElementById('setDisplayName');
    if (nameEl) {
        const profile = (typeof loadProfile === 'function') ? loadProfile() : {};
        const name = (nameEl.value || '').trim().slice(0, 30);
        profile.name = name;
        if (typeof saveProfile === 'function') saveProfile(profile);
        // Update the welcome message in the header.
        const welcomeEl = document.getElementById('heroWelcome');
        if (welcomeEl) {
            if (name) {
                welcomeEl.textContent = `Welcome back, ${name}`;
                welcomeEl.style.display = 'inline';
            } else {
                welcomeEl.style.display = 'none';
            }
        }
    }

    // --- TTS (Audio) ---
    const ttsCb = document.getElementById('setTtsEnabled');
    if (ttsCb) {
        TTS_ENABLED = !!ttsCb.checked;
        localStorage.setItem('signbridge.ttsEnabled', TTS_ENABLED ? '1' : '0');
        syncTtsToggleUI();
    }

    // --- Chime (Audio) ---
    const chimeCb = document.getElementById('setChimeEnabled');
    if (chimeCb) {
        CHIME_ON_LOW_CTQI = !!chimeCb.checked;
        localStorage.setItem('signbridge.lowCtqiChime', CHIME_ON_LOW_CTQI ? '1' : '0');
        syncLowCtqiChimeUI();
    }

    // --- Wake word (Audio) ---
    const wakeEl = document.getElementById('setWakeWord');
    if (wakeEl) {
        let val = (wakeEl.value || '').trim().split(/\s+/)[0] || 'Signer';
        if (val.length > 20) val = val.slice(0, 20);
        wakeEl.value = val;
        localStorage.setItem('signbridge.wakeWord', val);
        syncWakeWordUI();
    }

    // --- CTQI alert threshold (Advanced) ---
    const ctqiEl = document.getElementById('setCtqiThreshold');
    if (ctqiEl) {
        let v = parseInt(ctqiEl.value, 10);
        if (!isFinite(v)) v = 80;
        v = Math.max(0, Math.min(100, v));
        ctqiEl.value = v;
        CTQI_LOW_THRESHOLD = v;
        localStorage.setItem('signbridge.ctqiThreshold', String(v));
    }

    // --- CTQI hard floor (Advanced) ---
    const ctqiFloorEl = document.getElementById('setCtqiHardFloor');
    if (ctqiFloorEl) {
        let v = parseInt(ctqiFloorEl.value, 10);
        if (!isFinite(v)) v = 40;
        v = Math.max(0, Math.min(100, v));
        // Sanity: hard floor must be < alert threshold; if user set
        // floor above alert, clamp to alert - 1.
        if (v >= CTQI_LOW_THRESHOLD) v = Math.max(0, CTQI_LOW_THRESHOLD - 1);
        ctqiFloorEl.value = v;
        CTQI_HARD_FLOOR = v;
        localStorage.setItem('signbridge.ctqiHardFloor', String(v));
    }

    _settingsSnapshot = null;
    _settingsEditing = false;
    _settingsToggleInputs(false);
    if (typeof showToast === 'function') showToast('Settings saved', 'info', 1800);
}

function onSettingsClearData() {
    if (typeof clearProfileData === 'function') clearProfileData();
    syncSettingsPageUI();
}

// Called when the Settings tab is opened (and once on page load).
// Populates every input from current state. Also re-renders read-only
// sections (practice stats, history summary).
function syncSettingsPageUI() {
    // Profile section
    const profile = (typeof loadProfile === 'function') ? loadProfile() : {};
    const nameEl = document.getElementById('setDisplayName');
    if (nameEl) nameEl.value = profile.name || '';

    // Audio section
    const ttsCb   = document.getElementById('setTtsEnabled');
    if (ttsCb) ttsCb.checked = TTS_ENABLED;
    const chimeCb = document.getElementById('setChimeEnabled');
    if (chimeCb) chimeCb.checked = CHIME_ON_LOW_CTQI;
    const wakeEl  = document.getElementById('setWakeWord');
    if (wakeEl) wakeEl.value = _getWakeWord();

    // Advanced section
    const ctqiEl  = document.getElementById('setCtqiThreshold');
    if (ctqiEl) ctqiEl.value = CTQI_LOW_THRESHOLD;
    const ctqiFloorEl = document.getElementById('setCtqiHardFloor');
    if (ctqiFloorEl) ctqiFloorEl.value = CTQI_HARD_FLOOR;

    // Stats section (read-only -- just render current totals).
    const stats = (profile && profile.practiceStats) || {};
    let totalAttempts = 0, totalCorrect = 0;
    for (const g in stats) {
        totalAttempts += stats[g].attempts || 0;
        totalCorrect  += stats[g].correct || 0;
    }
    const aEl = document.getElementById('setStatAttempts');
    const cEl = document.getElementById('setStatCorrect');
    const accEl = document.getElementById('setStatAccuracy');
    if (aEl) aEl.textContent = totalAttempts;
    if (cEl) cEl.textContent = totalCorrect;
    if (accEl) accEl.textContent = totalAttempts > 0
        ? Math.round(totalCorrect / totalAttempts * 100) + '%' : '—';

    // Data section: history summary + link visibility.
    if (typeof renderProfileHistory === 'function') {
        // Reuse the existing function but redirect its DOM writes to
        // the Settings-page IDs we exposed.
        try {
            const hist = (profile && profile.conversations) || [];
            const sumEl = document.getElementById('setHistorySummary');
            const linkEl = document.getElementById('setHistoryLink');
            if (sumEl) sumEl.textContent = hist.length
                ? `${hist.length} conversation${hist.length === 1 ? '' : 's'} saved`
                : 'No conversations saved yet';
            if (linkEl) linkEl.style.display = hist.length ? 'inline-block' : 'none';
        } catch (_) {}
    }

    // If we were in edit mode and the page just re-synced, keep inputs
    // disabled until the user re-clicks Edit.
    if (!_settingsEditing) _settingsToggleInputs(false);
}

let headerTogglesLocked = false;
let transcriptCollapsed = false;

function toggleTranscript() {
    const panel = document.getElementById('transcriptSide');
    const showBtn = document.getElementById('btnShowTranscript');
    transcriptCollapsed = !transcriptCollapsed;

    if (transcriptCollapsed) {
        panel.classList.add('collapsed');
        showBtn.style.display = 'flex';
    } else {
        panel.classList.remove('collapsed');
        showBtn.style.display = 'none';
    }
}

function setMethod(method) {
    if (headerTogglesLocked) return;
    METHOD = method;
    document.getElementById('htogSign').classList.toggle('active', method === 'sign');
    document.getElementById('htogSpeak').classList.toggle('active', method === 'speak');
}

function lockHeaderToggles() {
    headerTogglesLocked = true;
    document.querySelectorAll('.htog-btn').forEach(b => b.classList.add('locked'));
}

function unlockHeaderToggles() {
    headerTogglesLocked = false;
    document.querySelectorAll('.htog-btn').forEach(b => b.classList.remove('locked'));
}

// ══════════════════════════════════════════════════════════════
// SCENARIO SELECTION
// ══════════════════════════════════════════════════════════════
function hideHomeContent() {
    ['heroSection', 'demoIntro', 'demoScenarios', 'liveScenarios', 'signBankScreen', 'uploadScreen', 'historyScreen'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.style.display = 'none';
    });
    const hs = document.querySelector('.home-screen');
    if (hs) hs.style.display = 'none';
}

function selectScenario(domain, scenarioName) {
    selectedDomain = domain;
    selectedScenario = scenarioName;
    if (MODE === 'live') recordRecentScenario(domain, scenarioName);
    hideHomeContent();
    document.getElementById('phaseConvo').style.display = 'flex';

    const modeLabel = interactionMode === 'in-person' ? 'In-Person' : 'Video Call';
    document.getElementById('scenarioLabel').textContent = scenarioName + ' \u2022 ' + modeLabel;

    lockHeaderToggles();
    pushRoute('conversation/' + domain);

    console.log('[Scenario] Selected:', scenarioName, '| Domain:', domain, '| Mode:', interactionMode, '| Method:', METHOD, '| AppMode:', MODE);

    loadVocabulary(domain);

    // Model warmup. Block Start Conversation until the OpenHands model +
    // MediaPipe Holistic are loaded server-side -- otherwise the user can
    // click Start before the inference path is ready, hit Send, and stall
    // on a 5-10s cold start. While warming, the button shows "Loading model".
    if (MODE === 'live' && METHOD === 'sign') {
        const origLabel = btnStart.dataset.armed === '1' ? 'Restart' : 'Start Conversation';
        btnStart.disabled = true;
        btnStart.textContent = 'Loading model...';
        fetch('/api/warm-model', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ domain }),
        }).then(r => r.json()).then(j => {
            console.log('[Warmup]', j.success ? `model for "${j.domain}" ready` : 'failed: ' + j.error);
        }).catch(e => console.warn('[Warmup] fetch failed:', e.message)).finally(() => {
            // Re-enable -- only if the user is still on this scenario and
            // hasn't already navigated away or clicked into the conversation.
            if (selectedDomain === domain) {
                btnStart.disabled = false;
                btnStart.textContent = origLabel;
            }
        });
    }
}

// ══════════════════════════════════════════════════════════════
// VOCABULARY PANEL
// ══════════════════════════════════════════════════════════════
const WORD_CATEGORIES = {
    medical: ['allergy','blood','breathe','cough','doctor','headache','medicine','sick','stomach','surgery','temperature'],
    question: ['how','what','where','name','answer'],
    action: ['can','eat','drink','feel','give','hear','help','know','play','sit','stand','stop','tell','wait','walk','want','dance','study','change','enjoy','breathe'],
    time: ['before','morning','night','today','time','year','thursday','birthday','thanksgiving','later','appointment'],
    descriptor: ['better','fine','full','hot','more','right','tall','wrong','worse','deaf','stress'],
    general: []
};

function categorizeWord(word) {
    for (const [cat, words] of Object.entries(WORD_CATEGORIES)) {
        if (cat === 'general') continue;
        if (words.includes(word)) return cat;
    }
    return 'general';
}

let vocabCache = {};

async function loadVocabulary(domain) {
    const panel = document.getElementById('vocabPanel');
    const chipsEl = document.getElementById('vocabChips');
    const countEl = document.getElementById('vocabCount');
    const toggleBtn = document.getElementById('vocabToggleBtn');

    chipsEl.innerHTML = '';
    chipsEl.classList.remove('open');
    toggleBtn.classList.remove('open');
    panel.style.display = 'none';

    if (vocabCache[domain]) {
        renderVocabulary(vocabCache[domain]);
        return;
    }

    try {
        const resp = await fetch('/api/vocabulary/' + domain);
        if (!resp.ok) {
            showToast('Could not load vocabulary for this domain', 'warn');
            return;
        }
        const data = await resp.json();
        vocabCache[domain] = data.glosses;
        renderVocabulary(data.glosses);
    } catch (e) {
        showToast('Failed to load vocabulary', 'error');
        console.warn('[Vocab] Failed to load:', e);
    }
}

function renderVocabulary(glosses) {
    const panel = document.getElementById('vocabPanel');
    const chipsEl = document.getElementById('vocabChips');
    const countEl = document.getElementById('vocabCount');

    countEl.textContent = glosses.length;
    chipsEl.innerHTML = '';

    const grouped = { medical: [], question: [], action: [], time: [], descriptor: [], general: [] };
    for (const word of glosses) {
        const cat = categorizeWord(word);
        grouped[cat].push(word);
    }

    for (const [cat, words] of Object.entries(grouped)) {
        for (const word of words) {
            const chip = document.createElement('span');
            chip.className = 'vocab-chip cat-' + cat;
            chip.textContent = word;
            chipsEl.appendChild(chip);
        }
    }

    panel.style.display = 'block';
    // Start collapsed — user can expand if needed
    chipsEl.classList.remove('open');
    document.getElementById('vocabToggleBtn').classList.remove('open');
}

function toggleVocabPanel() {
    const chipsEl = document.getElementById('vocabChips');
    const toggleBtn = document.getElementById('vocabToggleBtn');
    const isOpen = chipsEl.classList.contains('open');
    chipsEl.classList.toggle('open', !isOpen);
    toggleBtn.classList.toggle('open', !isOpen);
}

// ══════════════════════════════════════════════════════════════
// SIGN BANK (2.1) — Training mode reference library
// ══════════════════════════════════════════════════════════════
let signBankData = null; // cached sign bank data
let signBankLoaded = false;

function getSbDomainMeta(domainKey) {
    // Pull icon from registry if available, fall back to defaults
    const entry = registryData && registryData.domains && registryData.domains[domainKey];
    const icon = entry ? entry.icon : (domainKey === 'generic' ? 'G' : '?');
    return {
        icon,
        color: 'var(--accent)',
        bg: 'rgba(196,135,42,0.1)',
    };
}

async function loadSignBank() {
    if (signBankLoaded) return;
    try {
        const resp = await fetch('/api/sign-bank');
        if (!resp.ok) throw new Error('Failed to load');
        const data = await resp.json();
        signBankData = data;
        signBankLoaded = true;
        renderSignBank(data);
    } catch (e) {
        showToast('Could not load Sign Bank', 'error');
        console.error('[SignBank] Load error:', e);
    }
}

function renderSignBank(data) {
    const container = document.getElementById('sbGrid');
    const totalEl = document.getElementById('sbTotalCount');
    const visibleEl = document.getElementById('sbVisibleCount');
    const emptyEl = document.getElementById('sbEmpty');

    totalEl.textContent = data.count;
    container.innerHTML = '';

    if (data.count === 0) {
        emptyEl.style.display = 'block';
        visibleEl.textContent = '';
        return;
    }

    emptyEl.style.display = 'none';
    visibleEl.textContent = data.count + ' shown';

    data.groups.forEach(group => {
        const meta = getSbDomainMeta(group.domain);

        const section = document.createElement('div');
        section.className = 'sb-group';
        section.dataset.domain = group.domain;

        const header = document.createElement('div');
        header.className = 'sb-group-header';
        header.innerHTML = `
            <span class="sb-group-icon" style="background:${meta.bg};color:${meta.color}">${meta.icon}</span>
            <span class="sb-group-label">${group.label}</span>
            <span class="sb-group-count">${group.signs.length} sign${group.signs.length !== 1 ? 's' : ''}</span>
        `;
        section.appendChild(header);

        const grid = document.createElement('div');
        grid.className = 'sb-group-grid';

        // Cap preview cards per group at 4. Loading 2000 thumbnails
        // froze the page; the rest are accessible via the picker
        // below. Each card here still does metadata preload, but
        // 4 * (number of domains) is bounded.
        const PREVIEW_LIMIT = 4;
        const previewSigns = group.signs.slice(0, PREVIEW_LIMIT);
        const restSigns    = group.signs.slice(PREVIEW_LIMIT);

        previewSigns.forEach(sign => {
            const card = document.createElement('div');
            card.className = 'sb-card';
            card.dataset.gloss = sign.gloss.toLowerCase();
            card.innerHTML = `
                <div class="sb-card-video">
                    <video src="${sign.video_url}" muted playsinline loop preload="metadata"></video>
                    <div class="sb-play-hint">\u25B6</div>
                </div>
                <div class="sb-card-label">
                    <span class="sb-card-gloss">${sign.gloss.toUpperCase()}</span>
                </div>
                <div class="sb-card-actions">
                    <button class="sb-practice-btn" title="Practice this sign">&#x270B; Practice</button>
                </div>
            `;

            const video = card.querySelector('video');
            const videoBox = card.querySelector('.sb-card-video');
            videoBox.addEventListener('mouseenter', () => { video.play().catch(() => {}); });
            videoBox.addEventListener('mouseleave', () => { video.pause(); video.currentTime = 0; });
            videoBox.addEventListener('click', () => {
                if (video.controls) {
                    video.controls = false;
                    video.muted = true;
                } else {
                    video.controls = true;
                    video.muted = false;
                    video.currentTime = 0;
                    video.play().catch(() => {});
                }
            });

            card.querySelector('.sb-practice-btn').addEventListener('click', (e) => {
                e.stopPropagation();
                startPracticeSession({ ...sign, domain: group.domain });
            });

            grid.appendChild(card);
        });

        section.appendChild(grid);

        // Picker for "the rest" -- any sign in this domain beyond the
        // 4 previewed. Selecting an entry jumps straight to practice
        // for that gloss (no thumbnail load for the whole list).
        if (restSigns.length > 0) {
            const pickerRow = document.createElement('div');
            pickerRow.className = 'sb-group-picker-row';
            const picker = document.createElement('select');
            picker.className = 'sb-group-picker';
            const placeholder = document.createElement('option');
            placeholder.value = '';
            placeholder.textContent = `Practice another ${group.label} sign\u2026 (${restSigns.length} more)`;
            placeholder.disabled = true;
            placeholder.selected = true;
            picker.appendChild(placeholder);
            restSigns.forEach(sign => {
                const opt = document.createElement('option');
                opt.value = sign.gloss.toLowerCase();
                opt.textContent = sign.gloss;
                picker.appendChild(opt);
            });
            picker.addEventListener('change', () => {
                const sel = picker.value;
                if (!sel) return;
                const sign = group.signs.find(s => s.gloss.toLowerCase() === sel);
                if (sign) startPracticeSession({ ...sign, domain: group.domain });
                picker.selectedIndex = 0;  // reset for next pick
            });
            pickerRow.appendChild(picker);
            section.appendChild(pickerRow);
        }

        container.appendChild(section);
    });
}

function filterSignBank() {
    if (!signBankData) return;
    const query = document.getElementById('sbSearch').value.toLowerCase().trim();
    const container = document.getElementById('sbGrid');
    const visibleEl = document.getElementById('sbVisibleCount');
    const emptyEl = document.getElementById('sbEmpty');

    let visible = 0;

    container.querySelectorAll('.sb-group').forEach(group => {
        const cards = group.querySelectorAll('.sb-card');
        let groupVisible = 0;
        cards.forEach(card => {
            const match = !query || card.dataset.gloss.includes(query);
            card.style.display = match ? '' : 'none';
            if (match) { visible++; groupVisible++; }
        });
        group.style.display = groupVisible === 0 ? 'none' : '';
        const countEl = group.querySelector('.sb-group-count');
        if (countEl && query) {
            countEl.textContent = groupVisible + ' shown';
        } else if (countEl) {
            const total = cards.length;
            countEl.textContent = total + ' sign' + (total !== 1 ? 's' : '');
        }
    });

    visibleEl.textContent = query ? visible + ' of ' + signBankData.count : signBankData.count + ' shown';
    emptyEl.style.display = visible === 0 ? 'block' : 'none';
}

// ══════════════════════════════════════════════════════════════
// PRACTICE MODE (2.6) — Record yourself and get AI feedback
// ══════════════════════════════════════════════════════════════
let practiceSign = null;       // { gloss, video_url, domain }

function _renderPracComparison(comp) {
    const panel = document.getElementById('pracComparisonPanel');
    if (!panel) return;

    if (!comp || comp.error) {
        panel.style.display = comp && comp.error ? 'block' : 'none';
        if (comp && comp.error) {
            document.getElementById('pracComparisonIssues').innerHTML =
                `<div style="color:#f07070;font-size:0.82rem">Comparison failed: ${comp.error}</div>`;
        }
        return;
    }

    panel.style.display = 'block';
    const issuesEl = document.getElementById('pracComparisonIssues');
    const rawEl = document.getElementById('pracComparisonRaw');
    issuesEl.innerHTML = '';

    const severityColors = {
        error: '#f07070',
        warn: '#f0a050',
        info: 'var(--text-secondary)',
        ok: '#4caf80',
    };
    const severityIcons = { error: '\u274C', warn: '\u26A0\uFE0F', info: '\u2139\uFE0F', ok: '\u2705' };

    const issues = comp.issues || [];
    const severity = comp.severity || [];
    for (let i = 0; i < issues.length; i++) {
        const sev = severity[i] || 'info';
        const row = document.createElement('div');
        row.style.cssText = `font-size:0.82rem;margin:4px 0;padding:4px 8px;border-left:3px solid ${severityColors[sev]};color:var(--text-primary)`;
        row.innerHTML = `<span style="margin-right:6px">${severityIcons[sev]}</span>${issues[i]}`;
        issuesEl.appendChild(row);
    }

    // Raw metrics — everything except the issues list for clean debugging
    const raw = { ...comp };
    delete raw.issues;
    delete raw.severity;
    rawEl.textContent = JSON.stringify(raw, null, 2);
}

// Toggle the "What was captured" video between the raw recording and a
// server-rendered pose-only stick-figure video. The pose video is rendered
// lazily on first toggle (visualize_pose.exe + ffmpeg), then cached on disk
// and as a blob URL for the rest of this diagnose session.
async function togglePracVideoView() {
    const btn = document.getElementById('pracVideoModeToggle');
    const videoEl = document.getElementById('pracCapturedVideo');
    if (!btn || !videoEl) return;

    const mode = btn.dataset.mode || 'original';
    if (mode === 'original') {
        // Switching to pose
        if (!videoEl._sampleId) {
            showToast('No saved sample to render', 'error');
            return;
        }
        if (!videoEl._poseUrl) {
            const wasPlaying = !videoEl.paused;
            videoEl.pause();
            const orig = btn.textContent;
            btn.textContent = 'Generating...';
            btn.disabled = true;
            try {
                const r = await fetch(`/api/practice-pose-video/${videoEl._sampleId}`);
                if (!r.ok) {
                    let msg = `HTTP ${r.status}`;
                    try { const j = await r.json(); if (j && j.error) msg = j.error; } catch {}
                    throw new Error(msg);
                }
                const blob = await r.blob();
                videoEl._poseUrl = URL.createObjectURL(blob);
            } catch (e) {
                console.error('[Practice] Pose video render failed:', e);
                showToast('Pose render failed: ' + e.message, 'error');
                btn.textContent = orig;
                btn.disabled = false;
                if (wasPlaying) videoEl.play().catch(() => {});
                return;
            }
            btn.disabled = false;
        }
        videoEl.src = videoEl._poseUrl;
        videoEl.play().catch(() => {});
        btn.textContent = 'Show original';
        btn.dataset.mode = 'pose';
    } else {
        // Switching back to original
        if (videoEl._origUrl) {
            videoEl.src = videoEl._origUrl;
            videoEl.play().catch(() => {});
        }
        btn.textContent = 'Show pose';
        btn.dataset.mode = 'original';
    }
}

let practiceCameraStream = null;
let practiceStats = { correct: 0, attempts: 0 };

async function startPracticeSession(sign) {
    practiceSign = sign;
    practiceStats = { correct: 0, attempts: 0 };

    // Hide the sign bank browse UI, show the practice session inline
    const hero = document.querySelector('.sb-hero');
    if (hero) hero.style.display = 'none';
    const searchRow = document.querySelector('.sb-search-row');
    if (searchRow) searchRow.style.display = 'none';
    const picker = document.getElementById('customPracticePicker');
    if (picker) picker.style.display = 'none';
    document.getElementById('sbGrid').style.display = 'none';
    document.getElementById('sbEmpty').style.display = 'none';
    document.getElementById('practiceSession').style.display = 'block';
    document.getElementById('pracSignName').textContent = sign.gloss.toUpperCase();
    updatePracticeScore();

    // Update static practice header: "Practicing 'Coffee' in Domain RESTAURANT"
    // Gloss = first letter upper + rest lower; domain label = ALL CAPS.
    _updatePracticeHeader(sign);

    // Reset prediction badge from any prior session.
    const _predBadge = document.getElementById('pracPredictionBadge');
    if (_predBadge) { _predBadge.style.display = 'none'; _predBadge.className = 'prac-prediction-badge'; }
    const _statusBadge = document.getElementById('pracStatusBadge');
    if (_statusBadge) _statusBadge.style.display = 'none';

    // Set up reference video (may not exist for all signs — graceful fallback)
    const refVid = document.getElementById('pracRefVideo');
    const refPanel = refVid.closest('.prac-panel');
    if (sign.video_url) {
        refVid.style.display = 'block';
        refVid.src = sign.video_url;
        refVid.play().catch(() => {});
        if (refPanel) {
            refPanel.style.display = '';
            const msg = refPanel.querySelector('.prac-no-ref');
            if (msg) msg.remove();
        }
    } else {
        refVid.style.display = 'none';
        refVid.src = '';
        if (refPanel) {
            // Hide the entire reference panel when no video — gives camera full width
            const isMobile = window.matchMedia('(max-width: 640px)').matches;
            if (isMobile) {
                refPanel.style.display = 'none';
            } else {
                refPanel.style.display = '';
                let msg = refPanel.querySelector('.prac-no-ref');
                if (!msg) {
                    msg = document.createElement('div');
                    msg.className = 'prac-no-ref';
                    msg.style.cssText = 'padding:40px 20px;text-align:center;color:var(--text-muted);font-size:0.85rem;background:var(--bg-page);border-radius:8px;';
                    msg.innerHTML = `<div style="font-size:2rem;margin-bottom:8px">\uD83D\uDCD6</div>No reference video yet.<br>Sign <strong>${sign.gloss.toUpperCase()}</strong> based on what you know.`;
                    const videoBox = refPanel.querySelector('.prac-video-box');
                    if (videoBox) videoBox.appendChild(msg);
                }
            }
        }
    }

    // Reset feedback
    document.getElementById('pracFeedback').style.display = 'none';
    document.getElementById('pracRecordBtn').disabled = false;
    document.getElementById('pracRecordBtn').innerHTML = '&#x1F534; Record Sign';

    // Start camera
    const placeholder = document.getElementById('pracCameraPlaceholder');
    const feed = document.getElementById('pracCameraFeed');
    try {
        const isMobile = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent);
        const isPortrait = window.innerHeight > window.innerWidth;
        // Desktop: 640x480 matches /ws-test + live-mode streaming pipeline so
        // framing/zoom carry forward unchanged.
        const videoConstraint = (isMobile && isPortrait)
            ? { facingMode: 'user', width: { ideal: 720 }, height: { ideal: 1280 }, aspectRatio: { ideal: 9/16 } }
            : { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } };
        practiceCameraStream = await navigator.mediaDevices.getUserMedia({
            video: videoConstraint,
            audio: false,
        });
        feed.srcObject = practiceCameraStream;
        feed.style.display = 'block';
        placeholder.style.display = 'none';
        // Adapt practice camera box to native video aspect — but only on desktop/tablet.
        // On mobile phones, CSS handles portrait (9/16) with object-fit: cover.
        const applyPracAspect = () => {
            if (!feed.videoWidth) return;
            const isMobilePhone = window.matchMedia('(max-width: 640px)').matches;
            const box = feed.closest('.prac-video-box') || feed.closest('.prac-camera-box') || feed.parentElement;
            if (!box) return;
            if (isMobilePhone) {
                // Let CSS rule. Clear any inline overrides.
                box.style.aspectRatio = '';
                box.style.maxHeight = '';
            } else {
                box.style.aspectRatio = `${feed.videoWidth} / ${feed.videoHeight}`;
                box.style.maxHeight = '';
            }
        };
        feed.addEventListener('loadedmetadata', applyPracAspect);
        if (feed.videoWidth) applyPracAspect();
    } catch (e) {
        placeholder.textContent = 'Camera access needed for practice';
        placeholder.style.display = 'flex';
        feed.style.display = 'none';
        showToast('Camera access denied — practice mode needs your camera', 'error');
    }
}

// ── Custom Practice: jump straight to the signing page ──
// The standalone picker is no longer needed; the signing screen has inline
// word + domain selectors. We just pick a sensible default sign here and
// hand off to startPracticeSession; the user can switch from there.
async function openCustomPractice() {
    // Default: emergency / first word in that vocab, with localStorage
    // memory of the user's last domain so we don't reset on every click.
    let domain = 'emergency';
    try {
        const saved = localStorage.getItem('signbridge.lastPracticeDomain');
        if (saved) domain = saved;
    } catch (_) {}

    let firstGloss = '';
    try {
        const resp = await fetch(`/api/vocabulary/${domain}`);
        const data = await resp.json();
        const glosses = (data.glosses || []).slice().sort();
        firstGloss = glosses[0] || '';
    } catch (e) {
        showToast('Could not load vocabulary', 'error');
        return;
    }
    if (!firstGloss) {
        showToast('No glosses available for ' + domain, 'error');
        return;
    }

    // See if a reference video exists; not required.
    const videoUrl = `/sign-bank/${firstGloss.toLowerCase()}.mp4`;
    let video_url = null;
    try {
        const resp = await fetch(videoUrl, { method: 'HEAD' });
        if (resp.ok) video_url = videoUrl;
    } catch (_) {}

    _practiceLaunchedFromPicker = false;   // picker is gone; back returns to grid
    await startPracticeSession({ gloss: firstGloss, video_url, domain });
}

// Legacy picker still exists in the DOM but is never opened from the new flow.
// Keep these stubs so any latent reference doesn't throw.
async function _legacy_openCustomPractice() {
    const picker = document.getElementById('customPracticePicker');
    picker.style.display = 'block';

    const domainSelect = document.getElementById('customPracticeDomain');
    if (domainSelect.options.length === 0) {
        try {
            const resp = await fetch('/api/registry');
            const registry = await resp.json();
            const domains = registry.domains || {};
            Object.entries(domains).forEach(([key, entry]) => {
                if (entry.model_dir) {
                    const opt = document.createElement('option');
                    opt.value = key;
                    opt.textContent = entry.label || key;
                    if (key === selectedDomain) opt.selected = true;
                    domainSelect.appendChild(opt);
                }
            });
        } catch (e) {
            showToast('Could not load domains', 'error');
            return;
        }
    }
    await loadCustomPracticeVocab();
}

function closeCustomPractice() {
    document.getElementById('customPracticePicker').style.display = 'none';
}

async function loadCustomPracticeVocab() {
    const domain = document.getElementById('customPracticeDomain').value;
    const glossSelect = document.getElementById('customPracticeGloss');
    glossSelect.innerHTML = '<option>Loading...</option>';
    try {
        const resp = await fetch(`/api/vocabulary/${domain}`);
        const data = await resp.json();
        glossSelect.innerHTML = '';
        (data.glosses || []).forEach(g => {
            const opt = document.createElement('option');
            opt.value = g;
            opt.textContent = g;
            glossSelect.appendChild(opt);
        });
    } catch (e) {
        glossSelect.innerHTML = '<option>Failed to load</option>';
    }
}

// Tracks whether the current practice session was launched from the
// Custom Practice picker. If yes, the Back button reopens the picker
// instead of returning all the way to the Sign Bank grid.
let _practiceLaunchedFromPicker = false;

function startCustomPractice() {
    const domain = document.getElementById('customPracticeDomain').value;
    const gloss = document.getElementById('customPracticeGloss').value;
    if (!gloss) return;
    _practiceLaunchedFromPicker = true;

    // Check if we have a sign bank video for this gloss (optional)
    const videoUrl = `/sign-bank/${gloss.toLowerCase()}.mp4`;
    fetch(videoUrl, { method: 'HEAD' }).then(resp => {
        const sign = {
            gloss: gloss,
            video_url: resp.ok ? videoUrl : null,
            domain: domain,
        };
        closeCustomPractice();
        startPracticeSession(sign);
    }).catch(() => {
        closeCustomPractice();
        startPracticeSession({ gloss, video_url: null, domain });
    });
}

// ── Static practice header populator ────────────────────────────────────────
// The in-page word/domain selectors were removed (cluttered the practice
// UI). To switch signs the user clicks "Back to Sign Bank" and picks
// another one. This helper just fills in the static "Practicing 'Gloss'
// in Domain DOMAIN" label on session start.
//
// Casing: gloss = first letter upper + rest lower (e.g. "Coffee", "A lot");
// domain label = ALL CAPS via toUpperCase().
function _firstLetterCap(s) {
    if (!s) return '';
    const t = String(s);
    return t.charAt(0).toUpperCase() + t.slice(1).toLowerCase();
}

async function _updatePracticeHeader(sign) {
    const glossEl  = document.getElementById('pracHeaderGloss');
    const domainEl = document.getElementById('pracHeaderDomain');
    if (glossEl)  glossEl.textContent  = `'${_firstLetterCap(sign.gloss || '')}'`;
    // For the domain side, try to resolve a friendly label via the
    // registry; fall back to the key if the registry isn't available.
    let domainText = String(sign.domain || '').toUpperCase();
    try {
        const resp = await fetch('/api/registry');
        if (resp.ok) {
            const registry = await resp.json();
            const entry = (registry.domains || {})[sign.domain];
            if (entry && entry.label) domainText = String(entry.label).toUpperCase();
        }
    } catch (_) {}
    if (domainEl) domainEl.textContent = domainText;
}

// Defensive shims for any leftover call sites of the now-removed
// selectors. They still get called from startPracticeSession's history
// (and from any unforeseen path), so keep them callable as no-ops.
async function _populatePracticeDomainSelector() { /* removed */ }
async function _populatePracticeWordSelector(_domain) { /* removed */ }
async function switchPracticeWord(_newGloss) { /* removed */ }
async function switchPracticeDomain(_newDomain) { /* removed */ }

// Mirror of the existing reference-video setup inside startPracticeSession,
// pulled out so switchPracticeWord can reuse it without duplicating code.
function _applyPracticeReferenceVideo() {
    if (!practiceSign) return;
    const refVid = document.getElementById('pracRefVideo');
    const refPanel = refVid ? refVid.closest('.prac-panel') : null;
    const existingNoRef = refPanel ? refPanel.querySelector('.prac-no-ref') : null;
    if (existingNoRef) existingNoRef.remove();
    if (practiceSign.video_url) {
        if (refPanel) refPanel.style.display = '';
        refVid.src = practiceSign.video_url;
        try { refVid.play().catch(()=>{}); } catch(_) {}
    } else {
        if (refVid) refVid.removeAttribute('src');
        if (refPanel) {
            const isMobile = window.matchMedia('(max-width: 640px)').matches;
            if (isMobile) { refPanel.style.display = 'none'; }
            else {
                refPanel.style.display = '';
                const msg = document.createElement('div');
                msg.className = 'prac-no-ref';
                msg.style.cssText = 'padding:40px 20px;text-align:center;color:var(--text-muted);font-size:0.85rem;background:var(--bg-page);border-radius:8px;';
                msg.innerHTML = `<div style="font-size:2rem;margin-bottom:8px">📖</div>No reference video yet.<br>Sign <strong>${(practiceSign.gloss || '').toUpperCase()}</strong> based on what you know.`;
                const box = refPanel.querySelector('.prac-video-box');
                if (box) box.appendChild(msg);
            }
        }
    }
}

function exitPracticeSession() {
    // Stop the streaming practice WS if listening.
    if (practiceStream) {
        try { practiceStream.stop(); } catch (_) {}
        practiceStream = null;
    }
    // Stop camera
    if (practiceCameraStream) {
        practiceCameraStream.getTracks().forEach(t => t.stop());
        practiceCameraStream = null;
    }
    const feed = document.getElementById('pracCameraFeed');
    feed.srcObject = null;
    feed.style.display = 'none';
    document.getElementById('pracCameraPlaceholder').style.display = 'flex';
    document.getElementById('pracCameraPlaceholder').textContent = 'Camera will start when you record';

    // Stop reference video
    const refVid = document.getElementById('pracRefVideo');
    refVid.pause();
    refVid.removeAttribute('src');

    document.getElementById('practiceSession').style.display = 'none';

    // Custom Practice picker is gone -- Back always returns to the Sign Bank grid.
    _practiceLaunchedFromPicker = false;
    practiceSign = null;

    const hero = document.querySelector('.sb-hero');
    if (hero) hero.style.display = '';
    const searchRow = document.querySelector('.sb-search-row');
    if (searchRow) searchRow.style.display = '';
    document.getElementById('sbGrid').style.display = '';
}

function updatePracticeScore() {
    const el = document.getElementById('pracScore');
    if (practiceStats.attempts === 0) {
        el.textContent = 'No attempts yet';
    } else {
        el.textContent = `${practiceStats.correct} / ${practiceStats.attempts} correct`;
    }
}

// Streaming practice handle (phase 1: full pipeline port to Sign Bank).
// Holds the active StreamingLiveMode instance while practice is listening
// for a sign. Cleaned up by practiceRetry() / exitPracticeSession().
let practiceStream = null;

async function practiceRecord() {
    if (!practiceCameraStream || !practiceSign) return;
    // Reuse if already listening (idempotent click).
    if (practiceStream) return;

    const btn = document.getElementById('pracRecordBtn');
    const feedback = document.getElementById('pracFeedback');
    feedback.style.display = 'none';

    // Clear any prior prediction badge.
    const predBadge = document.getElementById('pracPredictionBadge');
    if (predBadge) { predBadge.style.display = 'none'; predBadge.className = 'prac-prediction-badge'; }

    // Just disable the button -- no spinner / no label change. The corner
    // status badge will appear only when the pipeline is actually busy
    // (Signing / Recognizing); the "Listening" state is implicit because
    // the button is gray.
    btn.disabled = true;
    _practiceSetStatus(null);

    // Use the same StreamingLiveMode class that drives /ws/live-stream, but
    // point at the practice endpoint and give it the target gloss so the
    // server can compute target-match on each segment.
    const videoEl = document.getElementById('pracCameraFeed');
    const targetGloss = (practiceSign.gloss || '').toUpperCase();
    const targetDomain = practiceSign.domain === 'other' ? 'generic' : practiceSign.domain;

    // Read current zone slider values (same control as live mode).
    const _zTop = parseInt((document.getElementById('streamZoneTop') || {}).value || 0) / 100;
    const _zBot = parseInt((document.getElementById('streamZoneBottom') || {}).value || 90) / 100;

    practiceStream = new StreamingLiveMode({
        wsPath: '/ws/practice-stream',
        domain: targetDomain,
        zoneTop: _zTop,
        zoneBottom: _zBot,
    });
    // Practice's start message includes target_gloss for server-side match check.
    practiceStream.targetGloss = targetGloss;
    // Patch _openWS to include target_gloss in the start payload. Simpler than
    // adding a constructor field since this is a one-line tweak.
    const origOpenWS = practiceStream._openWS.bind(practiceStream);
    practiceStream._openWS = function () {
        origOpenWS();
        const _origOnOpen = this._ws.onopen;
        this._ws.onopen = (ev) => {
            // Re-send start with target_gloss after the base class's start fires.
            try {
                this._ws.send(JSON.stringify({
                    type: 'tune',
                    target_gloss: this.targetGloss || '',
                }));
            } catch (_) {}
            if (_origOnOpen) _origOnOpen.call(this._ws, ev);
        };
    };

    // Intercept the WS message handler to react to practice_result events.
    // The base class's _handleMessage doesn't know about practice events --
    // we install a wrapper that handles them and falls through to the base.
    const baseHandle = practiceStream._handleMessage.bind(practiceStream);
    practiceStream._handleMessage = function (m) {
        if (m.type === 'practice_result') {
            _practiceOnResult(m);
            return;
        }
        if (m.type === 'practice_error') {
            console.warn('[Practice WS] error:', m.msg);
            return;
        }
        baseHandle(m);
    };

    // Drive the corner status badge ("✨ Listening / Signing / Recognizing")
    // off the streaming class's state events. This replaces the older big
    // fullscreen "LISTENING" overlay which was intrusive and short-lived.
    // Only two statuses are surfaced on the camera: Signing (motion detected,
    // segment being captured) and Recognizing (server-side pose+inference in
    // flight). The implicit "waiting for input" state is conveyed by the
    // grayed-out Record button -- no badge needed.
    practiceStream.onStateChange = (state) => {
        if (state === 'signing')        _practiceSetStatus('Signing');
        else if (state === 'analyzing') _practiceSetStatus('Recognizing');
        else                            _practiceSetStatus(null);
    };

    await practiceStream.init(videoEl);
}

function _practiceSetStatus(text) {
    const badge = document.getElementById('pracStatusBadge');
    const label = document.getElementById('pracStatusText');
    if (!badge) return;
    if (!text) { badge.style.display = 'none'; return; }
    if (label) label.textContent = text;
    badge.style.display = 'flex';
}

// Called by the wrapped message handler when the server emits practice_result.
function _practiceOnResult(m) {
    const btn = document.getElementById('pracRecordBtn');
    // Tear down the streaming session — we got our one result for this attempt.
    if (practiceStream) {
        try { practiceStream.stop(); } catch (_) {}
        practiceStream = null;
    }
    // Hide the status badge; show the prediction badge on the camera.
    _practiceSetStatus(null);
    const predBadge = document.getElementById('pracPredictionBadge');
    if (predBadge) {
        const predicted = (m.gloss || '?').toUpperCase();
        const conf = Math.round((m.confidence || 0) * 100);
        const matched = !!m.match_top1;
        predBadge.textContent = `${predicted} ${matched ? '✓' : '✗'} ${conf}%`;
        predBadge.className = 'prac-prediction-badge ' + (matched ? 'match' : 'miss');
        predBadge.style.display = 'block';
    }
    const overlay = document.getElementById('pracBigCountdown');
    if (overlay) overlay.style.display = 'none';

    practiceStats.attempts++;
    const predicted = (m.gloss || '').toLowerCase();
    const expected  = (practiceSign.gloss || '').toLowerCase();
    const confidence = Math.round((m.confidence || 0) * 100);
    const topKGlosses = (m.top_k || []).map(k => (k.gloss || '').toLowerCase());

    // Diagnostic shape matches what showPracticeFeedback expects.
    const diag = {
        videoBlob: null,         // phase 2: provide via /api/practice-clips/<id>
        sampleId: null,
        topK: m.top_k || [{ gloss: m.gloss, confidence: m.confidence }],
        expected: expected,
        userPose: null,          // phase 2
        userFrames: m.frames_after_trim,
        referencePose: null,     // phase 2
        referenceFrames: 0,
        comparison: null,        // phase 2
    };

    // Feedback panel removed -- result is conveyed by the corner prediction
    // badge on the camera. We just record the stat for the score counter.
    if (m.match_top1) {
        practiceStats.correct++;
        recordPracticeResult(expected, true);
    } else {
        recordPracticeResult(expected, false);
    }
    updatePracticeScore();
    btn.disabled = false;
    // After the first attempt, the button always says "Try Again" -- clicking
    // it starts a new recognition pass (same handler as Record Sign).
    btn.innerHTML = '&#x1F501; Try Again';
}

// ── Legacy timed-record path retained below for reference / fallback. Not
// called anymore by practiceRecord(), but kept compiling so unused helpers
// remain available if needed.
async function practiceRecordLegacy() {
    if (!practiceCameraStream || !practiceSign) return;

    const btn = document.getElementById('pracRecordBtn');
    const feedback = document.getElementById('pracFeedback');
    feedback.style.display = 'none';

    btn.disabled = true;

    // Full-screen overlay for unmistakable pre-record countdown and recording indicator
    let overlay = document.getElementById('pracBigCountdown');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.id = 'pracBigCountdown';
        overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,0.55);z-index:10000;display:flex;flex-direction:column;align-items:center;justify-content:center;pointer-events:none;color:#fff;';
        document.body.appendChild(overlay);
    }
    overlay.style.display = 'flex';

    const countdownEl = document.getElementById('pracRecCountdown');
    const prepSeconds = 3;
    for (let s = prepSeconds; s > 0; s--) {
        if (!practiceCameraStream) { overlay.style.display = 'none'; return; }
        overlay.innerHTML = `
            <div style="font-size:8rem;font-weight:900;color:#ffd700;text-shadow:0 4px 20px rgba(0,0,0,0.6)">${s}</div>
            <div style="font-size:1.2rem;margin-top:12px">Get ready to sign</div>
        `;
        btn.innerHTML = `Get ready... ${s}`;
        await sleep(1000);
    }
    overlay.innerHTML = `
        <div style="font-size:6rem;font-weight:900;color:#4caf80;text-shadow:0 4px 20px rgba(0,0,0,0.6)">\u25CF REC</div>
        <div style="font-size:1.5rem;margin-top:12px">Sign NOW</div>
    `;
    btn.innerHTML = '<span class="spinner"></span> Recording...';

    // Live recording — the overlay stays to make recording state VERY obvious on mobile
    const recordDuration = 3000;  // 3 seconds to give full sign time
    const countdownStart = Date.now();
    const countdownInterval = setInterval(() => {
        const remaining = Math.max(0, recordDuration - (Date.now() - countdownStart));
        if (remaining > 0) {
            overlay.innerHTML = `
                <div style="font-size:6rem;font-weight:900;color:#ff4040;text-shadow:0 4px 20px rgba(0,0,0,0.6);animation:pulse 0.5s infinite alternate">\u25CF REC</div>
                <div style="font-size:3rem;font-weight:700;margin-top:8px">${(remaining / 1000).toFixed(1)}s</div>
            `;
            countdownEl.classList.add('on');
            countdownEl.innerHTML = `<span class="rec-dot"></span> ${(remaining / 1000).toFixed(1)}s`;
        } else {
            countdownEl.classList.remove('on');
        }
    }, 100);

    // Record
    const videoBlob = await practiceRecordCamera(recordDuration);
    clearInterval(countdownInterval);
    countdownEl.classList.remove('on');
    if (overlay) overlay.style.display = 'none';

    if (!videoBlob) {
        btn.disabled = false;
        btn.innerHTML = '&#x1F534; Record Sign';
        showToast('Recording failed — try again', 'error');
        return;
    }

    btn.innerHTML = '<span class="spinner"></span> Analyzing...';

    // Send to diagnostic endpoint — returns predictions + user pose + reference pose
    const formData = new FormData();
    formData.append('video', videoBlob, 'practice.webm');
    formData.append('domain', practiceSign.domain === 'other' ? 'generic' : practiceSign.domain);
    formData.append('target', practiceSign.gloss);

    try {
        const resp = await fetch('/api/practice-diagnose', { method: 'POST', body: formData });
        let result;
        try {
            result = await resp.json();
        } catch (parseErr) {
            const text = await resp.text().catch(() => '');
            showPracticeFeedback('incorrect',
                'Server error',
                `Server returned status ${resp.status}. ${text.substring(0, 200)}`
            );
            showToast(`Server error (${resp.status})`, 'error');
            updatePracticeScore();
            btn.disabled = false;
            btn.innerHTML = '&#x1F534; Record Sign';
            return;
        }

        practiceStats.attempts++;

        if (result.success) {
            const predicted = result.gloss.toLowerCase();
            const expected = practiceSign.gloss.toLowerCase();
            const confidence = Math.round((result.confidence || 0) * 100);

            const topKGlosses = (result.top_k || []).map(k => k.gloss.toLowerCase());
            const isExactMatch = predicted === expected;
            const isInTopK = topKGlosses.includes(expected);

            const diag = {
                videoBlob: videoBlob,
                sampleId: result.sample_id || null,
                topK: result.top_k || [{ gloss: result.gloss, confidence: result.confidence }],
                expected: expected,
                userPose: result.user_pose,
                userFrames: result.user_frames,
                referencePose: result.reference_pose,
                referenceFrames: result.reference_frames,
                comparison: result.comparison,
            };

            if (isExactMatch) {
                practiceStats.correct++;
                recordPracticeResult(expected, true);
                showPracticeFeedback('correct',
                    'Correct!',
                    `The AI recognized "${predicted.toUpperCase()}" with ${confidence}% confidence.`,
                    diag
                );
            } else if (isInTopK) {
                recordPracticeResult(expected, false);
                const rank = topKGlosses.indexOf(expected) + 1;
                showPracticeFeedback('close-match',
                    'Close!',
                    `The AI's top prediction was "${predicted.toUpperCase()}" (${confidence}%), but "${expected.toUpperCase()}" was #${rank} in the top predictions.`,
                    diag
                );
            } else {
                recordPracticeResult(expected, false);
                showPracticeFeedback('incorrect',
                    'Not quite \u2014 try again',
                    `The AI saw "${predicted.toUpperCase()}" (${confidence}%) instead of "${expected.toUpperCase()}". Watch the captured video to see what was recorded.`,
                    diag
                );
            }
        } else {
            showPracticeFeedback('incorrect',
                'Could not analyze',
                result.error || 'The sign could not be recognized. Make sure you are well-lit and centered in frame.',
                { videoBlob: videoBlob, topK: [], expected: practiceSign.gloss.toLowerCase() }
            );
        }
    } catch (e) {
        console.error('[Practice] Error:', e);
        showPracticeFeedback('incorrect',
            'Processing error',
            e.message || 'Something went wrong. Check the browser console and Flask server logs for details.'
        );
        showToast('Sign processing failed \u2014 check server logs', 'error');
    }

    updatePracticeScore();
    btn.disabled = false;
    btn.innerHTML = '&#x1F534; Record Sign';
}

function showPracticeFeedback(type, title, detail, diagnostic) {
    const el = document.getElementById('pracFeedback');
    const icons = { correct: '\u2705', incorrect: '\u274C', 'close-match': '\uD83E\uDD14' };
    el.className = 'prac-feedback ' + type;
    document.getElementById('pracFeedbackIcon').textContent = icons[type] || '';
    document.getElementById('pracFeedbackText').textContent = title;
    document.getElementById('pracFeedbackDetail').textContent = detail;

    // Diagnostic panel: captured video + top-5 bars (helps debug segmentation vs. recognition)
    const diagEl = document.getElementById('pracDiagnostic');
    if (diagnostic && diagnostic.videoBlob) {
        const videoEl = document.getElementById('pracCapturedVideo');
        // Revoke any URLs from previous diagnose so we don't leak.
        if (videoEl._origUrl) URL.revokeObjectURL(videoEl._origUrl);
        if (videoEl._poseUrl) URL.revokeObjectURL(videoEl._poseUrl);
        videoEl._origUrl = URL.createObjectURL(diagnostic.videoBlob);
        videoEl._poseUrl = null;                       // lazy-fetched on first toggle
        videoEl._sampleId = diagnostic.sampleId || null;
        videoEl.src = videoEl._origUrl;
        videoEl.play().catch(() => {});

        // Reset toggle button to "Show pose" (original is always the initial view).
        const toggleBtn = document.getElementById('pracVideoModeToggle');
        if (toggleBtn) {
            toggleBtn.textContent = 'Show pose';
            toggleBtn.dataset.mode = 'original';
            toggleBtn.disabled = !videoEl._sampleId;   // grey out if no server-side sample
        }

        const topKEl = document.getElementById('pracTopK');
        topKEl.innerHTML = '';
        const topK = diagnostic.topK || [];
        const expected = (diagnostic.expected || '').toLowerCase();
        topK.slice(0, 5).forEach((item, i) => {
            const conf = Math.round((item.confidence || 0) * 100);
            const isTarget = item.gloss.toLowerCase() === expected;
            const row = document.createElement('div');
            row.className = 'prac-topk-row' + (isTarget ? ' target' : '');
            row.innerHTML = `
                <span class="prac-topk-rank">${i + 1}.</span>
                <span class="prac-topk-gloss">${item.gloss.toUpperCase()}${isTarget ? ' \u2190 your target' : ''}</span>
                <span class="prac-topk-bar"><span class="prac-topk-fill" style="width:${conf}%"></span></span>
                <span class="prac-topk-pct">${conf}%</span>
            `;
            topKEl.appendChild(row);
        });
        diagEl.style.display = 'block';

        // Programmatic diagnostic findings (pose video player removed)
        const compareEl = document.getElementById('pracPoseCompare');
        if (diagnostic.comparison) {
            compareEl.style.display = 'block';


            _renderPracComparison(diagnostic.comparison);
        } else {
            compareEl.style.display = 'none';
        }
    } else {
        diagEl.style.display = 'none';
    }

    el.style.display = 'block';
}

function practiceRetry() {
    document.getElementById('pracFeedback').style.display = 'none';
    // If a streaming attempt is still listening (shouldn't be, but be safe),
    // tear it down so the next Record Sign click starts a clean session.
    if (practiceStream) {
        try { practiceStream.stop(); } catch (_) {}
        practiceStream = null;
    }
    const btn = document.getElementById('pracRecordBtn');
    if (btn) {
        btn.disabled = false;
        btn.innerHTML = '&#x1F534; Record Sign';
    }
    // Clear the prediction + status badges for a fresh attempt.
    const predBadge = document.getElementById('pracPredictionBadge');
    if (predBadge) { predBadge.style.display = 'none'; predBadge.className = 'prac-prediction-badge'; }
    _practiceSetStatus(null);
    // Restart the reference video
    const refVid = document.getElementById('pracRefVideo');
    refVid.currentTime = 0;
    refVid.play().catch(() => {});
}

function practiceRecordCamera(durationMs) {
    return new Promise(resolve => {
        if (!practiceCameraStream) { resolve(null); return; }
        let chunks = [];
        let recorder;
        try {
            recorder = new MediaRecorder(practiceCameraStream, { mimeType: 'video/webm' });
        } catch (e) {
            try { recorder = new MediaRecorder(practiceCameraStream); } catch (e2) { resolve(null); return; }
        }
        recorder.ondataavailable = (e) => { if (e.data.size > 0) chunks.push(e.data); };
        recorder.onstop = () => {
            const blob = new Blob(chunks, { type: 'video/webm' });
            resolve(blob.size > 0 ? blob : null);
        };
        recorder.start();
        setTimeout(() => {
            if (recorder.state === 'recording') recorder.stop();
        }, durationMs);
    });
}

// ══════════════════════════════════════════════════════════════
// SIDEBAR NAVIGATION — flat: Sign Bank, Demo, Live, Upload, Profile
// ══════════════════════════════════════════════════════════════
function navigateTo(section) {
    currentNav = section;

    // Set MODE based on which section we're in
    if (section === 'demo') MODE = 'demo';
    else if (section === 'live') MODE = 'live';

    // Update sidebar active state
    document.querySelectorAll('.sidebar-item').forEach(btn => btn.classList.remove('active'));
    const navIds = {
        'sign-bank': 'navSignBank',
        demo: 'navDemo',
        live: 'navLive',
        upload: 'navUpload',
        settings: 'navSettings',
    };
    const activeBtn = document.getElementById(navIds[section]);
    if (activeBtn) activeBtn.classList.add('active');

    // Hide all content sections (including secondary screens)
    const sectionIds = ['demoScenarios', 'liveScenarios', 'signBankScreen', 'uploadScreen', 'historyScreen',
                        'settingsScreen', 'phaseConvo', 'phaseLiveDemo', 'phaseBreakdown', 'vocabPanel'];
    sectionIds.forEach(id => { const el = document.getElementById(id); if (el) el.style.display = 'none'; });

    // Show hero on live tab (default landing), demo intro on demo tab
    const heroSection = document.getElementById('heroSection');
    if (heroSection) heroSection.style.display = section === 'live' ? '' : 'none';
    const demoIntro = document.getElementById('demoIntro');
    if (demoIntro) demoIntro.style.display = section === 'demo' ? '' : 'none';

    // Show home-screen wrapper (for padding) on content tabs that use it
    const homeScreen = document.querySelector('.home-screen');
    if (homeScreen) homeScreen.style.display = (section === 'demo' || section === 'live') ? '' : 'none';

    if (section === 'demo') {
        const el = document.getElementById('demoScenarios'); if (el) el.style.display = 'block';
    } else if (section === 'live') {
        const el = document.getElementById('liveScenarios'); if (el) el.style.display = 'block';
    } else if (section === 'sign-bank') {
        const el = document.getElementById('signBankScreen'); if (el) el.style.display = 'block';
        if (!signBankLoaded) loadSignBank();
    } else if (section === 'upload') {
        const el = document.getElementById('uploadScreen'); if (el) el.style.display = 'block';
    } else if (section === 'settings') {
        const el = document.getElementById('settingsScreen'); if (el) el.style.display = 'block';
        // Pull the freshest stored values into the Settings page UI
        // every time the tab is opened (in case the user changed
        // something in the in-session cog popup).
        syncSettingsPageUI();
    }

    // Clean up practice camera if leaving sign bank
    if (section !== 'sign-bank' && practiceCameraStream) {
        exitPracticeSession();
    }
}

// Legacy aliases for compatibility
function updateSubTabVisibility() {
    navigateTo(currentNav);
}

// ══════════════════════════════════════════════════════════════
// LIVE DEMO -- Camera + Hardcoded Caption
// ══════════════════════════════════════════════════════════════
let liveDemoCameraStream = null;

async function startLiveDemo() {
    hideHomeContent();
    document.getElementById('phaseLiveDemo').style.display = 'flex';
    lockHeaderToggles();
    pushRoute('live-demo');

    const placeholder = document.getElementById('liveDemoPlaceholder');
    placeholder.textContent = 'Requesting camera access...';
    try {
        liveDemoCameraStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        const feed = document.getElementById('liveDemoCameraFeed');
        feed.srcObject = liveDemoCameraStream;
        feed.setAttribute('autoplay', '');
        feed.setAttribute('playsinline', '');
        feed.setAttribute('muted', '');
        await feed.play();
        feed.style.display = 'block';
        placeholder.style.display = 'none';
    } catch (e) {
        placeholder.textContent = 'Camera error: ' + (e.name || '') + ' - ' + (e.message || 'Unknown error');
        showToast('Could not access camera: ' + (e.message || 'Unknown error'), 'error');
    }
}

function stopLiveDemo() {
    if (liveDemoCameraStream) {
        liveDemoCameraStream.getTracks().forEach(t => t.stop());
        liveDemoCameraStream = null;
    }
    const feed = document.getElementById('liveDemoCameraFeed');
    feed.style.display = 'none';
    feed.srcObject = null;
    document.getElementById('liveDemoPlaceholder').style.display = 'flex';
    document.getElementById('liveDemoPlaceholder').textContent = 'Starting camera...';
    document.getElementById('phaseLiveDemo').style.display = 'none';
    showHomeScreen();
}

// ══════════════════════════════════════════════════════════════
// LIVE CAPTIONS -- State
// ══════════════════════════════════════════════════════════════
let lcCameraStream = null;
let lcGlossBuffer = [];
let lcParagraphText = '';
let lcActive = false;
let lcSelectedDomain = 'doctor_visit';
let lcSelectedScenario = 'Doctor Visit';
let lcHistory = [];
const LC_BUFFER_THRESHOLD = 4;

function uploadDomainChanged() {
    const select = document.getElementById('uploadDomainSelect');
    if (select) {
        lcSelectedDomain = select.value;
        lcSelectedScenario = select.options[select.selectedIndex].textContent;
    }
}

// Legacy compat — no longer used since upload is inline with dropdown
function selectLiveCaptionsScenario(domain, scenarioName) {
    lcSelectedDomain = domain;
    lcSelectedScenario = scenarioName;
    navigateTo('upload');
}

// ══════════════════════════════════════════════════════════════
// CAPTION VIDEO -- Upload, poll, download
// ══════════════════════════════════════════════════════════════
let cvJobId = null;
let cvPollInterval = null;

function cvHandleDrop(e) {
    e.preventDefault();
    document.getElementById('cvUploadZone').style.borderColor = '#d0c4b0';
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('video/')) cvHandleFile(file);
}

async function cvHandleFile(file) {
    if (!file) return;
    cvReset();

    document.getElementById('cvUploadZone').style.display = 'none';
    document.getElementById('cvProgress').style.display = 'block';
    document.getElementById('cvFileName').textContent = file.name;
    document.getElementById('cvProgressPct').textContent = '0%';
    document.getElementById('cvProgressBar').style.width = '0%';
    document.getElementById('cvStatusMsg').textContent = 'Uploading...';

    const form = new FormData();
    form.append('video', file);
    let resp;
    try {
        resp = await fetch('/api/caption-video', { method: 'POST', body: form });
    } catch (e) {
        cvShowError('Upload failed: ' + e.message);
        showToast('Video upload failed', 'error');
        return;
    }
    const data = await resp.json();
    if (!data.success) { cvShowError(data.error || 'Upload failed'); return; }

    cvJobId = data.job_id;
    document.getElementById('cvStatusMsg').textContent = 'Pipeline starting...';
    document.getElementById('cvProgressBar').style.width = '5%';
    document.getElementById('cvProgressPct').textContent = '5%';

    cvPollInterval = setInterval(cvPollStatus, 2000);
}

async function cvPollStatus() {
    if (!cvJobId) return;
    try {
        const resp = await fetch(`/api/caption-video/${cvJobId}`);
        const job = await resp.json();
        if (!job.success) return;

        const pct = job.progress || 0;
        document.getElementById('cvProgressBar').style.width = pct + '%';
        document.getElementById('cvProgressPct').textContent = pct + '%';
        document.getElementById('cvStatusMsg').textContent = job.message || '';

        if (job.status === 'done') {
            clearInterval(cvPollInterval);
            cvShowResult();
            showToast('Video captioning complete!', 'success');
        } else if (job.status === 'error') {
            clearInterval(cvPollInterval);
            cvShowError(job.message || 'Pipeline error');
            showToast('Captioning pipeline error', 'error');
        }
    } catch (e) {
        console.warn('[CaptionVideo] Poll error:', e);
    }
}

function cvShowResult() {
    document.getElementById('cvProgress').style.display = 'none';
    document.getElementById('cvResult').style.display = 'block';

    const videoUrl = `/api/caption-video/${cvJobId}/download`;
    fetch(videoUrl)
        .then(r => r.blob())
        .then(blob => {
            const url = URL.createObjectURL(blob);
            const vid = document.getElementById('cvResultVideo');
            vid.src = url;

            const btn = document.getElementById('cvDownloadBtn');
            btn.onclick = () => {
                const a = document.createElement('a');
                a.href = url;
                a.download = 'captioned.mp4';
                a.click();
            };
        });
}

function cvShowError(msg) {
    document.getElementById('cvProgress').style.display = 'block';
    document.getElementById('cvStatusMsg').textContent = '\u2717 Error: ' + msg;
    document.getElementById('cvStatusMsg').style.color = '#f07070';
}

function cvReset() {
    if (cvPollInterval) { clearInterval(cvPollInterval); cvPollInterval = null; }
    cvJobId = null;
    document.getElementById('cvUploadZone').style.display = 'block';
    document.getElementById('cvProgress').style.display = 'none';
    document.getElementById('cvResult').style.display = 'none';
    document.getElementById('cvStatusMsg').style.color = '';
    const vid = document.getElementById('cvResultVideo');
    if (vid.src) { URL.revokeObjectURL(vid.src); vid.src = ''; }
    document.getElementById('cvFileInput').value = '';
}

function showHomeScreen(skipPush) {
    // Auto-save conversation (live mode only, if enabled)
    if (history.length > 0 && MODE === 'live') {
        const profile = loadProfile();
        if (profile.saveHistory !== false) {
            saveConversation();
        }
    }
    isPlaying = false;
    cancelAllSleeps();
    if ('speechSynthesis' in window) speechSynthesis.cancel();
    if (signVid) { signVid.pause(); signVid.removeAttribute('src'); signVid.style.display = 'none'; }
    if (liveRecognition) { try { liveRecognition.abort(); } catch(e) {} liveRecognition = null; }

    // Hide all secondary screens
    document.getElementById('phaseConvo').style.display = 'none';
    document.getElementById('phaseLiveDemo').style.display = 'none';
    document.getElementById('phaseBreakdown').style.display = 'none';
    document.getElementById('historyScreen').style.display = 'none';

    // Restore home content via current tab
    const hs = document.querySelector('.home-screen');
    if (hs) hs.style.display = '';

    if (liveDemoCameraStream) {
        liveDemoCameraStream.getTracks().forEach(t => t.stop());
        liveDemoCameraStream = null;
    }
    // Clean up practice camera if active
    if (practiceCameraStream) {
        practiceCameraStream.getTracks().forEach(t => t.stop());
        practiceCameraStream = null;
    }
    if (cvPollInterval) { clearInterval(cvPollInterval); cvPollInterval = null; }
    unlockHeaderToggles();
    updateSubTabVisibility();
    document.getElementById('vocabPanel').style.display = 'none';
    document.getElementById('vocabChips').classList.remove('open');
    document.getElementById('vocabToggleBtn').classList.remove('open');
    if (transcriptCollapsed) toggleTranscript();
    btnRestart.click();

    if (!skipPush) pushRoute('home');
}

// DOM
const btnStart = document.getElementById('btnStart');
const btnRestart = document.getElementById('btnRestart');
const transcriptMsgs = document.getElementById('transcriptMsgs');
const cameraFeed = document.getElementById('cameraFeed');
const cameraPlaceholder = document.getElementById('cameraPlaceholder');
const statusBar = document.getElementById('statusBar');
const recBadge = document.getElementById('recBadge');
const recBorder = document.getElementById('recBorder');
const cameraView = document.getElementById('cameraView');

// Sign video overlay element
const signVid = document.createElement('video');
signVid.muted = true;
signVid.playsInline = true;
signVid.setAttribute('webkit-playsinline', '');
signVid.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;object-fit:contain;z-index:4;background:#0a0a14;display:none;';
cameraView.appendChild(signVid);

// ══════════════════════════════════════════════════════════════
// HELPERS
// ══════════════════════════════════════════════════════════════
let _sleepTimers = [];
const sleep = ms => new Promise(r => {
    const id = setTimeout(r, ms);
    _sleepTimers.push(id);
});
function cancelAllSleeps() {
    _sleepTimers.forEach(id => clearTimeout(id));
    _sleepTimers = [];
}

function isLLMSentenceValid(llmSentence, glosses, hardcoded) {
    if (!llmSentence || llmSentence.length === 0 || llmSentence.length > 150) return false;

    const lower = llmSentence.toLowerCase();

    const glossRoots = {
        'meet': ['meet', 'meeting'],
        'doctor': ['doctor', 'dr'],
        'son': ['son'],
        'bed': ['bed'],
        'time': ['time'],
        'work': ['work', 'works', 'working'],
        'full': ['full'],
        'enjoy': ['enjoy', 'enjoys', 'enjoyed'],
        'go': ['go', 'goes', 'going'],
        'cook': ['cook', 'cooking', 'cooks'],
        'chair': ['chair', 'sit', 'sits', 'seat'],
        'tall': ['tall'],
        'finish': ['finish', 'finished', 'done'],
        'what': ['what'],
        'black': ['black']
    };

    let matched = 0;
    for (const g of glosses) {
        const variants = glossRoots[g] || [g];
        if (variants.some(v => lower.includes(v))) matched++;
    }

    if (matched < Math.ceil(glosses.length / 2)) return false;

    const hardcodedWords = hardcoded.toLowerCase().split(/\s+/);
    const keyWords = hardcodedWords.filter(w => w.length > 3 && !['that', 'this', 'have', 'been', 'with', 'from', 'here', 'will', 'does', 'them', 'their', 'your', "it's", "he's"].includes(w));
    if (keyWords.length > 0) {
        const keyMatched = keyWords.some(w => lower.includes(w));
        if (!keyMatched) return false;
    }

    const llmIsQuestion = llmSentence.includes('?');
    const hardcodedIsQuestion = hardcoded.includes('?');
    if (llmIsQuestion !== hardcodedIsQuestion) return false;

    return true;
}

// ══════════════════════════════════════════════════════════════
// CAMERA (with 1.5 permission modal)
// ══════════════════════════════════════════════════════════════

// Adapt the camera-view container's aspect ratio to match the actual video's
// native resolution. Makes portrait video (iPhone vertical) show tall, and
// landscape video (laptop/iPad horizontal) show wide, so hands stay in frame.
// Re-apply aspect when orientation/window size changes (phone rotation)
window.addEventListener('resize', () => {
    const cf = document.getElementById('cameraFeed');
    if (cf && cf.videoWidth) _applyCameraAspect(cf);
});

function _applyCameraAspect(videoEl) {
    if (!videoEl || !videoEl.videoWidth || !videoEl.videoHeight) return;
    const view = videoEl.closest('.camera-view');
    if (!view) return;
    const isMobilePhone = window.matchMedia('(max-width: 640px)').matches;
    if (isMobilePhone) {
        // On phones, force portrait 9:16 regardless of what the camera delivers.
        // object-fit: cover on the video will fill the box, cropping sides.
        view.style.aspectRatio = '9 / 16';
        view.style.maxHeight = '65vh';
        view.style.minHeight = '';
        // Override object-fit to cover for mobile so the video fills the box
        videoEl.style.objectFit = 'cover';
    } else {
        view.style.aspectRatio = `${videoEl.videoWidth} / ${videoEl.videoHeight}`;
        view.style.maxHeight = '';
        view.style.minHeight = '';
        videoEl.style.objectFit = '';
    }
    console.log(`[Camera] View set to ${view.style.aspectRatio}, video native=${videoEl.videoWidth}x${videoEl.videoHeight}`);
}

async function startCamera() {
    try {
        if (externalCameraUrl) {
            // ── External camera (MJPEG via canvas bridge) ──
            await startExternalCamera(externalCameraUrl);
        } else {
            // ── Local device camera ──
            // METHOD === 'sign' requests audio because Tap-to-Speak relies
            // on the implicit mic-permission grant from getUserMedia. (Tried
            // dropping audio in Kiosk/Lanyard to free the mic for
            // continuous wake-word recognition; turns out Edge's
            // SpeechRecognition silently fails to receive audio without
            // that grant, which broke Tap-to-Speak. Reverted.)
            const needsAudio = (MODE === 'live' && METHOD === 'sign');
            cameraStream = await requestCameraWithPrompt(needsAudio);

            if (!cameraStream) {
                cameraPlaceholder.style.display = 'block';
                cameraPlaceholder.style.color = '#ff6666';
                cameraPlaceholder.textContent = 'Camera access was denied or skipped';
                return;
            }
        }

        const pipSelf = document.getElementById('pipSelf');
        const pipFeed = document.getElementById('pipFeed');
        const pipLabel = document.getElementById('pipLabel');
        const cameraTitle = document.querySelector('.camera-title');

        if (interactionMode === 'in-person') {
            cameraFeed.srcObject = cameraStream;
            cameraFeed.style.display = 'block';
            cameraPlaceholder.style.display = 'none';
            pipSelf.style.display = 'none';
            cameraTitle.textContent = externalCameraUrl ? 'WiFi Camera' : 'Your Camera';
            // Adapt camera-view container to the actual video aspect (portrait vs landscape)
            const applyAspect = () => _applyCameraAspect(cameraFeed);
            cameraFeed.removeEventListener('loadedmetadata', cameraFeed._aspectHandler || (() => {}));
            cameraFeed._aspectHandler = applyAspect;
            cameraFeed.addEventListener('loadedmetadata', applyAspect);
            if (cameraFeed.videoWidth) applyAspect();
        } else {
            cameraFeed.srcObject = cameraStream;
            cameraPlaceholder.textContent = MODE === 'demo' ? 'Waiting for signer...' : 'Waiting for remote feed...';
            cameraPlaceholder.style.color = '#555';
            cameraPlaceholder.style.display = 'block';

            pipSelf.style.display = 'block';
            pipLabel.textContent = METHOD === 'speak' ? 'You (Speaker)' : 'You (Signer)';
            pipFeed.srcObject = cameraStream;
            pipFeed.onloadedmetadata = () => {
                pipFeed.play().catch(() => {});
            };
            cameraTitle.textContent = METHOD === 'speak' ? 'Signer View' : 'Speaker View';
        }
    } catch (e) {
        console.error('[Camera] Failed:', e.name, e.message);
        cameraPlaceholder.style.display = 'block';
        cameraPlaceholder.style.color = '#ff6666';
        cameraPlaceholder.textContent = 'Camera failed \u2014 ' + (e.message || 'check connection');
        showToast('Camera failed: ' + (e.message || 'Unknown error'), 'error');
    }
}

// ── Canvas bridge: MJPEG stream → MediaStream ──
async function startExternalCamera(mjpegUrl) {
    stopExternalCamera(); // clean up any previous

    // Still acquire mic for speech-to-text
    try {
        const audioStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
        // Keep audio tracks available for speech recognition
        // (Web Speech API uses system mic independently, but this ensures permission is granted)
        audioStream.getTracks(); // just hold reference
    } catch (e) {
        console.warn('[Camera] Mic access denied — speech-to-text may not work:', e.message);
    }

    // Route through Flask proxy to avoid mixed-content (HTTPS page → HTTP camera)
    const proxyUrl = '/api/camera-proxy?url=' + encodeURIComponent(mjpegUrl);

    return new Promise((resolve, reject) => {
        // Chrome only decodes subsequent MJPEG frames when the <img> is
        // attached to the document. An off-DOM `new Image()` decodes the
        // FIRST frame, fires onload, and then never updates -- the canvas
        // ends up redrawing the same pixels at 15fps, the server's motion
        // diff is always 0, no segments ever open. Attaching the element
        // (off-screen, no layout impact) keeps the decoder ticking.
        externalCameraImg = document.createElement('img');
        externalCameraImg.crossOrigin = 'anonymous';
        // Off-screen but at NATURAL size so Chrome actually decodes the
        // full image bitmap. A 1px CSS layout box appears to make Chrome
        // skip frame updates / decode only the visible portion.
        externalCameraImg.style.cssText = 'position:fixed;top:0;left:-10000px;pointer-events:none;visibility:hidden';
        document.body.appendChild(externalCameraImg);

        externalCameraCanvas = document.createElement('canvas');
        const ctx = externalCameraCanvas.getContext('2d');
        let dimensionsSet = false;

        externalCameraImg.onload = () => {
            if (!dimensionsSet) {
                // Set canvas to match camera resolution
                externalCameraCanvas.width = externalCameraImg.naturalWidth || 640;
                externalCameraCanvas.height = externalCameraImg.naturalHeight || 480;
                dimensionsSet = true;

                // Start draw loop at ~15fps. Horizontally mirror the MJPEG
                // frame so the canvas matches device-camera handedness:
                // signer's right hand at viewer's LEFT, which is what both
                // (a) the .camera-view CSS scaleX(-1) expects (so the
                // preview reads as a selfie like device cam) and
                // (b) the inference model expects (standard training-data
                // orientation -- ASL right-vs-left signs differ).
                // Applied here at the source-of-truth canvas, so every
                // downstream consumer (preview video, _sendFrame's
                // drawImage) inherits the correct orientation.
                externalCameraInterval = setInterval(() => {
                    if (externalCameraImg && externalCameraImg.complete) {
                        ctx.save();
                        ctx.translate(externalCameraCanvas.width, 0);
                        ctx.scale(-1, 1);
                        ctx.drawImage(externalCameraImg, 0, 0);
                        ctx.restore();
                    }
                }, 67);

                // Capture canvas as MediaStream
                cameraStream = externalCameraCanvas.captureStream(15);
                console.log(`[Camera] External camera connected: ${mjpegUrl} (${externalCameraCanvas.width}x${externalCameraCanvas.height})`);
                resolve();
            }
        };

        externalCameraImg.onerror = () => {
            reject(new Error('Could not connect to camera at ' + mjpegUrl));
        };

        // Timeout if camera doesn't respond in 8 seconds
        setTimeout(() => {
            if (!dimensionsSet) {
                reject(new Error('Camera connection timed out'));
            }
        }, 8000);

        externalCameraImg.src = proxyUrl;
    });
}

function stopExternalCamera() {
    if (externalCameraInterval) {
        clearInterval(externalCameraInterval);
        externalCameraInterval = null;
    }
    if (externalCameraImg) {
        externalCameraImg.src = '';
        if (externalCameraImg.parentNode) externalCameraImg.parentNode.removeChild(externalCameraImg);
        externalCameraImg = null;
    }
    externalCameraCanvas = null;
}

function stopCamera() {
    stopExternalCamera();
    if (cameraStream) {
        cameraStream.getTracks().forEach(t => t.stop());
        cameraStream = null;
    }
    const pipSelf = document.getElementById('pipSelf');
    if (pipSelf) pipSelf.style.display = 'none';
}

// ══════════════════════════════════════════════════════════════
// TEXT-TO-SPEECH
// ══════════════════════════════════════════════════════════════
function speak(text) {
    return new Promise(resolve => {
        // Honor the per-device TTS toggle (Settings -> "Speak responses
        // aloud"). Off = silent; caller still resolves so any code that
        // chains `.then(...)` after a speak() continues normally.
        if (!TTS_ENABLED) { resolve(); return; }
        if (!('speechSynthesis' in window)) { resolve(); return; }

        function doSpeak() {
            speechSynthesis.cancel();
            const u = new SpeechSynthesisUtterance(text);
            u.rate = 0.95;
            u.pitch = 1.3;
            const voices = speechSynthesis.getVoices();
            const preferred = voices.find(v => /male/i.test(v.name) && /en/i.test(v.lang) && !/female/i.test(v.name))
                || voices.find(v => /David|Mark|James|Daniel|Guy/i.test(v.name))
                || voices.find(v => /en/i.test(v.lang) && !/female|zira|hazel|susan|jenny/i.test(v.name));
            if (preferred) u.voice = preferred;
            const timeout = setTimeout(() => { speechSynthesis.cancel(); resolve(); },
                Math.max(text.split(' ').length * 500, 2500) + 3000);
            u.onend = () => { clearTimeout(timeout); resolve(); };
            u.onerror = () => { clearTimeout(timeout); resolve(); };
            speechSynthesis.speak(u);
        }

        if (speechSynthesis.getVoices().length > 0) {
            doSpeak();
        } else {
            speechSynthesis.onvoiceschanged = () => { doSpeak(); };
            setTimeout(() => { doSpeak(); }, 500);
        }
    });
}

// ══════════════════════════════════════════════════════════════
// SPEECH RECOGNITION
// ══════════════════════════════════════════════════════════════
function listenForSpeech(expectedText) {
    return new Promise(resolve => {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

        if (!SpeechRecognition) {
            const hint = statusBar.querySelector('.script-hint');
            if (hint) hint.textContent = 'Tap here when done speaking';
            function onClick() {
                statusBar.removeEventListener('click', onClick);
                resolve(expectedText);
            }
            statusBar.addEventListener('click', onClick);
            return;
        }

        const recognition = new SpeechRecognition();
        recognition.lang = 'en-US';
        recognition.interimResults = true;
        recognition.continuous = false;
        recognition.maxAlternatives = 1;

        let finalResult = '';
        let resolved = false;

        const timeout = setTimeout(() => {
            if (!resolved) {
                resolved = true;
                recognition.stop();
                resolve(finalResult || expectedText);
            }
        }, 15000);

        recognition.onresult = (event) => {
            let interim = '';
            for (let i = event.resultIndex; i < event.results.length; i++) {
                const transcript = event.results[i][0].transcript;
                if (event.results[i].isFinal) {
                    finalResult += transcript;
                } else {
                    interim = transcript;
                }
            }
            const display = finalResult || interim;
            if (display) {
                const hint = statusBar.querySelector('.script-hint');
                if (hint) hint.innerHTML = '<span class="listening-indicator"></span>Hearing: "' + display + '"';
            }
        };

        recognition.onend = () => {
            if (!resolved) {
                resolved = true;
                clearTimeout(timeout);
                resolve(finalResult || expectedText);
            }
        };

        recognition.onerror = () => {
            if (!resolved) {
                resolved = true;
                clearTimeout(timeout);
                resolve(expectedText);
            }
        };

        recognition.start();
    });
}

// ══════════════════════════════════════════════════════════════
// SIGN VIDEO PLAYBACK
// ══════════════════════════════════════════════════════════════
function playSignVideo(url) {
    return new Promise(resolve => {
        if (!isPlaying && !liveActive) { resolve(false); return; }
        let done = false;
        function finish(r) {
            if (done) return;
            done = true;
            signVid.onended = null;
            signVid.onerror = null;
            signVid.onloadeddata = null;
            signVid.onpause = null;
            clearTimeout(to);
            resolve(r);
        }

        signVid.pause();
        signVid.onended = null;
        signVid.onerror = null;
        signVid.onloadeddata = null;

        const to = setTimeout(() => finish(false), 5000);
        signVid.onpause = () => { if (!isPlaying && !liveActive) finish(false); };

        signVid.onerror = () => finish(false);
        signVid.onloadedmetadata = () => {
            clearTimeout(to);
            const dynamicTimeout = Math.max((signVid.duration || 5) * 1000 + 3000, 5000);
            const to2 = setTimeout(() => finish(false), dynamicTimeout);
            const origFinish = finish;
            finish = (r) => { clearTimeout(to2); origFinish(r); };
        };
        signVid.onloadeddata = () => {
            signVid.onloadeddata = null;
            signVid.onended = () => finish(true);
            signVid.play().catch(() => finish(false));
        };

        signVid.style.display = 'block';
        signVid.src = url;
        signVid.load();
    });
}

// ══════════════════════════════════════════════════════════════
// Init sub-tab visibility on page load
updateSubTabVisibility();

// ══════════════════════════════════════════════════════════════
// LOAD DATA (always load demo conversation so it's ready when needed)
// ══════════════════════════════════════════════════════════════
(async function loadConversation() {
    try {
        const resp = await fetch('/api/conversation');
        conversation = await resp.json();
        btnStart.disabled = false;
    } catch (e) {
        console.warn('[Demo] Could not load conversation data:', e);
    }
})();

// ══════════════════════════════════════════════════════════════
// CONVERSATION FLOW -- DEMO MODE
// ══════════════════════════════════════════════════════════════
btnStart.addEventListener('click', async () => {
    // Single button that morphs between "Start Conversation" and "Restart".
    // On subsequent clicks we just delegate to btnRestart's handler. btnRestart
    // stays in the DOM (still hidden) so the existing handler keeps working.
    if (btnStart.dataset.armed === '1') {
        btnRestart.click();
        return;
    }
    btnStart.dataset.armed = '1';
    btnStart.textContent = 'Restart';
    // Reuse the existing secondary-button styling so it visually reads as restart.
    btnStart.classList.remove('btn-primary');
    btnStart.classList.add('btn-secondary');
    // btnRestart stays hidden -- btnStart now plays both roles.
    btnRestart.style.display = 'none';
    history = [];

    if ('speechSynthesis' in window) {
        const warmup = new SpeechSynthesisUtterance('');
        warmup.volume = 0;
        speechSynthesis.speak(warmup);
    }

    if (MODE === 'live') {
        // Kiosk/Lanyard: arm the single-shot wake-word session
        // synchronously here so it inherits this click's user-gesture
        // context. startLiveMode is async; once it awaits, the gesture
        // is gone and SpeechRecognition.start() will be rejected. The
        // listener guards on liveActive, so flip it true now (startLiveMode
        // sets it again -- idempotent).
        if (SIGNBRIDGE_MODE === 'kiosk' || SIGNBRIDGE_MODE === 'lanyard') {
            liveActive = true;
            stopWakeWordListener();
            startWakeWordListener();
        }
        await startLiveMode();
        return;
    }

    if (!conversation) return;
    currentTurn = 0;
    isPlaying = true;

    await startCamera();
    playNextTurn();
});

btnRestart.addEventListener('click', async () => {
    console.log('[Restart] clicked. liveActive=', liveActive,
                ' hasSegmenter=', !!signSegmenter,
                ' segmenterPaused=', signSegmenter && typeof signSegmenter.isPaused === 'function' ? signSegmenter.isPaused() : '?');
    // If Tap-to-Speak is mid-listen, stop it first -- otherwise its
    // delayed onend would fire after this handler completes, calling
    // resume() on a streamer that we've already restarted past, and
    // (worse) appending the half-finished speaker text into the fresh
    // turn we're about to start. Wrapped defensively so a bug in the
    // cleanup path can never block the actual Restart sequence below.
    try { _stopTapReplyListening('restart'); }
    catch (e) { console.warn('[Restart] tap-reply cleanup threw:', e); }

    // In streaming-live mode, Restart is a SOFT reset of the in-flight
    // utterance only. Sequence:
    //   1. Pause streaming -- no new segments after this point.
    //   2. Pause also sends `reset` to the server, clearing motion state.
    //   3. Wait for any segment whose inference was already dispatched
    //      to finish (so its gloss doesn't appear as the first gloss of
    //      the new utterance).
    //   4. Clear local state (glosses, caption, gloss panel).
    //   5. Resume streaming with a fresh server motion gate.
    // Without the pause/drain, a sign in mid-capture leaks a gloss past
    // the clear -- which manifests as "first word missed" / "phantom first
    // gloss" after Restart.
    if (liveActive && signSegmenter && typeof signSegmenter.clearGlosses === 'function') {
        statusBar.innerHTML = '<span class="processing ai"><span class="ai-sparkle">✨</span>Restarting... <span class="spinner"></span></span>';
        if (typeof signSegmenter.pause === 'function') signSegmenter.pause();
        // Drain in-flight inference (max 4s).
        const drainStart = Date.now();
        while (typeof signSegmenter.inFlightCount === 'function'
               && signSegmenter.inFlightCount() > 0
               && (Date.now() - drainStart) < 4000) {
            await sleep(120);
        }
        // Also wait for any in-flight LLM call to settle (cheap; usually 0).
        while (_captionPending && (Date.now() - drainStart) < 5000) {
            await sleep(120);
        }
        // Now safe to clear -- no more late events will land.
        signSegmenter.clearGlosses();
        liveCollectedGlosses = [];
        _clearGlossRows();
        _setLiveCaption(null);
        _currentTurnRowEl = null;
        _currentTurnHistoryIndex = -1;
        _runningCaption = '';
        _lastSentGlossCount = 0;
        _lastCtqi = null;
        _prevCtqiWasHigh = true;
        _captionIsEdited = false;
        _lastRenderedCaptionHtml = '';
    _regenerateAttempts = 0;
        if (typeof signSegmenter.resume === 'function') signSegmenter.resume();
        const btnSendMessage = document.getElementById('btnSendMessage');
        if (btnSendMessage) {
            btnSendMessage.textContent = 'Send Message';
            btnSendMessage.style.background = '#4caf80';
            btnSendMessage.disabled = false;
            btnSendMessage.onclick = () => sendNowAndPause();
        }
        statusBar.innerHTML = '<span class="processing">Sign when ready — your signs will appear here</span>';
        return;
    }

    // Full reset (demo / non-live modes).
    currentTurn = 0;
    history = [];
    isPlaying = false;
    transcriptMsgs.innerHTML = '';
    btnStart.disabled = false;
    // Morph the merged Start/Restart button back to its initial state.
    btnStart.textContent = 'Start Conversation';
    btnStart.dataset.armed = '0';
    btnStart.classList.remove('btn-secondary');
    btnStart.classList.add('btn-primary');
    btnRestart.style.display = 'none';
    recBadge.classList.remove('on');
    recBorder.classList.remove('on');
    signVid.style.display = 'none';
    document.getElementById('btnHowItWorks').style.display = 'none';
    statusBar.innerHTML = '<span class="prompt">Press "Start" to begin the conversation</span>';
    stopLiveMode();
    stopCamera();
    cameraFeed.style.display = 'none';
    cameraPlaceholder.style.display = '';
    cameraPlaceholder.textContent = 'Camera will start when you begin';

    const btnRecordSign = document.getElementById('btnRecordSign');
    const btnSendMessage = document.getElementById('btnSendMessage');
    if (btnRecordSign) btnRecordSign.style.display = 'none';
    if (btnSendMessage) btnSendMessage.style.display = 'none';
    const btnSpeakSend = document.getElementById('btnSpeakSend');
    if (btnSpeakSend) btnSpeakSend.style.display = 'none';

    // Reset countdown
    const countdown = document.getElementById('recCountdown');
    if (countdown) countdown.classList.remove('on');
});

async function playNextTurn() {
    if (currentTurn >= conversation.turns.length || !isPlaying) {
        if (isPlaying) {
            statusBar.innerHTML = '<span class="sentence">Conversation complete!</span>';
            stopCamera();
            if (MODE === 'demo') {
                document.getElementById('btnHowItWorks').style.display = 'inline-block';
            }
        }
        return;
    }

    const turn = conversation.turns[currentTurn];
    if (turn.speaker === 'doctor') {
        await playDoctorTurnDemo(turn);
    } else {
        await playPatientTurnDemo(turn);
    }

    if (!isPlaying) return;
    currentTurn++;
    await sleep(600);
    if (!isPlaying) return;
    playNextTurn();
}

async function playDoctorTurnDemo(turn) {
    if (!isPlaying) return;
    signVid.style.display = 'none';

    if (METHOD === 'sign') {
        statusBar.innerHTML = `
            <div class="doctor-script">
                <div class="script-label">Speaker says:</div>
                <div class="script-line">"${turn.text}"</div>
            </div>`;
        addMsg('doctor', SPEAKER_LABEL, turn.text);
        history.push({ speaker: 'doctor', text: turn.text });
        await sleep(Math.max(turn.text.split(' ').length * 400, 2000));
        return;
    }

    const hasSpeechRec = !!(window.SpeechRecognition || window.webkitSpeechRecognition);
    const hintText = hasSpeechRec
        ? '<span class="listening-indicator"></span>Listening...'
        : 'Tap here when done speaking';

    statusBar.innerHTML = `
        <div class="doctor-script">
            <div class="script-label">Read this line aloud:</div>
            <div class="script-line">"${turn.text}"</div>
            <div class="script-hint">${hintText}</div>
        </div>`;

    const recognized = await listenForSpeech(turn.text);
    const displayText = recognized.trim() || turn.text;
    addMsg('doctor', SPEAKER_LABEL, displayText);
    history.push({ speaker: 'doctor', text: turn.text });
}

// ══════════════════════════════════════════════════════════════
// LIVE MODE -- Free-form conversation
// ══════════════════════════════════════════════════════════════
let liveRecognition = null;
let liveActive = false;
let liveCollectedGlosses = [];
let isRecordingSign = false;
let signSegmenter = null;  // Phase 1+2: real-time sign segmenter
let poseOverlay = null;    // Real-time skeleton overlay

async function startLiveMode() {
    liveActive = true;
    await startCamera();

    const btnRecordSign = document.getElementById('btnRecordSign');
    const btnSendMessage = document.getElementById('btnSendMessage');

    if (METHOD === 'sign') {
        // Streaming live mode: WS pipeline auto-records continuously. The
        // legacy "Record Signs" button is hidden (it competed with the WS
        // pipeline by calling /api/process-signing-clip). The Send Message
        // button is rewired to the streaming-mode flush-and-pause flow.
        liveCollectedGlosses = [];

        if (btnRecordSign) {
            btnRecordSign.style.display = 'none';
            btnRecordSign.onclick = null;
        }
        if (btnSendMessage) {
            btnSendMessage.style.display = 'inline-block';
            btnSendMessage.textContent = 'Send Message';
            btnSendMessage.style.background = '#4caf80';
            btnSendMessage.disabled = false;
            btnSendMessage.onclick = () => sendNowAndPause();
        }

        statusBar.innerHTML = '<span class="processing">Sign when ready \u2014 your signs will appear here</span>';

        // (Settings cog is part of the camera toolbar and is always visible
        // once the live phase loads \u2014 no per-session show/hide needed.)

        // Start streaming pipeline (WS + server-side motion gate + inference).
        try {
            const seg = await ensureSegmenter();
            if (cameraStream) {
                seg.start(cameraStream, true);
                console.log('[Live] StreamingLiveMode started');
            }
        } catch (e) {
            console.warn('[Live] Segmenter init failed:', e.message);
        }
    } else {
        // Speaker mode
        const speakHint = SPEAK_MODE_TRANSLATE_TO_ASL
            ? 'Speak naturally \u2014 your words will be converted to ASL glosses'
            : 'Speak naturally \u2014 your words will appear as text for the signer';
        statusBar.innerHTML = `<span class="prompt"><span class="listening-indicator"></span>${speakHint}</span>`;
        startContinuousListeningForSpeaker();

        const btnSpeakSend = document.getElementById('btnSpeakSend');
        if (btnSpeakSend) {
            btnSpeakSend.style.display = 'inline-block';
            btnSpeakSend.onclick = () => sendSpeakerMessage();
        }
    }

    // Mode switch button removed — was covering the camera view
    kioskResetIdleTimer();
    // Kiosk / Lanyard: expose the Tap-to-Reply button for the speaker's
    // turn. No-op in any other SIGNBRIDGE_MODE.
    showTapReplyButton();
}

function stopLiveMode() {
    liveActive = false;
    continuousRecording = false;
    liveCollectedGlosses = [];
    isRecordingSign = false;
    speakerBuffer = '';
    if (signSegmenter) {
        signSegmenter.stop();
        signSegmenter = null;  // streaming mode needs a fresh instance per session
    }
    if (poseOverlay) {
        poseOverlay.stop();
    }
    if (liveRecognition) {
        try { liveRecognition.stop(); } catch(e) {}
        liveRecognition = null;
    }
    // Streaming-live: clear rolling caption + turn-tracking state.
    _setLiveCaption(null);
    _currentTurnRowEl = null;
    _currentTurnHistoryIndex = -1;
    _captionPending = false;
    _runningCaption = '';
    _lastSentGlossCount = 0;
    _lastCtqi = null;
    _prevCtqiWasHigh = true;   // next session's first low caption chimes
    _captionIsEdited = false;
    _lastRenderedCaptionHtml = '';
    _regenerateAttempts = 0;
    _hideZoneOverlay();
    // Close the settings panel if it was left open.
    const spanel = document.getElementById('settingsPanel');
    if (spanel) spanel.style.display = 'none';
    const btnSpeakSend = document.getElementById('btnSpeakSend');
    if (btnSpeakSend) btnSpeakSend.style.display = 'none';
    // Tear down Tap-to-Reply too.
    hideTapReplyButton();
}

// ── Speaker mode: speech recognition → English → ASL glosses ──
let speakerBuffer = '';

function startContinuousListeningForSpeaker() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        statusBar.innerHTML = '<span class="prompt">Speech recognition not available in this browser</span>';
        showToast('Speech recognition not available', 'warn');
        return;
    }

    liveRecognition = new SpeechRecognition();
    liveRecognition.lang = 'en-US';
    liveRecognition.interimResults = true;
    liveRecognition.continuous = true;
    liveRecognition.maxAlternatives = 1;

    liveRecognition.onresult = (event) => {
        let interim = '';
        let finalText = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript;
            if (event.results[i].isFinal) {
                finalText += transcript;
            } else {
                interim = transcript;
            }
        }

        // Show what the user is saying in real time
        const displayText = speakerBuffer + (speakerBuffer ? ' ' : '') + (finalText || interim);
        if (displayText.trim()) {
            statusBar.innerHTML = `<span class="processing"><span class="listening-indicator"></span>${displayText}</span>`;
        }

        if (finalText.trim()) {
            speakerBuffer += (speakerBuffer ? ' ' : '') + finalText.trim();
            // Show "Send to Signer" button prominently
            const btnSpeakSend = document.getElementById('btnSpeakSend');
            if (btnSpeakSend) btnSpeakSend.style.display = 'inline-block';
        }
    };

    liveRecognition.onend = () => {
        if (liveActive) {
            setTimeout(() => {
                if (liveActive) {
                    try { liveRecognition.start(); } catch(e) {}
                }
            }, 300);
        }
    };

    liveRecognition.onerror = (e) => {
        if (e.error === 'aborted') return;
    };

    try {
        liveRecognition.start();
    } catch(e) {
        console.error('[Speaker] Could not start recognition:', e);
    }
}

async function sendSpeakerMessage() {
    kioskResetIdleTimer();
    if (!speakerBuffer.trim()) {
        showToast('Say something first, then click Send', 'info');
        return;
    }

    const text = speakerBuffer.trim();
    speakerBuffer = '';

    const btnSpeakSend = document.getElementById('btnSpeakSend');
    btnSpeakSend.disabled = true;

    // Add speaker's message to transcript
    addMsg('doctor', 'Speaker', text);
    history.push({ speaker: 'doctor', text });

    if (SPEAK_MODE_TRANSLATE_TO_ASL) {
        // Full ASL translation: English → glosses → sign bank videos
        btnSpeakSend.innerHTML = '<span class="spinner"></span> Converting...';
        statusBar.innerHTML = '<span class="processing">Converting to ASL glosses... <span class="spinner"></span></span>';

        try {
            const resp = await fetch('/api/english-to-glosses', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text,
                    domain: selectedDomain,
                    conversation_history: history,
                }),
            });

            if (!resp.ok) {
                const errText = await resp.text().catch(() => '');
                throw new Error(`Server ${resp.status}: ${errText.substring(0, 200)}`);
            }

            const result = await resp.json();

            if (result.success && result.glosses.length > 0) {
                const glosses = result.glosses;
                const videos = result.sign_videos || {};

                statusBar.innerHTML = `<span class="gloss">ASL: ${glosses.join(' ')}</span>`;
                addMsg('patient', 'ASL Translation', glosses.join(' '));
                history.push({ speaker: 'patient', text: glosses.join(' ') });

                for (const gloss of glosses) {
                    const videoUrl = videos[gloss];
                    if (videoUrl && liveActive) {
                        await playSignVideo(videoUrl);
                        if (!liveActive) break;
                        signVid.style.display = 'none';
                        await sleep(200);
                    }
                }
            } else {
                statusBar.innerHTML = '<span class="sentence">Could not convert \u2014 no matching glosses found</span>';
                showToast(result.error || 'No glosses found for this sentence', 'warn');
            }
        } catch (e) {
            console.error('[Speaker] Error:', e);
            statusBar.innerHTML = '<span class="sentence">Error converting to ASL</span>';
            showToast('Failed to convert: ' + (e.message || 'check server logs'), 'error');
        }
    } else {
        // Speech-to-text only: deaf user reads English directly
        btnSpeakSend.innerHTML = '<span class="spinner"></span> Sending...';
        statusBar.innerHTML = `<span class="sentence">${text}</span>`;
    }

    btnSpeakSend.disabled = false;
    btnSpeakSend.innerHTML = '&#x1F399; Send to Signer';

    await sleep(2000);
    if (liveActive) {
        const hint = SPEAK_MODE_TRANSLATE_TO_ASL
            ? 'Speak naturally \u2014 your words will be converted to ASL glosses'
            : 'Speak naturally \u2014 your words will appear as text for the signer';
        statusBar.innerHTML = `<span class="prompt"><span class="listening-indicator"></span>${hint}</span>`;
    }
}

// ── Signer mode: continuous speech recognition for ambient speaker capture ──
function startContinuousListening() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        statusBar.innerHTML = '<span class="prompt">Speech recognition not available \u2014 speaker text won\'t be captured</span>';
        showToast('Speech recognition not available in this browser', 'warn');
        return;
    }

    liveRecognition = new SpeechRecognition();
    liveRecognition.lang = 'en-US';
    liveRecognition.interimResults = true;
    liveRecognition.continuous = true;
    liveRecognition.maxAlternatives = 1;

    liveRecognition.onresult = (event) => {
        let interim = '';
        let finalText = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript;
            if (event.results[i].isFinal) {
                finalText += transcript;
            } else {
                interim = transcript;
            }
        }

        if (!isRecordingSign) {
            if (interim) {
                statusBar.innerHTML = `<span class="processing"><span class="listening-indicator"></span>${interim}</span>`;
            }
        }

        if (finalText.trim()) {
            const text = finalText.trim();
            addMsg('doctor', SPEAKER_LABEL, text);
            history.push({ speaker: 'doctor', text: text });

            if (!isRecordingSign) {
                statusBar.innerHTML = `
                    <div class="doctor-script">
                        <div class="script-label">Speaker said:</div>
                        <div class="script-line">"${text}"</div>
                    </div>`;
                setTimeout(() => {
                    if (liveActive && !isRecordingSign) {
                        statusBar.innerHTML = '<span class="prompt"><span class="listening-indicator"></span>Listening for speaker...</span>';
                    }
                }, 4000);
            }
        }
    };

    liveRecognition.onend = () => {
        if (liveActive) {
            setTimeout(() => {
                if (liveActive) {
                    try { liveRecognition.start(); } catch(e) {}
                }
            }, 300);
        }
    };

    liveRecognition.onerror = (e) => {
        if (e.error === 'aborted') return;
    };

    try {
        liveRecognition.start();
    } catch(e) {
        console.error('[Live] Could not start recognition:', e);
    }
}

// ── Recording with segmentation (Phase 1+2) ──
let continuousRecording = false;
let pendingSend = false;
let recCountdownInterval = null;

function updateRecCountdown(remaining) {
    const el = document.getElementById('recCountdown');
    if (!el) return;
    if (remaining > 0) {
        el.classList.add('on');
        el.innerHTML = `<span class="rec-dot"></span> ${(remaining / 1000).toFixed(1)}s`;
    } else {
        el.classList.remove('on');
    }
}

// ============================================================================
// STREAMING LIVE MODE -- caption overlay + rewrite-in-place transcript (option C)
// ----------------------------------------------------------------------------
// The wrist-velocity SignSegmenter has been replaced with the WebSocket-driven
// StreamingLiveMode (server-side motion gate + pose + OpenHands inference).
// The old SignSegmenter / PoseOverlay / pose_player code is kept on disk for
// revert; it is no longer called from the live path.
// ============================================================================

let _currentTurnRowEl = null;
let _currentTurnHistoryIndex = -1;
let _captionPending = false;
let _sendInProgress = false;   // true while sendNowAndPause is running

// Running-caption (extend mode) state. Reset on Send Message and on Restart.
// _runningCaption: the most recent LLM output for the in-flight utterance.
// _lastSentGlossCount: how many of liveCollectedGlosses have already been
//   handed to the LLM; the next call sends only glosses[_lastSentGlossCount:].
let _runningCaption = '';
let _lastSentGlossCount = 0;
let _lastCtqi = null;                // most recently computed CTQI (for re-render on cancel-edit)
// CTQI_LOW_THRESHOLD lives at top-of-file globals (TDZ avoidance, +
// user-tunable from Settings tab).

// Chime state lives at the top of the file (TDZ avoidance for the
// page-load init block). Helpers below; edge-triggered firing happens
// from _renderAndShowCaption.
function updateLowCtqiChime() {
    const cb = document.getElementById('lowCtqiChimeToggle');
    if (!cb) return;
    CHIME_ON_LOW_CTQI = !!cb.checked;
    localStorage.setItem('signbridge.lowCtqiChime', CHIME_ON_LOW_CTQI ? '1' : '0');
    const mirror = document.getElementById('setChimeEnabled');
    if (mirror) mirror.checked = CHIME_ON_LOW_CTQI;
}

function syncLowCtqiChimeUI() {
    const cb = document.getElementById('lowCtqiChimeToggle');
    if (cb) cb.checked = CHIME_ON_LOW_CTQI;
}

function _playLowCtqiChime() {
    if (!CHIME_ON_LOW_CTQI) return;
    try {
        // Lazy-init the AudioContext on first chime -- creating it
        // pre-emptively can throw in browsers that require a user
        // gesture (Chrome autoplay policy). By the time we get here a
        // session is well underway, so a gesture has already happened.
        if (!_chimeAudioCtx) {
            const Ctx = window.AudioContext || window.webkitAudioContext;
            if (!Ctx) return;
            _chimeAudioCtx = new Ctx();
        }
        const ctx = _chimeAudioCtx;
        if (ctx.state === 'suspended') ctx.resume();
        const now = ctx.currentTime;
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        // Two short pips (descending): E5 -> C5. Distinguishable from a
        // notification ding, short enough to not disrupt conversation.
        osc.type = 'sine';
        osc.frequency.setValueAtTime(659.25, now);             // E5
        osc.frequency.setValueAtTime(523.25, now + 0.12);      // C5
        gain.gain.setValueAtTime(0.0001, now);
        gain.gain.exponentialRampToValueAtTime(0.18, now + 0.02);
        gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.30);
        osc.connect(gain).connect(ctx.destination);
        osc.start(now);
        osc.stop(now + 0.32);
    } catch (e) {
        console.warn('[Chime] play failed:', e);
    }
}
let _captionIsEdited = false;        // user has manually edited the caption
let _lastRenderedCaptionHtml = '';   // cached for cancel-edit restoration
let _regenerateAttempts = 0;         // resets per turn; capped at REGENERATE_LIMIT
const REGENERATE_LIMIT = 3;

function _liveCaptionEl() { return document.getElementById('liveCaption'); }

// Position the signing-zone overlay lines on the camera view to match the
// server's effective config. Called when a config arrives from the WS server
// (initially after start, and after every tune() reply).
function _updateZoneOverlay(zoneTop, zoneBottom) {
    const lineTop = document.getElementById('liveZoneLineTop');
    const lineBot = document.getElementById('liveZoneLineBottom');
    if (lineTop) {
        if (zoneTop != null && zoneTop > 0) {
            lineTop.style.top = (zoneTop * 100).toFixed(1) + '%';
            lineTop.style.display = 'block';
        } else {
            lineTop.style.display = 'none';   // top is 0; no line to draw
        }
    }
    if (lineBot && zoneBottom != null) {
        lineBot.style.top = (zoneBottom * 100).toFixed(1) + '%';
        lineBot.style.display = (zoneBottom < 1.0) ? 'block' : 'none';
    }
    // Mirror values into the sliders (without re-firing the input handler).
    if (zoneTop != null) {
        const slider = document.getElementById('streamZoneTop');
        const label  = document.getElementById('streamZoneTopLabel');
        if (slider) slider.value = Math.round(zoneTop * 100);
        if (label)  label.textContent = Math.round(zoneTop * 100) + '%';
    }
    if (zoneBottom != null) {
        const slider = document.getElementById('streamZoneBottom');
        const label  = document.getElementById('streamZoneBottomLabel');
        if (slider) slider.value = Math.round(zoneBottom * 100);
        if (label)  label.textContent = Math.round(zoneBottom * 100) + '%';
    }
}

function _hideZoneOverlay() {
    const lineTop = document.getElementById('liveZoneLineTop');
    const lineBot = document.getElementById('liveZoneLineBottom');
    if (lineTop) lineTop.style.display = 'none';
    if (lineBot) lineBot.style.display = 'none';
}

// Dead since the camera-source and zone-settings panels were merged into the
// unified `settingsPanel` (HTML now wires onclick="toggleSettings()"). The
// older `streamSettingsPanel` and `cameraSourcePanel` ids no longer exist.
// Function kept as a thin alias so any latent caller still works.
function toggleStreamSettings() { toggleSettings(); }

// Slider handler — pushes the new zone to the server via tune(). The server
// echoes back an ack-control with the effective config, which lands in
// onConfig and updates the line position. We also nudge the line locally so
// the visual lags by 0ms instead of one round-trip.
function updateStreamZone() {
    const top    = parseInt(document.getElementById('streamZoneTop').value) / 100;
    const bottom = parseInt(document.getElementById('streamZoneBottom').value) / 100;
    _updateZoneOverlay(top, bottom);
    if (signSegmenter && typeof signSegmenter.tune === 'function') {
        signSegmenter.tune({ zone_top: top, zone_bottom: bottom });
    }
}

function _setLiveCaption(text, opts) {
    const el = _liveCaptionEl();
    if (!el) return;
    if (text == null || text === '') {
        el.style.display = 'none';
        el.innerHTML = '';
        el.classList.remove('updating');
        el.classList.remove('low-ctqi');
        return;
    }
    el.style.display = 'block';
    if (opts && opts.html) el.innerHTML = text;
    else                   el.textContent = text;
    if (opts && opts.updating) el.classList.add('updating');
    else el.classList.remove('updating');
}

function _escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, c => (
        { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
    ));
}

// Render caption HTML: escaped sentence + CTQI badge + (when CTQI is low or
// missing and not currently editing) the regenerate + edit action chips.
// Caches the result so cancel-edit can restore.
function _renderCaptionHtml(sentence, ctqi) {
    // Hard floor: below this CTQI the model is essentially guessing.
    // Hide the constructed sentence and show "[unclear]" so we don't
    // present a confidently-wrong caption. User can still hit Edit to
    // manually type what they meant. Doesn't apply to manually-edited
    // captions (the edit is ground truth).
    const belowHardFloor = !_captionIsEdited && ctqi != null && ctqi < CTQI_HARD_FLOOR;
    const displaySentence = belowHardFloor ? '[unclear]' : sentence;
    const escaped = _escapeHtml(displaySentence);
    let scoreHtml;
    if (_captionIsEdited) {
        scoreHtml = `<span class="ctqi-edited">(edited)</span>`;
    } else if (ctqi != null) {
        const cls = ctqi >= CTQI_LOW_THRESHOLD ? 'ctqi-good' : 'ctqi-bad';
        scoreHtml = `<span class="${cls}">(CTQI = ${ctqi.toFixed(0)})</span>`;
    } else {
        scoreHtml = `<span class="ctqi-missing">(CTQI = ?)</span>`;
    }
    // Action chips when CTQI is low or missing, but not while editing.
    let actionsHtml = '';
    const showActions = !_captionIsEdited && (ctqi == null || ctqi < CTQI_LOW_THRESHOLD);
    if (showActions) {
        const regenLeft = REGENERATE_LIMIT - _regenerateAttempts;
        if (regenLeft > 0) {
            const tooltip = `Regenerate translation (${regenLeft} left)`;
            actionsHtml += `<span class="caption-action" data-action="regenerate" title="${tooltip}">🔄</span>`;
        } else {
            // Regenerate cap reached — gentle hint that Edit is the path forward.
            actionsHtml += `<span class="caption-action caption-action-disabled" title="Out of retries — try Edit">🔄</span>`;
        }
        actionsHtml += `<span class="caption-action" data-action="edit" title="Edit caption manually">✏️</span>`;
    }
    return `${escaped}${scoreHtml}${actionsHtml}`;
}

function _renderAndShowCaption(sentence, ctqi) {
    const html = _renderCaptionHtml(sentence, ctqi);
    _lastRenderedCaptionHtml = html;
    _lastCtqi = ctqi;
    _setLiveCaption(html, { updating: false, html: true });
    // Low-CTQI alert: pulsing red border on the banner. Skip when the
    // user has manually edited the caption -- their edit is the ground
    // truth, no need to nag.
    const el = _liveCaptionEl();
    const isLow = !_captionIsEdited && ctqi != null && ctqi < CTQI_LOW_THRESHOLD;
    if (el) el.classList.toggle('low-ctqi', isLow);
    // Edge-triggered chime: only on a high->low (or missing->low)
    // transition. Updates the "previous" flag whether we chime or not.
    if (isLow && _prevCtqiWasHigh) {
        _playLowCtqiChime();
    }
    _prevCtqiWasHigh = !isLow;
    _initCaptionInteractions();
}

// One-time event-delegation wiring for the caption banner. Subsequent
// innerHTML rewrites are fine because the listener is on the parent element.
function _initCaptionInteractions() {
    const el = _liveCaptionEl();
    if (!el || el.dataset.interactionsInit === '1') return;
    el.addEventListener('click', (e) => {
        const target = e.target.closest('[data-action]');
        if (!target) return;
        const action = target.getAttribute('data-action');
        if (action === 'regenerate')        _regenerateCaption();
        else if (action === 'edit')         _enterCaptionEditMode();
        else if (action === 'edit-confirm') _confirmCaptionEdit();
        else if (action === 'edit-cancel')  _cancelCaptionEdit();
    });
    el.dataset.interactionsInit = '1';
}

// Regenerate: re-run the LLM with the SAME accumulated glosses but in fresh
// mode + a "previous attempt was rejected" hint, so it produces a different
// interpretation. Used when the user clicks 🔄 on a low-CTQI caption.
async function _regenerateCaption() {
    if (_captionPending) return;
    if (!liveActive || !signSegmenter) return;
    if (_regenerateAttempts >= REGENERATE_LIMIT) {
        // Out of retries — auto-open edit so the user has a clear path forward.
        _enterCaptionEditMode();
        return;
    }
    const allGlosses = signSegmenter.getCollectedGlosses();
    if (allGlosses.length === 0) return;

    _captionPending = true;
    _regenerateAttempts += 1;
    const rejectedAttempt = _runningCaption || '';

    statusBar.innerHTML = '<span class="processing ai"><span class="ai-sparkle">✨</span>Regenerating... <span class="spinner"></span></span>';
    if (_liveCaptionEl()) _liveCaptionEl().classList.add('updating');

    let sentence = allGlosses.map(p => p.gloss).join(' ');
    let plausibility = null;
    try {
        const resp = await fetch('/api/construct-sentence-live', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                gloss_predictions: allGlosses,
                running_caption: '',            // fresh build, not extend
                previous_attempt: rejectedAttempt,
                conversation_history: history,
                domain: selectedDomain,
            }),
        });
        const result = await resp.json();
        if (result.success && result.sentence) sentence = result.sentence.trim();
        if (typeof result.plausibility === 'number') plausibility = result.plausibility;
    } catch (e) {
        console.warn('[Live] Regenerate error:', e.message);
    }

    _runningCaption = sentence;
    _lastSentGlossCount = allGlosses.length;
    _captionIsEdited = false;

    let ctqi = null;
    if (plausibility != null && allGlosses.length > 0) {
        const avgConf = allGlosses.reduce((s, g) => s + (g.confidence || 0), 0) / allGlosses.length;
        ctqi = avgConf * 100 * (0.5 + 0.5 * plausibility / 100);
    }
    _renderAndShowCaption(sentence, ctqi);
    statusBar.innerHTML = _statusForIdle();
    _captionPending = false;
}

// Edit mode: replace caption content with an input and confirm/cancel chips.
// On confirm, the user's text becomes the running caption (and is marked
// "edited" so the next LLM call extends from their text, not the LLM's).
function _enterCaptionEditMode() {
    if (_captionPending) return;
    const el = _liveCaptionEl();
    if (!el) return;
    const current = _runningCaption || '';
    el.innerHTML =
        `<input class="caption-edit-input" type="text" value="${_escapeHtml(current)}">` +
        `<span class="caption-action" data-action="edit-confirm" title="Save (Enter)">✓</span>` +
        `<span class="caption-action" data-action="edit-cancel" title="Cancel (Esc)">✗</span>`;
    const input = el.querySelector('.caption-edit-input');
    if (input) {
        input.focus();
        input.select();
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter')       { e.preventDefault(); _confirmCaptionEdit(); }
            else if (e.key === 'Escape') { e.preventDefault(); _cancelCaptionEdit();  }
        });
    }
}

function _confirmCaptionEdit() {
    const el = _liveCaptionEl();
    if (!el) return;
    const input = el.querySelector('.caption-edit-input');
    if (!input) return;
    const edited = (input.value || '').trim();
    if (!edited) { _cancelCaptionEdit(); return; }
    _runningCaption = edited;
    _captionIsEdited = true;
    // Edited caption: no CTQI (LLM didn't produce this); show "(edited)" badge.
    _renderAndShowCaption(edited, null);
}

function _cancelCaptionEdit() {
    // Restore whatever was last rendered (sentence + CTQI badge + chips).
    if (_lastRenderedCaptionHtml) {
        _setLiveCaption(_lastRenderedCaptionHtml, { updating: false, html: true });
    } else if (_runningCaption) {
        _renderAndShowCaption(_runningCaption, _lastCtqi);
    } else {
        _setLiveCaption(null);
    }
}

// Compact top-K format: "WORD 87% (ALT1, ALT2)" -- top-1 with confidence,
// 2nd/3rd in parens without confidence. Full top-K with confidences still
// goes to the LLM payload, so prompt guidance is unaffected.
function _formatGlossWithTopK(pred) {
    const conf = Math.round((pred.confidence || 0) * 100);
    const alts = (pred.top_k || []).slice(1, 3)
        .map(t => (t.gloss || '').toUpperCase())
        .filter(Boolean);
    const main = `${(pred.gloss || '').toUpperCase()} ${conf}%`;
    return alts.length ? `${main} (${alts.join(', ')})` : main;
}

// Triggered every glossBatchSize new glosses OR every idleSendMs idle.
// The configurable trigger lives in streaming_live.js. This function owns
// the rewrite-in-place caption AND the rewrite-in-place transcript row.
async function updateRollingCaption(glosses) {
    if (!glosses || glosses.length === 0) return;
    if (_captionPending) return;

    // Extend mode: only send glosses NEW since the last LLM call. The server
    // gets the running caption + new glosses and extends rather than rebuilds.
    const newGlosses = glosses.slice(_lastSentGlossCount);
    if (newGlosses.length === 0) return;   // nothing fresh to send

    _captionPending = true;
    statusBar.innerHTML = '<span class="processing ai"><span class="ai-sparkle">✨</span>Constructing sentence... <span class="spinner"></span></span>';
    if (_liveCaptionEl()) _liveCaptionEl().classList.add('updating');

    let sentence = _runningCaption || newGlosses.map(p => p.gloss).join(' ');
    let plausibility = null;
    try {
        const resp = await fetch('/api/construct-sentence-live', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                gloss_predictions: newGlosses,
                running_caption: _runningCaption,
                conversation_history: history,
                domain: selectedDomain,
            }),
        });
        const result = await resp.json();
        if (result.success && result.sentence) {
            sentence = result.sentence.trim();
        } else if (result.fallback) {
            sentence = result.fallback;
        }
        if (typeof result.plausibility === 'number') {
            plausibility = result.plausibility;
        }
    } catch (e) {
        console.warn('[Live] LLM error:', e.message);
    }

    // Persist running-caption state so the next LLM call extends from here.
    _runningCaption = sentence;
    _lastSentGlossCount = glosses.length;

    // Live CTQI estimate. Real CTQI needs ground truth (Gloss Accuracy +
    // Coverage F1) which we don't have in live mode. We approximate:
    //   GA  = mean top-1 confidence across all glosses in this utterance
    //   CF1 = 1.0 (no reference; can't measure semantic preservation)
    //   P   = LLM's self-critique plausibility (geometric mean of the
    //         grammatical/semantic/naturalness sub-scores)
    //
    // Formula matches CTQI v2:  CTQI = (GA/100) * (CF1/100) * (0.5 + 0.5*P/100) * 100
    let ctqi = null;
    if (plausibility != null && glosses.length > 0) {
        const avgConf = glosses.reduce((s, g) => s + (g.confidence || 0), 0) / glosses.length;
        const ga = avgConf * 100;
        ctqi = ga * (0.5 + 0.5 * plausibility / 100);
    }
    // LLM call produced a new caption — it overrides any prior manual edit.
    _captionIsEdited = false;
    _renderAndShowCaption(sentence, ctqi);
    statusBar.innerHTML = _statusForIdle();

    // NOTE: the transcript / conversation panel is *not* updated here. The
    // running caption is the in-flight working copy of the LLM's interpretation;
    // only on "Send Message" (sendNowAndPause) does it get committed to the
    // transcript + history. If the user doesn't like what the LLM produced,
    // they can click Restart to abandon this utterance.

    _captionPending = false;
}

// Default status-bar HTML when no transient state (signing/recognizing/
// constructing/sending) is in force. Slight variation if there are
// uncommitted glosses pending Send.
function _statusForIdle() {
    if (liveCollectedGlosses && liveCollectedGlosses.length > 0) {
        return '<span class="processing">Keep signing or click <b>Send Message</b></span>';
    }
    return '<span class="processing">Sign when ready — your signs will appear here</span>';
}

// "Send Message" handler -- explicit flush + pause flow for I-SIGN mode.
//   1. Pause frame send so no new segments open.
//   2. Wait for any in-flight server-side segments to resolve to gloss events
//      so they make it into the LLM payload.
//   3. Wait for any in-flight LLM call to finish.
//   4. Fire one final LLM call with the full accumulated gloss buffer.
//   5. Speak the resulting sentence (in-person / speak-method).
//   6. Lock in the turn (next user gloss starts a new transcript row even
//      without a doctor message in between).
//   7. Flip the button to "Resume Signing" — detection stays paused until
//      the user clicks it.
async function sendNowAndPause() {
    if (!liveActive) return;

    // Kiosk/Lanyard: re-arm the wake-word listener inside this click's
    // user-gesture window. Use deferred start (300ms) -- synchronous
    // restart-after-abort silently fails in Chrome (new SR session
    // never receives audio). The user-activation context survives the
    // 300ms delay in practice.
    if (SIGNBRIDGE_MODE === 'kiosk' || SIGNBRIDGE_MODE === 'lanyard') {
        try {
            stopWakeWordListener();
            _armWakeWordDeferred('send');
        } catch (e) { console.warn('[Send] wake re-arm failed:', e); }
    }

    _sendInProgress = true;   // suppresses status-bar rewrites from state ticks
    const btn = document.getElementById('btnSendMessage');

    if (btn) {
        btn.disabled = true;
        btn.textContent = 'Sending...';
    }

    // 1. Pause frame send.
    if (signSegmenter && typeof signSegmenter.pause === 'function') {
        signSegmenter.pause();
    }
    statusBar.innerHTML = '<span class="processing ai"><span class="ai-sparkle">✨</span>Sending message... <span class="spinner"></span></span>';

    // 2. Wait (up to 8s) for in-flight server segments to drain into glosses.
    const start = Date.now();
    const timeoutMs = 8000;
    while (signSegmenter && typeof signSegmenter.inFlightCount === 'function'
           && signSegmenter.inFlightCount() > 0
           && (Date.now() - start) < timeoutMs) {
        await sleep(150);
    }

    // 3. Wait for any in-flight LLM call to finish.
    while (_captionPending && (Date.now() - start) < timeoutMs) {
        await sleep(150);
    }

    // 4. Final LLM flush with everything collected. updateRollingCaption only
    //    updates the in-flight caption; commit-to-transcript happens here.
    const glosses = signSegmenter ? signSegmenter.getCollectedGlosses() : liveCollectedGlosses.slice();
    if (glosses.length > 0) {
        await updateRollingCaption(glosses);
    }

    // 5. Commit the latest LLM-constructed sentence to the transcript + history.
    //    Use _runningCaption (raw LLM output, no CTQI suffix) rather than
    //    reading the caption banner DOM — the banner includes "(CTQI = X)"
    //    which we don't want in the permanent transcript.
    const finalText = (_runningCaption || _liveCaptionEl()?.textContent || '').trim();
    if (finalText) {
        addMsg('patient', SIGNER_LABEL, finalText);
        history.push({ speaker: 'patient', text: finalText });
        if (interactionMode === 'in-person' || METHOD === 'speak') {
            try { await speak(finalText); } catch (e) { console.warn('[Live] speak failed:', e); }
        }
    }

    // 6. Clear all in-flight state. The next turn starts fresh.
    _currentTurnRowEl = null;
    _currentTurnHistoryIndex = -1;
    _runningCaption = '';
    _lastSentGlossCount = 0;
    _lastCtqi = null;
    _captionIsEdited = false;
    _lastRenderedCaptionHtml = '';
    _regenerateAttempts = 0;
    if (signSegmenter && typeof signSegmenter.clearGlosses === 'function') {
        signSegmenter.clearGlosses();
    }
    liveCollectedGlosses = [];
    _clearGlossRows();
    _setLiveCaption(null);   // clear the caption banner after commit

    // 7. Flip button to Resume.
    if (btn) {
        btn.disabled = false;
        btn.textContent = 'Resume Signing';
        btn.style.background = '#b07cf0';
        btn.onclick = () => resumeSigning();
    }
    statusBar.innerHTML = '<span class="prompt">Message sent. Click <b>Resume Signing</b> to continue.</span>';
    _sendInProgress = false;
}

function resumeSigning() {
    if (!liveActive) return;
    if (signSegmenter && typeof signSegmenter.resume === 'function') {
        signSegmenter.resume();
    }
    const btn = document.getElementById('btnSendMessage');
    if (btn) {
        btn.textContent = 'Send Message';
        btn.style.background = '#4caf80';
        btn.onclick = () => sendNowAndPause();
    }
    statusBar.innerHTML = '<span class="processing">Sign when ready — your signs will appear here</span>';
}

// Initialize the sign engine (called once when first needed). Variable name
// `signSegmenter` is kept for compatibility with the rest of the file; it now
// holds a StreamingLiveMode instance instead of a SignSegmenter instance.
async function ensureSegmenter() {
    if (signSegmenter) return signSegmenter;

    // Read current slider values so a zone tuned before Start Conversation
    // is honored by the server when the WS opens.
    const _zTopEl = document.getElementById('streamZoneTop');
    const _zBotEl = document.getElementById('streamZoneBottom');
    const _zTop = _zTopEl ? parseInt(_zTopEl.value) / 100 : null;
    const _zBot = _zBotEl ? parseInt(_zBotEl.value) / 100 : null;

    signSegmenter = new StreamingLiveMode({
        domain: selectedDomain,
        zoneTop: _zTop,
        zoneBottom: _zBot,
        // glossBatchSize defaults to 3. Override at runtime via
        // ?glossBatchSize=N or localStorage signbridge.glossBatchSize.
        // (Idle timer removed -- Send Message is the only manual flush.)
    });

    const videoEl = document.getElementById('cameraFeed');
    await signSegmenter.init(videoEl);

    // Compact gloss event from the WS pipeline. Receives EVERY prediction
    // (confident or not). The diagnostic panel shows all of them so the user
    // can see what the model is hearing; only confident ones are reflected
    // in liveCollectedGlosses (the LLM-bound buffer).
    signSegmenter.onGloss = (pred, collected) => {
        if (!liveActive) return;
        if (collected) {
            liveCollectedGlosses = signSegmenter.getCollectedGlosses();
        }
        updateDebugPanel({ signs: liveCollectedGlosses.length });
        _appendGlossRow(pred);
    };

    // State adapter -- handles 'signing' | 'analyzing' | 'idle'.
    // The status bar tracks the *latest* state event. While the LLM is in
    // flight, updateRollingCaption temporarily overwrites this to
    // "Constructing..." and restores it via _statusForIdle when done.
    signSegmenter.onStateChange = (state) => {
        if (!liveActive) return;
        updateDebugPanel({ state });
        // Don't clobber Send/Resume/Constructing messages with state ticks.
        if (_captionPending || _sendInProgress) return;
        // Also don't clobber the speaker-turn "Listening / Processing"
        // status while wake-word capture is in progress. State ticks
        // come at ~15fps from the streamer; without this guard they
        // overwrite the speaker-turn indicator within 67ms of it being
        // set, making it invisible to the user.
        if (_wakeCapturing) return;
        switch (state) {
            case 'signing':
                recBadge.classList.add('on');
                recBorder.classList.add('on');
                statusBar.innerHTML = '<span class="processing">Signing...</span>';
                break;
            case 'analyzing':
                recBadge.classList.remove('on');
                recBorder.classList.remove('on');
                statusBar.innerHTML = '<span class="processing ai"><span class="ai-sparkle">✨</span>Recognizing... <span class="spinner"></span></span>';
                break;
            case 'idle':
            default:
                recBadge.classList.remove('on');
                recBorder.classList.remove('on');
                statusBar.innerHTML = _statusForIdle();
                break;
        }
    };

    // LLM trigger: streaming_live.js fires every glossBatchSize new glosses
    // OR every idleSendMs idle. Rewrite caption + transcript row in place;
    // gloss buffer is NOT cleared (LLM gets a rolling context window).
    signSegmenter.onReadyToSend = async (glosses) => {
        if (!liveActive || glosses.length === 0) return;
        await updateRollingCaption(glosses);
    };

    // Server echoes its effective config after start/tune; sync the zone
    // overlay + sliders so the user always sees what's actually in force.
    signSegmenter.onConfig = (cfg) => {
        if (cfg && (cfg.zone_top != null || cfg.zone_bottom != null)) {
            _updateZoneOverlay(cfg.zone_top, cfg.zone_bottom);
        }
    };


    console.log('[Live] StreamingLiveMode ready (domain=' + selectedDomain +
        ', batch=' + signSegmenter.glossBatchSize + ')');
    return signSegmenter;
}

function togglePoseOverlay() {
    if (!poseOverlay) return false;
    const visible = poseOverlay.toggle();
    const btn = document.getElementById('poseOverlayBtn');
    if (btn) {
        btn.classList.toggle('active', visible);
        btn.title = visible ? 'Hide pose overlay' : 'Show pose overlay';
    }
    localStorage.setItem('signbridge-pose-overlay', visible ? '1' : '0');
    return visible;
}

// ── Continuous sign capture via HybridSegmenter ──
// User taps Record, signs naturally for N seconds, server auto-segments and
// returns multiple gloss predictions per chunk.
let signingActive = false;
const CHUNK_RECORD_DURATION = 8000;   // seconds per chunk sent to server
const SENTENCE_EVERY_N = 3;

async function tapToSign() {
    const btnRecordSign = document.getElementById('btnRecordSign');
    const btnSendMessage = document.getElementById('btnSendMessage');

    if (signingActive) {
        signingActive = false;
        btnRecordSign.textContent = 'Stopping...';
        return;
    }

    signingActive = true;
    liveCollectedGlosses = [];
    btnRecordSign.textContent = 'Stop Signing';
    btnRecordSign.style.background = '#e74c3c';
    if (btnSendMessage) btnSendMessage.style.display = 'none';

    let glossesSinceLastSentence = 0;

    while (signingActive && liveActive) {
        const detected = liveCollectedGlosses.map(p => p.gloss.toUpperCase()).join(' ');

        recBadge.classList.add('on');
        recBorder.classList.add('on');
        statusBar.innerHTML = (detected
            ? `<span class="gloss">${detected}</span> <span class="processing" style="font-size:0.8em">\u2014 sign freely (${CHUNK_RECORD_DURATION/1000}s chunks auto-segmented)</span>`
            : `<span class="processing">Sign freely \u2014 server auto-detects each sign every ${CHUNK_RECORD_DURATION/1000}s</span>`);

        const blob = await recordFromCamera(CHUNK_RECORD_DURATION);
        recBadge.classList.remove('on');
        recBorder.classList.remove('on');
        if (!signingActive || !blob) break;

        statusBar.innerHTML = (detected
            ? `<span class="gloss">${detected}</span> <span class="processing" style="font-size:0.8em"><span class="spinner"></span> segmenting + recognizing...</span>`
            : '<span class="processing">Segmenting + recognizing... <span class="spinner"></span></span>');

        try {
            const formData = new FormData();
            formData.append('video', blob, 'chunk.webm');
            formData.append('domain', selectedDomain);

            const resp = await fetch('/api/process-signing-clip', { method: 'POST', body: formData });
            const result = await resp.json();
            // Always update the UI with results (even if user hit Stop while we were processing)

            if (result.success && result.glosses && result.glosses.length > 0) {
                for (const g of result.glosses) {
                    if (!g.confident) continue;   // skip unclear predictions per Stage 5
                    const isDup = liveCollectedGlosses.length > 0 &&
                        liveCollectedGlosses[liveCollectedGlosses.length - 1].gloss.toLowerCase() === g.gloss.toLowerCase();
                    if (isDup) continue;
                    liveCollectedGlosses.push({
                        gloss: g.gloss,
                        confidence: g.confidence,
                        top_k: g.top_k || []
                    });
                    glossesSinceLastSentence++;
                }
                const allDetected = liveCollectedGlosses.map(p => p.gloss.toUpperCase()).join(' ');
                statusBar.innerHTML = `<span class="gloss" style="font-size:1.1em">${allDetected || '(no confident signs in chunk)'}</span>`;

                // Auto-construct sentence every N confident glosses
                if (glossesSinceLastSentence >= SENTENCE_EVERY_N) {
                    glossesSinceLastSentence = 0;
                    statusBar.innerHTML = `<span class="gloss">${allDetected}</span> <span class="processing"><span class="spinner"></span></span>`;
                    try {
                        const sentResp = await fetch('/api/construct-sentence-live', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                gloss_predictions: liveCollectedGlosses,
                                conversation_history: history
                            })
                        });
                        const sentResult = await sentResp.json();
                        if (sentResult.success && sentResult.sentence) {
                            const sentence = sentResult.sentence.trim();
                            statusBar.innerHTML = `<span class="sentence">${sentence}</span>`;
                            addMsg('patient', SIGNER_LABEL, sentence);
                            history.push({ speaker: 'patient', text: sentence });
                            if (interactionMode === 'in-person' || METHOD === 'speak') {
                                await speak(sentence);
                            }
                            liveCollectedGlosses = [];
                        }
                    } catch (e) {
                        console.error('[ChunkMode] LLM error:', e);
                    }
                }
            } else if (result.success) {
                statusBar.innerHTML = (detected
                    ? `<span class="gloss">${detected}</span> <span class="processing" style="font-size:0.8em" style="color:#ff9800">no signs detected in last chunk</span>`
                    : '<span class="processing" style="color:#ff9800">No signs detected \u2014 sign more clearly</span>');
            } else {
                statusBar.innerHTML = `<span class="processing" style="color:#ff6666">${result.error || 'Processing failed'}</span>`;
            }
        } catch (e) {
            console.error('[ChunkMode] Error:', e);
            if (signingActive) {
                showToast('Processing failed \u2014 is the inference API running?', 'error');
                break;
            }
        }

        if (signingActive) await sleep(300);
    }

    recBadge.classList.remove('on');
    recBorder.classList.remove('on');
    signingActive = false;
    btnRecordSign.textContent = 'Record Signs';
    btnRecordSign.style.background = '#b07cf0';
    btnRecordSign.disabled = false;

    if (liveCollectedGlosses.length > 0) {
        const allDetected = liveCollectedGlosses.map(p => p.gloss.toUpperCase()).join(' ');
        statusBar.innerHTML = `<span class="gloss">Signs: ${allDetected}</span>`;
        if (btnSendMessage) btnSendMessage.style.display = 'inline-block';
    } else {
        statusBar.innerHTML = '<span class="processing">Tap "Record Signs" to start signing</span>';
    }
    kioskResetIdleTimer();
}

// toggleSignRecording is kept for legacy fallback mode only
async function toggleSignRecording() {
    await toggleSignRecordingLegacy();
}

async function sendSignMessage() {
    if (liveCollectedGlosses.length === 0) return;
    kioskResetIdleTimer();

    // Pause segmenter during send
    if (signSegmenter) signSegmenter.stop();
    continuousRecording = false;
    isRecordingSign = false;

    const btnSendMessage = document.getElementById('btnSendMessage');
    if (btnSendMessage) {
        btnSendMessage.disabled = true;
        btnSendMessage.textContent = 'Sending...';
    }

    statusBar.innerHTML = '<span class="processing">Constructing sentence... <span class="spinner"></span></span>';

    let sentence = liveCollectedGlosses.map(p => p.gloss).join(' ');
    try {
        const resp = await fetch('/api/construct-sentence-live', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                gloss_predictions: liveCollectedGlosses,
                conversation_history: history
            })
        });
        const result = await resp.json();
        if (result.success && result.sentence) {
            sentence = result.sentence.trim();
        } else if (result.fallback) {
            sentence = result.fallback;
        }
    } catch (e) {
        console.error('[Live] LLM fetch error:', e);
        showToast('LLM unavailable, using raw glosses', 'warn');
        await sleep(500);
    }

    statusBar.innerHTML = '<span class="sentence">' + sentence + '</span>';
    addMsg('patient', SIGNER_LABEL, sentence);
    history.push({ speaker: 'patient', text: sentence });

    if (interactionMode === 'in-person' || METHOD === 'speak') {
        await speak(sentence);
    }

    // Reset glosses
    liveCollectedGlosses = [];
    if (signSegmenter) signSegmenter.clearGlosses();

    // Reset Send button
    if (btnSendMessage) {
        btnSendMessage.disabled = false;
        btnSendMessage.textContent = 'Send Now \u27A4';
        btnSendMessage.style.display = 'none';
    }

    // Brief pause then listen for speaker, then auto-restart segmenter
    await sleep(1500);
    if (liveActive) {
        statusBar.innerHTML = '<span class="prompt"><span class="listening-indicator"></span>Listening for speaker...</span>';
        // After a moment, re-enable signing detection
        await sleep(2000);
        if (liveActive) {
            statusBar.innerHTML = '<span class="processing">Sign when ready \u2014 your signs will appear here</span>';
            continuousRecording = true;
            isRecordingSign = true;
            if (signSegmenter && cameraStream) {
                signSegmenter.start(cameraStream);
            }
        }
    }
}

// Legacy timed recording — fallback if MediaPipe fails to load
async function toggleSignRecordingLegacy() {
    const btnRecordSign = document.getElementById('btnRecordSign');

    continuousRecording = true;
    pendingSend = false;
    isRecordingSign = true;

    while (continuousRecording && liveActive) {
        recBadge.classList.add('on');
        recBorder.classList.add('on');

        const detected = liveCollectedGlosses.map(p => p.gloss.toUpperCase()).join(' ');
        statusBar.innerHTML = '<span class="processing">Recording sign...' +
            (detected ? ' <span class="gloss" style="animation:none;color:#6c8cff">[' + detected + ']</span>' : '') +
            '</span>';

        const recordDuration = 2500;
        let countdownStart = Date.now();
        recCountdownInterval = setInterval(() => {
            const elapsed = Date.now() - countdownStart;
            updateRecCountdown(Math.max(0, recordDuration - elapsed));
        }, 100);

        const videoBlob = await recordFromCamera(recordDuration);
        clearInterval(recCountdownInterval);
        updateRecCountdown(0);

        if (!continuousRecording) break;

        if (!videoBlob) {
            statusBar.innerHTML = '<span class="processing">No video captured, retrying...</span>';
            await sleep(500);
            continue;
        }

        recBadge.classList.remove('on');
        recBorder.classList.remove('on');
        statusBar.innerHTML = '<span class="processing">Analyzing sign... <span class="spinner"></span></span>';

        const formData = new FormData();
        formData.append('video', videoBlob, 'sign.webm');
        formData.append('domain', selectedDomain);

        try {
            const resp = await fetch('/api/process-sign', { method: 'POST', body: formData });
            const result = await resp.json();

            if (!continuousRecording) break;

            if (result.success && result.confidence >= 0.15) {
                const isDuplicate = liveCollectedGlosses.length > 0 &&
                    liveCollectedGlosses[liveCollectedGlosses.length - 1].gloss.toLowerCase() === result.gloss.toLowerCase();

                if (!isDuplicate) {
                    liveCollectedGlosses.push({
                        gloss: result.gloss,
                        confidence: result.confidence,
                        top_k: result.top_k || []
                    });
                    const allDetected = liveCollectedGlosses.map(p => p.gloss.toUpperCase()).join(' ');
                    statusBar.innerHTML = `<span class="gloss">Signs: ${allDetected}</span>`;
                }
            }
        } catch (e) {
            console.error('[Live] Inference error:', e);
            showToast('Sign inference failed \u2014 is the inference API running?', 'error');
        }

        if (continuousRecording) await sleep(300);
    }

    recBadge.classList.remove('on');
    recBorder.classList.remove('on');
    isRecordingSign = false;

    if (pendingSend && liveCollectedGlosses.length > 0) {
        pendingSend = false;
        await sendSignMessage();
    } else {
        btnRecordSign.textContent = 'Record Sign';
        btnRecordSign.style.background = '#b07cf0';
        btnRecordSign.disabled = false;
        if (liveCollectedGlosses.length > 0) {
            const allDetected = liveCollectedGlosses.map(p => p.gloss.toUpperCase()).join(' ');
            statusBar.innerHTML = `<span class="gloss">Signs: ${allDetected}</span>`;
        }
    }
}

function recordFromCamera(durationMs) {
    return new Promise(resolve => {
        if (!cameraStream) { resolve(null); return; }

        recordedChunks = [];
        try {
            mediaRecorder = new MediaRecorder(cameraStream, { mimeType: 'video/webm' });
        } catch (e) {
            try {
                mediaRecorder = new MediaRecorder(cameraStream);
            } catch (e2) {
                resolve(null);
                return;
            }
        }

        mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) recordedChunks.push(e.data);
        };

        mediaRecorder.onstop = () => {
            const blob = new Blob(recordedChunks, { type: 'video/webm' });
            resolve(blob.size > 0 ? blob : null);
        };

        mediaRecorder.start();
        setTimeout(() => {
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
            }
        }, durationMs);
    });
}

// ── Demo mode: play sign videos ──
async function playPatientTurnDemo(turn) {
    if (!isPlaying) return;
    const glosses = turn.glosses;
    const signs = turn.sign_videos || glosses;
    const sampleId = turn.sentence_sample;

    recBadge.classList.add('on');
    recBorder.classList.add('on');
    statusBar.innerHTML = '<span class="processing">Patient signing...</span>';

    let usedSentenceVideo = false;

    if (sampleId && isPlaying) {
        const sampleUrl = `/samples/${sampleId}/original_video.mp4`;
        const played = await playSignVideo(sampleUrl);
        if (played) {
            usedSentenceVideo = true;
            statusBar.innerHTML = '<span class="gloss">Detected: ' + glosses.map(g => g.toUpperCase()).join(' ') + '</span>';
        }
    }

    if (!usedSentenceVideo && isPlaying) {
        let accumulated = [];
        for (let i = 0; i < signs.length && isPlaying; i++) {
            const played = await playSignVideo(`/sign-bank/${signs[i]}.mp4`);
            if (!isPlaying) return;
            if (!played) await sleep(1200);
            if (!isPlaying) return;

            accumulated.push(signs[i].toUpperCase());
            statusBar.innerHTML = '<span class="gloss">Detected: ' + accumulated.join(' ') + '</span>';
            await sleep(250);
            if (!isPlaying) return;
        }
    }

    if (!isPlaying) return;
    recBadge.classList.remove('on');
    recBorder.classList.remove('on');
    signVid.style.display = 'none';
    await sleep(400);
    if (!isPlaying) return;

    statusBar.innerHTML = '<span class="processing">SignBridge constructing sentence... <span class="spinner"></span></span>';
    await sleep(600);
    if (!isPlaying) return;

    const sentence = turn.asl_sentence;

    statusBar.innerHTML = '<span class="sentence">' + sentence + '</span>';
    addMsg('patient', SIGNER_LABEL, sentence);
    history.push({ speaker: 'patient', text: sentence });

    if (isPlaying && (interactionMode === 'in-person' || METHOD === 'speak')) {
        await speak(sentence);
    }
    if (!isPlaying) return;
    await sleep(800);
}

// ══════════════════════════════════════════════════════════════
// TRANSCRIPT (1.7 — with timestamps)
// ══════════════════════════════════════════════════════════════
function addMsg(type, speaker, text) {
    const d = document.createElement('div');
    d.className = 'msg ' + type;
    const now = new Date();
    const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    d.innerHTML = `<div class="speaker">${speaker}<span class="msg-timestamp">${timeStr}</span></div><div class="msg-text">${text}</div>`;
    transcriptMsgs.appendChild(d);
    transcriptMsgs.scrollTop = transcriptMsgs.scrollHeight;
    return d;  // returned so callers can rewrite-in-place (streaming-live caption)
}

// ══════════════════════════════════════════════════════════════
// BREAKDOWN PHASE (1.12 — accessible independently)
// ══════════════════════════════════════════════════════════════
let breakdownData = [];
let breakdownLoaded = false;

async function loadBreakdownSamples() {
    if (breakdownLoaded) return;
    // Try from conversation data first, then from a known list
    const sampleIds = (conversation && conversation.breakdown_samples) ? conversation.breakdown_samples : [];
    if (sampleIds.length === 0) {
        showToast('No breakdown samples available yet. Complete a demo conversation first.', 'info');
        return;
    }
    breakdownData = [];
    for (const sampleId of sampleIds) {
        try {
            const resp = await fetch(`/api/samples/${sampleId}`);
            if (resp.ok) {
                const meta = await resp.json();
                breakdownData.push({ sampleId, meta });
            }
        } catch (e) {
            console.warn('[Breakdown] Failed to load sample:', sampleId, e);
        }
    }
    breakdownLoaded = true;
}

async function showBreakdown() {
    await loadBreakdownSamples();
    document.getElementById('phaseConvo').style.display = 'none';
    hideHomeContent();
    document.getElementById('phaseBreakdown').style.display = 'flex';
    lockHeaderToggles();
    window.scrollTo(0, 0);
    pushRoute('breakdown');
    renderBreakdown();
}

function hideBreakdown() {
    document.getElementById('phaseBreakdown').style.display = 'none';
    document.getElementById('phaseConvo').style.display = 'flex';
}

function renderBreakdown() {
    const tabsEl = document.getElementById('breakdownTabs');
    const sectionsEl = document.getElementById('breakdownSections');
    tabsEl.innerHTML = '';
    sectionsEl.innerHTML = '';

    if (breakdownData.length === 0) {
        sectionsEl.innerHTML = '<p style="color:var(--text-muted);text-align:center;padding:40px;">No breakdown samples available. Complete a demo conversation to see the pipeline breakdown.</p>';
        return;
    }

    breakdownData.forEach((item, idx) => {
        const segments = (item.meta.precomputed || {}).segments || [];
        const rawGlosses = segments.map(s => s.top_1).join(' ').toUpperCase() || item.sampleId;

        const tab = document.createElement('button');
        tab.className = 'sentence-tab' + (idx === 0 ? ' active' : '');
        tab.textContent = `Sentence ${idx + 1}: "${rawGlosses}"`;
        tab.onclick = () => {
            document.querySelectorAll('.sentence-tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.pipeline-section').forEach(s => s.classList.remove('active'));
            tab.classList.add('active');
            document.getElementById(`bd-section-${idx}`).classList.add('active');
        };
        tabsEl.appendChild(tab);

        const sec = document.createElement('div');
        sec.className = 'pipeline-section' + (idx === 0 ? ' active' : '');
        sec.id = `bd-section-${idx}`;
        sec.innerHTML = buildBreakdownHTML(item);
        sectionsEl.appendChild(sec);
    });
}

function buildBreakdownHTML(item) {
    const { sampleId, meta } = item;
    const precomputed = meta.precomputed || {};
    const segments = precomputed.segments || [];
    const rawSentence = precomputed.raw_sentence || segments.map(s => s.top_1).join(' ');
    const llmSentence = precomputed.llm_sentence || meta.reference_sentence || rawSentence;
    const refSentence = meta.reference_sentence || '';

    let html = '';

    html += `
    <div class="pipeline-step">
        <div class="step-header">
            <div class="step-number">1</div>
            <div class="step-title">Pose Estimation \u2014 Extracting body keypoints</div>
        </div>
        <div class="step-content">
            <p style="color:var(--text-secondary);margin-bottom:12px;">MediaPipe Holistic extracts 75 keypoints (hands, face, body) from the video, creating a skeletal representation of the signer.</p>
            <div style="text-align:center;">
                <video src="/demo-data/samples/${sampleId}/pose_video.mp4"
                       controls muted playsinline
                       style="max-width:100%;max-height:280px;border-radius:8px;background:#fff;"></video>
            </div>
        </div>
    </div>`;

    html += `
    <div class="pipeline-step">
        <div class="step-header">
            <div class="step-number">2</div>
            <div class="step-title">Segmentation \u2014 Splitting into ${segments.length} individual signs</div>
        </div>
        <div class="step-content">
            <p style="color:var(--text-secondary);margin-bottom:12px;">SignBridge detects motion boundaries in the pose data to separate the continuous signing into individual sign segments.</p>
            <div class="segment-grid">`;

    segments.forEach((seg, i) => {
        const videoFile = seg.video_file || `segments/segment_${String(i+1).padStart(3,'0')}.mp4`;
        html += `
                <div class="segment-card">
                    <video src="/demo-data/samples/${sampleId}/${videoFile}"
                           muted playsinline loop
                           onmouseenter="this.play()" onmouseleave="this.pause();this.currentTime=0"></video>
                    <div class="seg-label">Segment ${i+1}</div>
                </div>`;
    });

    html += `</div></div></div>`;

    html += `
    <div class="pipeline-step">
        <div class="step-header">
            <div class="step-number">3</div>
            <div class="step-title">Sign Recognition \u2014 Identifying each sign</div>
        </div>
        <div class="step-content">
            <p style="color:var(--text-secondary);margin-bottom:12px;">Each segment is fed through an ST-GCN model trained on the WLASL dataset. The model outputs confidence scores for each possible sign.</p>
            <div class="segment-grid">`;

    segments.forEach((seg, i) => {
        const videoFile = seg.video_file || `segments/segment_${String(i+1).padStart(3,'0')}.mp4`;
        const top1 = seg.top_1 || 'unknown';
        const confidence = seg.confidence || 0;
        const topK = seg.top_k || [{ gloss: top1, confidence }];
        const pct = Math.round(confidence * 100);
        const color = pct >= 80 ? '#4caf80' : pct >= 50 ? '#f0a050' : '#ff5555';

        html += `
                <div class="prediction-card">
                    <video src="/demo-data/samples/${sampleId}/${videoFile}"
                           muted playsinline loop
                           onmouseenter="this.play()" onmouseleave="this.pause();this.currentTime=0"></video>
                    <div class="pred-info">
                        <div class="pred-top1" style="color:${color}">${top1.toUpperCase()} \u2014 ${pct}%</div>
                        <div class="confidence-bar"><div class="conf-fill" style="width:${pct}%;background:${color}"></div></div>
                        <div class="top-k-list">`;
        topK.forEach(k => {
            html += `<div class="top-k-item"><span>${k.gloss}</span><span>${Math.round(k.confidence*100)}%</span></div>`;
        });
        html += `</div></div></div>`;
    });

    html += `</div></div></div>`;

    html += `
    <div class="pipeline-step">
        <div class="step-header">
            <div class="step-number">4</div>
            <div class="step-title">LLM Sentence Construction \u2014 Building natural English</div>
        </div>
        <div class="step-content">
            <p style="color:var(--text-secondary);margin-bottom:16px;">The recognized glosses are sent to Gemini along with conversation context. The LLM transforms raw ASL glosses into a grammatically correct English sentence.</p>
            <div class="construction-display">
                <div class="construct-box raw">
                    <div class="box-label">Raw Glosses (Model Output)</div>
                    <div class="box-content">${rawSentence.toUpperCase()}</div>
                </div>
                <div class="construct-arrow">\u2192</div>
                <div class="construct-box llm">
                    <div class="box-label">SignBridge Output (Gemini)</div>
                    <div class="box-content">"${llmSentence}"</div>
                </div>
            </div>
            ${refSentence ? `<p style="color:var(--text-secondary);margin-top:14px;font-size:0.82rem;text-align:center;">Reference: "${refSentence}"</p>` : ''}
        </div>
    </div>`;

    return html;
}

// ── Kiosk auto-start ──
if (IS_KIOSK) {
    // Auto-start after a brief delay for page to settle
    setTimeout(() => kioskShowTapToStart(), 500);
}
