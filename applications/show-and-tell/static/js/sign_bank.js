/**
 * Sign Bank - Record individual signs for the vocabulary
 * These recordings are used as tutorial videos in Learn mode
 */

// State management
let currentWord = null;
let mediaRecorder = null;
let recordedChunks = [];
let webcamStream = null;
let recordedBlob = null;
let recordedWords = new Set();  // Words recorded in Sign Bank
let availableWords = new Set(); // Words that already have tutorials (from demo samples)
let wordCategories = {};        // Will be loaded from API
let allGlosses = [];            // All available glosses

// DOM Elements
const elements = {
    wordSearch: document.getElementById('word-search'),
    wordCategories: document.getElementById('word-categories'),
    currentWord: document.getElementById('current-word'),
    wordStatus: document.getElementById('word-status'),
    recordingIndicator: document.getElementById('recording-indicator'),
    webcamVideo: document.getElementById('webcam-video'),
    previewVideo: document.getElementById('preview-video'),
    previewPlaceholder: document.getElementById('preview-placeholder'),
    btnStartRecord: document.getElementById('btn-start-record'),
    btnStopRecord: document.getElementById('btn-stop-record'),
    btnSave: document.getElementById('btn-save'),
    btnDiscard: document.getElementById('btn-discard'),
    recordedCount: document.getElementById('recorded-count'),
    statRecorded: document.getElementById('stat-recorded'),
    statRemaining: document.getElementById('stat-remaining'),
    statPercent: document.getElementById('stat-percent')
};

// Initialize the page
document.addEventListener('DOMContentLoaded', async () => {
    await initializeWebcam();
    await loadAvailableGlosses();
    await loadRecordedWords();
    await loadExistingTutorials();
    renderWordList();
    setupEventListeners();
    updateStats();
});

// Initialize webcam
async function initializeWebcam() {
    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: 'user' },
            audio: false
        });
        elements.webcamVideo.srcObject = webcamStream;
    } catch (error) {
        console.error('Failed to access webcam:', error);
        alert('Could not access webcam. Please ensure camera permissions are granted.');
    }
}

// Load available glosses from the API (uses the production model's vocabulary)
async function loadAvailableGlosses() {
    try {
        const response = await fetch('/api/available-glosses');
        if (response.ok) {
            const data = await response.json();
            allGlosses = data.all_glosses || [];
            wordCategories = data.categories || {};

            // If no categories returned, create a default "All Words" category
            if (Object.keys(wordCategories).length === 0 && allGlosses.length > 0) {
                wordCategories = { 'All Words': allGlosses };
            }

            console.log(`[SignBank] Loaded ${allGlosses.length} glosses in ${Object.keys(wordCategories).length} categories`);
        }
    } catch (error) {
        console.error('Failed to load available glosses:', error);
    }
}

// Load list of already recorded words in Sign Bank
async function loadRecordedWords() {
    try {
        const response = await fetch('/api/sign-bank/list');
        if (response.ok) {
            const data = await response.json();
            recordedWords = new Set(data.recorded_words || []);
            console.log(`[SignBank] Found ${recordedWords.size} recordings in Sign Bank`);
        }
    } catch (error) {
        console.error('Failed to load recorded words:', error);
    }
}

// Load existing tutorials (from demo samples) to mark words as already available
async function loadExistingTutorials() {
    try {
        // Check which glosses already have tutorials from demo samples
        const response = await fetch('/api/gloss-tutorials', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ glosses: allGlosses })
        });

        if (response.ok) {
            const data = await response.json();
            const tutorials = data.tutorials || [];

            for (const tutorial of tutorials) {
                // Mark as available if it has a video (from sign_bank or demo)
                if (tutorial.available && tutorial.video_url) {
                    availableWords.add(tutorial.gloss.toLowerCase());
                }
            }

            console.log(`[SignBank] Found ${availableWords.size} glosses with existing tutorials`);
        }
    } catch (error) {
        console.error('Failed to load existing tutorials:', error);
    }
}

// Render the word list with categories
function renderWordList(filter = '') {
    elements.wordCategories.innerHTML = '';

    for (const [category, words] of Object.entries(wordCategories)) {
        const filteredWords = words.filter(word =>
            word.toLowerCase().includes(filter.toLowerCase())
        );

        if (filteredWords.length === 0) continue;

        const section = document.createElement('div');
        section.className = 'category-section';

        const header = document.createElement('div');
        header.className = 'category-header';
        header.innerHTML = `
            <span>${category}</span>
            <span class="category-toggle">${filteredWords.length} words</span>
        `;

        const grid = document.createElement('div');
        grid.className = 'word-grid';

        filteredWords.forEach(word => {
            const chip = document.createElement('div');
            chip.className = 'word-chip';
            chip.textContent = word;
            const wordLower = word.toLowerCase();

            // Check if recorded in Sign Bank (highest priority - green)
            if (recordedWords.has(wordLower)) {
                chip.classList.add('recorded');
                chip.title = 'Recorded in Sign Bank';
            }
            // Check if available from demo samples (blue)
            else if (availableWords.has(wordLower)) {
                chip.classList.add('available');
                chip.title = 'Available from demo samples';
            }
            // Not recorded yet (gray)
            else {
                chip.classList.add('not-recorded');
                chip.title = 'Not recorded yet';
            }

            if (currentWord === word) {
                chip.classList.add('selected');
            }

            chip.addEventListener('click', () => selectWord(word));
            grid.appendChild(chip);
        });

        section.appendChild(header);
        section.appendChild(grid);
        elements.wordCategories.appendChild(section);
    }
}

// Select a word to record
async function selectWord(word) {
    currentWord = word;
    elements.currentWord.textContent = word.toUpperCase();
    const wordLower = word.toLowerCase();

    if (recordedWords.has(wordLower)) {
        elements.wordStatus.textContent = 'Recorded in Sign Bank - Record again to replace';
        // Load existing recording preview from Sign Bank
        await loadExistingRecording(word);
    } else if (availableWords.has(wordLower)) {
        elements.wordStatus.textContent = 'Available from demo samples - Record to add your own version';
        // Could load demo sample preview here if desired
        hidePreview();
    } else {
        elements.wordStatus.textContent = 'Ready to record';
        hidePreview();
    }

    renderWordList(elements.wordSearch.value);
    resetRecordingState();
}

// Load existing recording for preview
async function loadExistingRecording(word) {
    try {
        const response = await fetch(`/api/sign-bank/video/${word}`);
        if (response.ok) {
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            elements.previewVideo.src = url;
            elements.previewVideo.style.display = 'block';
            elements.previewPlaceholder.style.display = 'none';
        }
    } catch (error) {
        console.error('Failed to load existing recording:', error);
    }
}

// Hide preview video
function hidePreview() {
    elements.previewVideo.style.display = 'none';
    elements.previewVideo.src = '';
    elements.previewPlaceholder.style.display = 'flex';
}

// Setup event listeners
function setupEventListeners() {
    // Search filter
    elements.wordSearch.addEventListener('input', (e) => {
        renderWordList(e.target.value);
    });

    // Recording buttons
    elements.btnStartRecord.addEventListener('click', startRecording);
    elements.btnStopRecord.addEventListener('click', stopRecording);
    elements.btnSave.addEventListener('click', saveRecording);
    elements.btnDiscard.addEventListener('click', discardRecording);
}

// Start recording
function startRecording() {
    if (!currentWord) {
        alert('Please select a word first');
        return;
    }

    if (!webcamStream) {
        alert('Webcam not available');
        return;
    }

    recordedChunks = [];

    mediaRecorder = new MediaRecorder(webcamStream, {
        mimeType: 'video/webm;codecs=vp9'
    });

    mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
            recordedChunks.push(event.data);
        }
    };

    mediaRecorder.onstop = () => {
        recordedBlob = new Blob(recordedChunks, { type: 'video/webm' });
        const url = URL.createObjectURL(recordedBlob);
        elements.previewVideo.src = url;
        elements.previewVideo.style.display = 'block';
        elements.previewPlaceholder.style.display = 'none';

        showSaveControls();
    };

    mediaRecorder.start(100);

    // Update UI
    elements.recordingIndicator.classList.add('active');
    elements.btnStartRecord.style.display = 'none';
    elements.btnStopRecord.style.display = 'inline-flex';
    elements.btnSave.style.display = 'none';
    elements.btnDiscard.style.display = 'none';
}

// Stop recording
function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }

    elements.recordingIndicator.classList.remove('active');
    elements.btnStopRecord.style.display = 'none';
}

// Show save/discard controls
function showSaveControls() {
    elements.btnSave.style.display = 'inline-flex';
    elements.btnDiscard.style.display = 'inline-flex';
    elements.btnStartRecord.style.display = 'none';
}

// Save the recording
async function saveRecording() {
    if (!recordedBlob || !currentWord) {
        alert('No recording to save');
        return;
    }

    elements.btnSave.disabled = true;
    elements.btnSave.textContent = 'Saving...';

    try {
        const formData = new FormData();
        formData.append('video', recordedBlob, `${currentWord}.webm`);
        formData.append('gloss', currentWord);

        const response = await fetch('/api/sign-bank/save', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const data = await response.json();
            recordedWords.add(currentWord.toLowerCase());
            updateStats();
            renderWordList(elements.wordSearch.value);
            elements.wordStatus.textContent = 'Saved successfully!';

            // Move to next unrecorded word
            selectNextUnrecordedWord();
        } else {
            const error = await response.json();
            alert('Failed to save: ' + (error.error || 'Unknown error'));
        }
    } catch (error) {
        console.error('Failed to save recording:', error);
        alert('Failed to save recording');
    } finally {
        elements.btnSave.disabled = false;
        elements.btnSave.textContent = 'Save';
        resetRecordingState();
    }
}

// Discard recording and re-record
function discardRecording() {
    recordedBlob = null;
    recordedChunks = [];
    hidePreview();
    resetRecordingState();
    elements.wordStatus.textContent = 'Ready to record';
}

// Reset recording controls to initial state
function resetRecordingState() {
    elements.btnStartRecord.style.display = 'inline-flex';
    elements.btnStopRecord.style.display = 'none';
    elements.btnSave.style.display = 'none';
    elements.btnDiscard.style.display = 'none';
    elements.recordingIndicator.classList.remove('active');
}

// Select next unrecorded word (skips words that already have tutorials)
function selectNextUnrecordedWord() {
    const allWords = Object.values(wordCategories).flat();
    const nextWord = allWords.find(word => {
        const wordLower = word.toLowerCase();
        return !recordedWords.has(wordLower) && !availableWords.has(wordLower);
    });

    if (nextWord) {
        selectWord(nextWord);
    }
}

// Update statistics
function updateStats() {
    const totalWords = allGlosses.length || Object.values(wordCategories).flat().length;

    // Count Sign Bank recordings
    const signBankRecorded = recordedWords.size;

    // Count total available (Sign Bank + demo samples, avoiding duplicates)
    const allAvailable = new Set([...recordedWords, ...availableWords]);
    const totalAvailable = allAvailable.size;

    // Remaining = total - available
    const remaining = totalWords - totalAvailable;
    const percent = totalWords > 0 ? Math.round((totalAvailable / totalWords) * 100) : 0;

    elements.recordedCount.textContent = `${totalAvailable}/${totalWords}`;
    elements.statRecorded.textContent = totalAvailable;
    elements.statRemaining.textContent = remaining;
    elements.statPercent.textContent = `${percent}%`;
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
    }
});
