/**
 * Phase 4: Construct Sentence
 * Uses LLM to construct grammatically correct sentence from glosses
 * Supports both Live and Demo modes
 */

const state = {
    segments: [],
    rawSentence: '',
    llmSentence: '',
    selections: [],
    demoSample: null
};

const elements = {
    glossChips: document.getElementById('gloss-chips'),
    btnConstruct: document.getElementById('btn-construct'),
    llmSection: document.getElementById('llm-section'),
    loadingState: document.getElementById('loading-state'),
    constructResults: document.getElementById('construct-results'),
    rawSentence: document.getElementById('raw-sentence'),
    llmSentence: document.getElementById('llm-sentence'),
    selectionsList: document.getElementById('selections-list')
};

document.addEventListener('DOMContentLoaded', initialize);

function initialize() {
    const segmentsJson = sessionStorage.getItem('segments');

    if (!segmentsJson) {
        alert('No segments found. Please go back and complete previous steps.');
        window.location.href = '/';
        return;
    }

    state.segments = JSON.parse(segmentsJson);

    // Check if demo mode
    const poseFile = sessionStorage.getItem('poseFile');
    if (poseFile && poseFile.startsWith('demo:')) {
        state.demoSample = getSelectedSample();
    }

    displayGlossChips();

    elements.btnConstruct.addEventListener('click', constructSentence);

    // Auto-trigger construction in demo mode
    if (state.demoSample?.precomputed?.llm_sentence) {
        constructSentence();
    }
}

function displayGlossChips() {
    elements.glossChips.innerHTML = '';

    state.segments.forEach((segment, index) => {
        const chip = document.createElement('div');
        chip.className = 'gloss-chip';
        chip.innerHTML = `
            <span class="chip-number">${index + 1}</span>
            <span class="chip-gloss">${segment.top_1}</span>
            <span class="chip-confidence">${(segment.confidence * 100).toFixed(0)}%</span>
        `;
        elements.glossChips.appendChild(chip);
    });
}

async function constructSentence() {
    // Check if demo mode with pre-computed LLM sentence
    if (state.demoSample?.precomputed?.llm_sentence) {
        useDemoConstruction();
        return;
    }

    // Live mode - call LLM
    showLoading();

    try {
        const response = await fetch('/api/construct', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ predictions: state.segments })
        });

        const result = await response.json();

        if (result.success) {
            state.rawSentence = result.raw_sentence;
            state.llmSentence = result.llm_sentence;
            state.selections = result.selections || [];

            // Store for evaluation
            sessionStorage.setItem('rawSentence', state.rawSentence);
            sessionStorage.setItem('llmSentence', state.llmSentence);

            displayResults();
        } else {
            throw new Error(result.error || 'Construction failed');
        }

    } catch (error) {
        console.error('Construction error:', error);
        alert('Construction failed: ' + error.message);
        elements.llmSection.style.display = 'block';
        elements.loadingState.style.display = 'none';
    }
}

function useDemoConstruction() {
    showLoading();

    // Small delay for visual feedback
    setTimeout(() => {
        const precomputed = state.demoSample.precomputed;

        state.rawSentence = precomputed.raw_sentence;
        state.llmSentence = precomputed.llm_sentence;
        state.selections = [];

        // Store for evaluation
        sessionStorage.setItem('rawSentence', state.rawSentence);
        sessionStorage.setItem('llmSentence', state.llmSentence);

        displayResults();
    }, 300);
}

function displayResults() {
    elements.rawSentence.textContent = state.rawSentence;
    elements.llmSentence.textContent = state.llmSentence;

    // Display word selections
    if (state.selections.length > 0) {
        elements.selectionsList.innerHTML = '';
        state.selections.forEach((word, index) => {
            const item = document.createElement('div');
            item.className = 'selection-item';
            item.innerHTML = `
                <span class="selection-position">Position ${index + 1}:</span>
                <span class="selection-word">${word}</span>
            `;
            elements.selectionsList.appendChild(item);
        });
    } else {
        const wordSelections = document.getElementById('word-selections');
        if (wordSelections) {
            wordSelections.style.display = 'none';
        }
    }

    showResults();
}

function showLoading() {
    elements.llmSection.style.display = 'none';
    elements.loadingState.style.display = 'flex';
    elements.constructResults.style.display = 'none';
}

function showResults() {
    elements.llmSection.style.display = 'none';
    elements.loadingState.style.display = 'none';
    elements.constructResults.style.display = 'block';
}
