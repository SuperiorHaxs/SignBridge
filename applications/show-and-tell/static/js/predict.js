/**
 * Phase 3: Predict Glosses
 * Displays model predictions for each segment
 */

const state = {
    segments: []
};

const elements = {
    loadingState: document.getElementById('loading-state'),
    errorState: document.getElementById('error-state'),
    errorMessage: document.getElementById('error-message'),
    predictionsContent: document.getElementById('predictions-content'),
    predictionsGrid: document.getElementById('predictions-grid'),
    detectedGlosses: document.getElementById('detected-glosses'),
    cardTemplate: document.getElementById('prediction-card-template')
};

document.addEventListener('DOMContentLoaded', initialize);

function initialize() {
    // Get segments from session storage (already have predictions from Phase 2)
    const segmentsJson = sessionStorage.getItem('segments');

    if (!segmentsJson) {
        showError('No segments found. Please go back and segment the video first.');
        return;
    }

    state.segments = JSON.parse(segmentsJson);

    // Check if predictions exist
    if (state.segments.length > 0 && state.segments[0].top_1) {
        // Predictions already exist from segmentation step
        displayPredictions(state.segments);
    } else {
        showError('No predictions found. Please run segmentation again.');
    }
}

function displayPredictions(segments) {
    elements.predictionsGrid.innerHTML = '';

    segments.forEach((segment, index) => {
        const card = createPredictionCard(segment, index);
        elements.predictionsGrid.appendChild(card);
    });

    // Update detected glosses
    const glosses = segments.map(s => s.top_1).join(' ');
    elements.detectedGlosses.textContent = glosses;

    // Store raw sentence for construct phase
    sessionStorage.setItem('rawSentence', glosses);

    showResults();
}

function createPredictionCard(segment, index) {
    const template = elements.cardTemplate.content.cloneNode(true);
    const card = template.querySelector('.segment-card');

    // Set segment number
    card.querySelector('.seg-num').textContent = segment.segment_id;

    // Set video source (add cache buster)
    const video = card.querySelector('.segment-video');
    video.src = segment.video_url + '?t=' + Date.now();

    // Set Top-1 prediction
    card.querySelector('.prediction-gloss').textContent = segment.top_1;

    // Set confidence bar
    const confidence = segment.confidence * 100;
    const confidenceBar = card.querySelector('.confidence-bar');
    const confidenceValue = card.querySelector('.confidence-value');

    confidenceBar.style.width = `${confidence}%`;
    confidenceValue.textContent = `${confidence.toFixed(1)}%`;

    // Color code confidence
    if (confidence >= 80) {
        confidenceBar.classList.add('high');
    } else if (confidence >= 50) {
        confidenceBar.classList.add('medium');
    } else {
        confidenceBar.classList.add('low');
    }

    // Set Top-K predictions
    const topkList = card.querySelector('.topk-list');
    if (segment.top_k && segment.top_k.length > 1) {
        segment.top_k.slice(1, 3).forEach((pred, i) => {
            const li = document.createElement('li');
            const conf = (pred.confidence * 100).toFixed(1);
            li.innerHTML = `<span class="topk-rank">${i + 2}.</span> ${pred.gloss} <span class="topk-conf">${conf}%</span>`;
            topkList.appendChild(li);
        });
    } else {
        card.querySelector('.prediction-topk').style.display = 'none';
    }

    return card;
}

function showError(message) {
    elements.loadingState.style.display = 'none';
    elements.errorState.style.display = 'flex';
    elements.errorMessage.textContent = message;
    elements.predictionsContent.style.display = 'none';
}

function showResults() {
    elements.loadingState.style.display = 'none';
    elements.errorState.style.display = 'none';
    elements.predictionsContent.style.display = 'block';
}
