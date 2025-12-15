/**
 * Phase 5: Evaluate Results
 * Calculates and displays evaluation metrics
 * Supports both Live and Demo modes
 */

const state = {
    referenceSentence: '',
    rawSentence: '',
    llmSentence: '',
    demoSample: null
};

const elements = {
    referenceSentence: document.getElementById('reference-sentence'),
    rawSentence: document.getElementById('raw-sentence'),
    llmSentence: document.getElementById('llm-sentence'),
    btnEvaluate: document.getElementById('btn-evaluate'),
    evaluateControls: document.getElementById('evaluate-controls'),
    loadingState: document.getElementById('loading-state'),
    metricsSection: document.getElementById('metrics-section'),
    finalActions: document.getElementById('final-actions'),
    btnStartOver: document.getElementById('btn-start-over')
};

document.addEventListener('DOMContentLoaded', initialize);

function initialize() {
    // Load sentences from session storage
    state.referenceSentence = sessionStorage.getItem('referenceSentence') || '';
    state.rawSentence = sessionStorage.getItem('rawSentence') || '';
    state.llmSentence = sessionStorage.getItem('llmSentence') || '';

    // Check if demo mode
    const poseFile = sessionStorage.getItem('poseFile');
    if (poseFile && poseFile.startsWith('demo:')) {
        state.demoSample = getSelectedSample();
    }

    if (!state.referenceSentence) {
        alert('No reference sentence found. Please start from the beginning.');
        window.location.href = '/';
        return;
    }

    if (!state.rawSentence || !state.llmSentence) {
        alert('No sentences to evaluate. Please complete previous steps.');
        window.location.href = '/construct';
        return;
    }

    // Display sentences
    elements.referenceSentence.textContent = state.referenceSentence;
    elements.rawSentence.textContent = state.rawSentence;
    elements.llmSentence.textContent = state.llmSentence;

    // Event listeners
    elements.btnEvaluate.addEventListener('click', evaluateSentences);
    elements.btnStartOver.addEventListener('click', startOver);

    // Auto-trigger evaluation in demo mode
    if (state.demoSample?.precomputed?.evaluation) {
        evaluateSentences();
    }
}

async function evaluateSentences() {
    // Check if demo mode with pre-computed evaluation
    if (state.demoSample?.precomputed?.evaluation) {
        useDemoEvaluation();
        return;
    }

    // Live mode - calculate metrics
    showLoading();

    try {
        const response = await fetch('/api/evaluate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                raw_sentence: state.rawSentence,
                llm_sentence: state.llmSentence,
                reference: state.referenceSentence
            })
        });

        const result = await response.json();

        if (result.success) {
            displayMetrics(result.raw, result.llm);
        } else {
            throw new Error(result.error || 'Evaluation failed');
        }

    } catch (error) {
        console.error('Evaluation error:', error);
        alert('Evaluation failed: ' + error.message);
        elements.evaluateControls.style.display = 'block';
        elements.loadingState.style.display = 'none';
    }
}

function useDemoEvaluation() {
    showLoading();

    // Small delay for visual feedback
    setTimeout(() => {
        const evaluation = state.demoSample.precomputed.evaluation;
        displayMetrics(evaluation.raw, evaluation.llm);
    }, 300);
}

function displayMetrics(rawMetrics, llmMetrics) {
    const metrics = ['bleu', 'bert', 'quality', 'gloss_accuracy', 'composite'];

    metrics.forEach(metric => {
        const row = document.querySelector(`.metric-row[data-metric="${metric}"]`);
        if (!row) return;

        const rawValue = rawMetrics[metric] || 0;
        const llmValue = llmMetrics[metric] || 0;
        const delta = llmValue - rawValue;

        // Update values
        row.querySelector('.raw-value').textContent = formatScore(rawValue);
        row.querySelector('.llm-value').textContent = formatScore(llmValue);

        // Update delta with color
        const deltaCell = row.querySelector('.metric-delta');
        deltaCell.textContent = formatDelta(delta);
        deltaCell.classList.remove('positive', 'negative', 'neutral');

        if (delta > 0.5) {
            deltaCell.classList.add('positive');
        } else if (delta < -0.5) {
            deltaCell.classList.add('negative');
        } else {
            deltaCell.classList.add('neutral');
        }
    });

    showResults();
}

function formatScore(value) {
    if (typeof value !== 'number' || isNaN(value)) return '-';
    return value.toFixed(1);
}

function formatDelta(value) {
    if (typeof value !== 'number' || isNaN(value)) return '-';
    const sign = value >= 0 ? '+' : '';
    const arrow = value > 0 ? '\u25B2 ' : (value < 0 ? '\u25BC ' : '');
    return `${arrow}${sign}${value.toFixed(1)}`;
}

async function startOver() {
    try {
        await fetch('/api/reset', { method: 'POST' });
    } catch (e) {
        console.log('Reset request failed, continuing anyway');
    }

    sessionStorage.clear();
    window.location.href = '/';
}

function showLoading() {
    elements.evaluateControls.style.display = 'none';
    elements.loadingState.style.display = 'flex';
    elements.metricsSection.style.display = 'none';
    elements.finalActions.style.display = 'none';
}

function showResults() {
    elements.evaluateControls.style.display = 'none';
    elements.loadingState.style.display = 'none';
    elements.metricsSection.style.display = 'block';
    elements.finalActions.style.display = 'flex';
}
