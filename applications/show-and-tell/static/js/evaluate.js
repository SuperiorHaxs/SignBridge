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
    btnStartOver: document.getElementById('btn-start-over'),
    btnSaveSession: document.getElementById('btn-save-session')
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

    // Check if coming from fast mode (Live mode)
    const isFast = isFastMode();

    if (!state.referenceSentence) {
        alert('No reference sentence found. Please start from the beginning.');
        window.location.href = '/';
        return;
    }

    if (!state.rawSentence || !state.llmSentence) {
        if (isFast) {
            alert('Processing did not complete. Please try again.');
            window.location.href = '/';
        } else {
            alert('No sentences to evaluate. Please complete previous steps.');
            window.location.href = '/construct';
        }
        return;
    }

    // Display sentences
    elements.referenceSentence.textContent = state.referenceSentence;
    elements.rawSentence.textContent = state.rawSentence;
    elements.llmSentence.textContent = state.llmSentence;

    // Event listeners
    elements.btnEvaluate.addEventListener('click', evaluateSentences);
    elements.btnStartOver.addEventListener('click', startOver);
    if (elements.btnSaveSession) {
        elements.btnSaveSession.addEventListener('click', saveSession);
    }

    // Auto-trigger evaluation in fast mode or demo mode with precomputed
    if (isFast || state.demoSample?.precomputed?.evaluation) {
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
        // Get selected glosses (LLM selections from top-k) for Effective GA
        const selectedGlossesJson = sessionStorage.getItem('selectedGlosses');
        const selectedGlosses = selectedGlossesJson ? JSON.parse(selectedGlossesJson) : [];

        const response = await fetch('/api/evaluate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                raw_sentence: state.rawSentence,
                llm_sentence: state.llmSentence,
                reference: state.referenceSentence,
                selected_glosses: selectedGlosses  // For Effective GA calculation
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

async function useDemoEvaluation() {
    showLoading();

    // Re-fetch sample to ensure we have latest metadata (including selections)
    const sampleId = state.demoSample.id;

    try {
        const response = await fetch(`/api/samples/${sampleId}?t=${Date.now()}`);  // Cache bust
        if (response.ok) {
            const freshSample = await response.json();
            state.demoSample = freshSample;
            setSelectedSample(freshSample);  // Update sessionStorage
        }
    } catch (e) {
        // Silently continue with cached sample if fetch fails
    }

    const evaluation = state.demoSample.precomputed.evaluation;
    const rawMetrics = { ...evaluation.raw };
    const llmMetrics = { ...evaluation.llm };

    // Calculate gloss_accuracy if not present in precomputed data
    if (rawMetrics.model_gloss_accuracy === undefined || llmMetrics.effective_gloss_accuracy === undefined) {
        const segments = state.demoSample.precomputed.segments || [];
        const predictedGlosses = segments.map(s => (s.top_1 || '').toLowerCase());

        // Expected glosses derived from REFERENCE sentence (content words)
        const reference = state.referenceSentence.toLowerCase();
        const stopwords = new Set(['the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'of', 'in',
            'for', 'on', 'with', 'at', 'by', 'from', 'and', 'or', 'but', 'his', 'her', 'its',
            'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him', 'she', 'they', 'them',
            'their', 'it', 'this', 'that', 'these', 'those', 'be', 'been', 'being', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
            'might', 'must', 'shall', 'can', 'playing']);
        const refWords = reference.match(/\b[a-z]+\b/g) || [];
        const expectedGlosses = refWords.filter(w => !stopwords.has(w));

        // LLM-selected glosses - use precomputed.selections (freshly fetched)
        const precomputedSelections = state.demoSample.precomputed.selections;
        let selectedGlosses;
        if (precomputedSelections && precomputedSelections.length > 0) {
            selectedGlosses = precomputedSelections.map(g => g.toLowerCase());
        } else {
            selectedGlosses = predictedGlosses;
        }

        // Simple lemmatizer for comparison - only remove suffix if result is 3+ chars
        const lemmatize = (word) => {
            const result = word.replace(/(es|ed|ing|ly|s)$/, '');
            return result.length >= 3 ? result : word;  // Don't over-lemmatize short words
        };

        // Calculate Model GA (top-1 predictions vs expected)
        let modelCorrect = 0;
        const modelTotal = Math.max(predictedGlosses.length, expectedGlosses.length);
        if (modelTotal > 0) {
            for (let i = 0; i < modelTotal; i++) {
                const pred = lemmatize(predictedGlosses[i] || '');
                const exp = lemmatize(expectedGlosses[i] || '');
                if (pred && exp && pred === exp) modelCorrect++;
            }
        }
        const modelGA = modelTotal > 0 ? (modelCorrect / modelTotal) * 100 : 0;

        // Calculate Effective GA (LLM-selected vs expected)
        let effectiveCorrect = 0;
        const effectiveTotal = Math.max(selectedGlosses.length, expectedGlosses.length);
        if (effectiveTotal > 0) {
            for (let i = 0; i < effectiveTotal; i++) {
                const sel = lemmatize(selectedGlosses[i] || '');
                const exp = lemmatize(expectedGlosses[i] || '');
                if (sel && exp && sel === exp) effectiveCorrect++;
            }
        }
        const effectiveGA = effectiveTotal > 0 ? (effectiveCorrect / effectiveTotal) * 100 : 0;

        // Store both Model GA and Effective GA
        rawMetrics.model_gloss_accuracy = modelGA;
        rawMetrics.effective_gloss_accuracy = modelGA;  // Raw uses top-1, so same as Model GA
        llmMetrics.model_gloss_accuracy = modelGA;      // Model GA is same for both
        llmMetrics.effective_gloss_accuracy = effectiveGA;  // LLM may have better selections
    }

    displayMetrics(rawMetrics, llmMetrics);
}

function displayMetrics(rawMetrics, llmMetrics) {
    const metrics = ['model_gloss_accuracy', 'effective_gloss_accuracy', 'coverage_f1', 'quality', 'composite'];

    metrics.forEach(metric => {
        const row = document.querySelector(`.metric-row[data-metric="${metric}"]`);
        if (!row) return;

        const rawValue = rawMetrics[metric] ?? 0;
        const llmValue = llmMetrics[metric] ?? 0;
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

    // Display coverage details (missing/hallucinated words)
    displayCoverageDetails(rawMetrics, llmMetrics);

    showResults();
}

function displayCoverageDetails(rawMetrics, llmMetrics) {
    const coverageDetails = document.getElementById('coverage-details');
    if (!coverageDetails) return;

    const rawMissing = rawMetrics.missing_words || [];
    const rawHallucinated = rawMetrics.hallucinated_words || [];
    const llmMissing = llmMetrics.missing_words || [];
    const llmHallucinated = llmMetrics.hallucinated_words || [];

    // Check if there's anything to show
    const hasDetails = rawMissing.length > 0 || rawHallucinated.length > 0 ||
                       llmMissing.length > 0 || llmHallucinated.length > 0;

    if (!hasDetails) {
        coverageDetails.style.display = 'none';
        return;
    }

    // Update raw sentence details
    const rawMissingEl = document.querySelector('#raw-missing-words .words');
    const rawHallucinatedEl = document.querySelector('#raw-hallucinated-words .words');
    if (rawMissingEl) rawMissingEl.textContent = rawMissing.length > 0 ? rawMissing.join(', ') : 'None';
    if (rawHallucinatedEl) rawHallucinatedEl.textContent = rawHallucinated.length > 0 ? rawHallucinated.join(', ') : 'None';

    // Update LLM sentence details
    const llmMissingEl = document.querySelector('#llm-missing-words .words');
    const llmHallucinatedEl = document.querySelector('#llm-hallucinated-words .words');
    if (llmMissingEl) llmMissingEl.textContent = llmMissing.length > 0 ? llmMissing.join(', ') : 'None';
    if (llmHallucinatedEl) llmHallucinatedEl.textContent = llmHallucinated.length > 0 ? llmHallucinated.join(', ') : 'None';

    coverageDetails.style.display = 'block';
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

async function saveSession() {
    const btn = elements.btnSaveSession;
    const originalText = btn.textContent;

    try {
        btn.textContent = 'Saving...';
        btn.disabled = true;

        const response = await fetch('/api/save-session', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                reference: state.referenceSentence,
                raw_sentence: state.rawSentence,
                llm_sentence: state.llmSentence
            })
        });

        const data = await response.json();

        if (data.success) {
            btn.textContent = 'Saved!';
            btn.classList.add('btn-success');

            let message = `Session saved!\n\nFolder: ${data.session_name}`;
            message += `\nSegments: ${data.segment_count || 0}`;

            if (data.needs_resegment) {
                message += `\n\nNote: No segment files found. The video/pose will need to be re-segmented.`;
                message += `\n\nUse with prepare_demo_sample.py (will re-segment):`;
                message += `\npython prepare_demo_sample.py --video "${data.saved_path}\\capture.mp4" --reference "YOUR REFERENCE" --name "NAME"`;
            } else {
                message += `\n\nUse with prepare_demo_sample.py:`;
                message += `\npython prepare_demo_sample.py --from-session "${data.saved_path}" --reference "YOUR REFERENCE" --name "NAME"`;
            }

            alert(message);
        } else {
            throw new Error(data.error || 'Save failed');
        }
    } catch (e) {
        console.error('Save session failed:', e);
        btn.textContent = 'Save Failed';
        btn.classList.add('btn-danger');
        alert('Failed to save session: ' + e.message);
    } finally {
        setTimeout(() => {
            btn.textContent = originalText;
            btn.disabled = false;
            btn.classList.remove('btn-success', 'btn-danger');
        }, 3000);
    }
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
