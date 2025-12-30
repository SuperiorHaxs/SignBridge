/**
 * Live Mode Setup - Reference Sentence Selection/Building
 */

// State
let presetSentences = [];
let glossCategories = {};
let allGlosses = [];
let selectedGlosses = [];
let selectedCategory = null;
let finalReference = null;
let finalGlosses = [];

// DOM Elements
let presetDropdown;
let categoryTabs;
let wordsGrid;
let selectedWordsDiv;
let clearBtn, removeLastBtn, generateBtn;
let generatedSection, generatedInput;
let regenerateBtn, editBtn, useGeneratedBtn;
let finalSection, finalSentence, finalGlossesSpan;
let continueBtn;

// Initialize on page load
document.addEventListener('DOMContentLoaded', async () => {
    // Check mode - redirect if not Live mode
    if (getMode() !== AppMode.LIVE) {
        window.location.href = '/';
        return;
    }

    initializeElements();
    await loadData();
    setupEventListeners();
});

function initializeElements() {
    presetDropdown = document.getElementById('preset-dropdown');
    categoryTabs = document.getElementById('category-tabs');
    wordsGrid = document.getElementById('words-grid');
    selectedWordsDiv = document.getElementById('selected-words');
    clearBtn = document.getElementById('clear-words-btn');
    removeLastBtn = document.getElementById('remove-last-btn');
    generateBtn = document.getElementById('generate-btn');
    generatedSection = document.getElementById('generated-reference');
    generatedInput = document.getElementById('generated-sentence');
    regenerateBtn = document.getElementById('regenerate-btn');
    editBtn = document.getElementById('edit-btn');
    useGeneratedBtn = document.getElementById('use-generated-btn');
    finalSection = document.getElementById('final-selection');
    finalSentence = document.getElementById('final-sentence');
    finalGlossesSpan = document.getElementById('final-glosses');
    continueBtn = document.getElementById('continue-btn');
}

async function loadData() {
    try {
        // Load preset sentences
        const sentencesResp = await fetch('/api/reference-sentences');
        const sentencesData = await sentencesResp.json();
        presetSentences = sentencesData.sentences || [];
        populatePresetDropdown();

        // Load available glosses
        const glossesResp = await fetch('/api/available-glosses');
        const glossesData = await glossesResp.json();
        glossCategories = glossesData.categories || {};
        allGlosses = glossesData.all_glosses || [];
        populateCategoryTabs();

    } catch (error) {
        console.error('Error loading data:', error);
    }
}

function populatePresetDropdown() {
    presetDropdown.innerHTML = '<option value="">-- Select a preset sentence --</option>';

    presetSentences.forEach(sentence => {
        const option = document.createElement('option');
        option.value = sentence.id;
        option.textContent = sentence.reference;
        option.dataset.glosses = JSON.stringify(sentence.glosses);
        presetDropdown.appendChild(option);
    });
}

function populateCategoryTabs() {
    categoryTabs.innerHTML = '';

    const categories = Object.keys(glossCategories);
    categories.forEach((category, index) => {
        const tab = document.createElement('button');
        tab.className = 'category-tab' + (index === 0 ? ' active' : '');
        tab.textContent = category;
        tab.dataset.category = category;
        tab.addEventListener('click', () => selectCategory(category));
        categoryTabs.appendChild(tab);
    });

    // Select first category
    if (categories.length > 0) {
        selectCategory(categories[0]);
    }
}

function selectCategory(category) {
    selectedCategory = category;

    // Update tab styles
    document.querySelectorAll('.category-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.category === category);
    });

    // Populate words grid
    populateWordsGrid(category);
}

function populateWordsGrid(category) {
    wordsGrid.innerHTML = '';

    const words = glossCategories[category] || [];
    words.forEach(word => {
        const wordBtn = document.createElement('button');
        wordBtn.className = 'word-btn';
        if (selectedGlosses.includes(word)) {
            wordBtn.classList.add('selected');
        }
        wordBtn.textContent = word;
        wordBtn.addEventListener('click', () => toggleWord(word));
        wordsGrid.appendChild(wordBtn);
    });
}

function toggleWord(word) {
    const index = selectedGlosses.indexOf(word);
    if (index === -1) {
        // Add word
        selectedGlosses.push(word);
    } else {
        // Remove word
        selectedGlosses.splice(index, 1);
    }

    updateSelectedWordsDisplay();
    updateButtonStates();

    // Refresh word grid to show selection state
    if (selectedCategory) {
        populateWordsGrid(selectedCategory);
    }

    // Clear preset selection when building custom
    if (selectedGlosses.length > 0) {
        presetDropdown.value = '';
        document.getElementById('preset-info').style.display = 'none';
    }

    // Hide generated section when words change
    generatedSection.style.display = 'none';
    finalSection.style.display = 'none';
    continueBtn.disabled = true;
}

function updateSelectedWordsDisplay() {
    if (selectedGlosses.length === 0) {
        selectedWordsDiv.innerHTML = '<span class="placeholder-text">Click words above to add them</span>';
    } else {
        selectedWordsDiv.innerHTML = selectedGlosses.map((word, idx) => `
            <span class="selected-word" data-index="${idx}">
                ${word}
                <button class="remove-word" data-word="${word}">&times;</button>
            </span>
        `).join('');

        // Add remove handlers
        document.querySelectorAll('.remove-word').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const word = btn.dataset.word;
                const index = selectedGlosses.indexOf(word);
                if (index !== -1) {
                    selectedGlosses.splice(index, 1);
                    updateSelectedWordsDisplay();
                    updateButtonStates();
                    if (selectedCategory) {
                        populateWordsGrid(selectedCategory);
                    }
                }
            });
        });
    }
}

function updateButtonStates() {
    const hasWords = selectedGlosses.length > 0;
    clearBtn.disabled = !hasWords;
    removeLastBtn.disabled = !hasWords;
    generateBtn.disabled = !hasWords;
}

function setupEventListeners() {
    // Preset dropdown
    presetDropdown.addEventListener('change', handlePresetSelect);

    // Build section buttons
    clearBtn.addEventListener('click', clearAllWords);
    removeLastBtn.addEventListener('click', removeLastWord);
    generateBtn.addEventListener('click', generateReference);

    // Generated section buttons
    regenerateBtn.addEventListener('click', generateReference);
    editBtn.addEventListener('click', enableEdit);
    useGeneratedBtn.addEventListener('click', useGeneratedSentence);

    // Continue button
    continueBtn.addEventListener('click', proceedToLearn);
}

function handlePresetSelect() {
    const selectedOption = presetDropdown.options[presetDropdown.selectedIndex];

    if (!selectedOption.value) {
        document.getElementById('preset-info').style.display = 'none';
        finalSection.style.display = 'none';
        continueBtn.disabled = true;
        return;
    }

    const glosses = JSON.parse(selectedOption.dataset.glosses || '[]');

    // Show preset info
    const presetInfo = document.getElementById('preset-info');
    presetInfo.style.display = 'block';
    document.getElementById('preset-glosses-list').textContent = glosses.join(' → ');

    // Set final selection
    finalReference = selectedOption.textContent;
    finalGlosses = glosses;

    // Clear custom build section
    selectedGlosses = [];
    updateSelectedWordsDisplay();
    updateButtonStates();
    generatedSection.style.display = 'none';
    if (selectedCategory) {
        populateWordsGrid(selectedCategory);
    }

    // Show final selection
    showFinalSelection();
}

function clearAllWords() {
    selectedGlosses = [];
    updateSelectedWordsDisplay();
    updateButtonStates();
    generatedSection.style.display = 'none';
    finalSection.style.display = 'none';
    continueBtn.disabled = true;
    if (selectedCategory) {
        populateWordsGrid(selectedCategory);
    }
}

function removeLastWord() {
    if (selectedGlosses.length > 0) {
        selectedGlosses.pop();
        updateSelectedWordsDisplay();
        updateButtonStates();
        generatedSection.style.display = 'none';
        finalSection.style.display = 'none';
        continueBtn.disabled = true;
        if (selectedCategory) {
            populateWordsGrid(selectedCategory);
        }
    }
}

async function generateReference() {
    if (selectedGlosses.length === 0) return;

    generateBtn.disabled = true;
    generateBtn.textContent = 'Generating...';

    try {
        const response = await fetch('/api/generate-reference', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ glosses: selectedGlosses })
        });

        const data = await response.json();

        if (data.success) {
            generatedInput.value = data.reference;
            generatedInput.readOnly = true;
            generatedSection.style.display = 'block';
            editBtn.textContent = 'Edit';
        } else {
            alert('Failed to generate reference: ' + (data.error || 'Unknown error'));
        }

    } catch (error) {
        console.error('Error generating reference:', error);
        alert('Error generating reference sentence');
    } finally {
        generateBtn.disabled = false;
        generateBtn.textContent = 'Generate Reference';
    }
}

function enableEdit() {
    if (generatedInput.readOnly) {
        generatedInput.readOnly = false;
        generatedInput.focus();
        editBtn.textContent = 'Done';
    } else {
        generatedInput.readOnly = true;
        editBtn.textContent = 'Edit';
    }
}

function useGeneratedSentence() {
    finalReference = generatedInput.value.trim();
    finalGlosses = [...selectedGlosses];

    if (!finalReference) {
        alert('Please generate or enter a reference sentence');
        return;
    }

    // Clear preset selection
    presetDropdown.value = '';
    document.getElementById('preset-info').style.display = 'none';

    showFinalSelection();
}

function showFinalSelection() {
    finalSentence.textContent = finalReference;
    finalGlossesSpan.textContent = finalGlosses.join(' → ');
    finalSection.style.display = 'block';
    continueBtn.disabled = false;

    // Scroll to final section
    finalSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function proceedToLearn() {
    if (!finalReference || finalGlosses.length === 0) {
        alert('Please select or build a reference sentence first');
        return;
    }

    // Save to session storage
    sessionStorage.setItem('liveReference', finalReference);
    sessionStorage.setItem('liveGlosses', JSON.stringify(finalGlosses));

    // Navigate to Learn page
    window.location.href = '/live-learn';
}
