// ══════════════════════════════════════════════════════════════
// Upload tab -- paragraph-length signing video -> narrative transcript
// (Backlog item 10)
//
// Flow:
//   1. User picks a video file (drag-drop or browse).
//   2. POST to /api/process-signing-clip with segmenter=motion_gate
//      (and persist=true so the source video is also pushed to R2 when
//      PERSIST_UPLOADS is on server-side).
//   3. POST the resulting gloss list to /api/construct-paragraph; the
//      LLM returns sentences[] + overall plausibility.
//   4. Render result: video preview (R2 URL if persisted, else local
//      blob), per-sentence cards with Edit chip, copy / download.
//
// Self-contained module. Reads `uploadDomainSelect` (populated by
// app.js loadDomainsToUI), nothing else from app.js's globals.
// ══════════════════════════════════════════════════════════════

(function () {
    'use strict';

    // ── Configurable knobs ────────────────────────────────────────
    // Match the CTQI thresholds the live/batch flow uses so the user
    // sees a consistent "good vs bad" signal between tabs. These are
    // read from app.js globals when available (defaults match its
    // defaults) so changing them in Settings flows through here too.
    const _ctqiLow   = () => (typeof CTQI_LOW_THRESHOLD === 'number' ? CTQI_LOW_THRESHOLD : 80);
    const _ctqiFloor = () => (typeof CTQI_HARD_FLOOR    === 'number' ? CTQI_HARD_FLOOR    : 40);

    // Max upload size mirrored from the server (Flask MAX_CONTENT_LENGTH).
    // Pre-flight check so we fail fast with a clear message instead of a
    // mysterious 413.
    const MAX_UPLOAD_BYTES = 100 * 1024 * 1024;

    // ── State ─────────────────────────────────────────────────────
    let _selectedFile  = null;
    let _previewBlobUrl = null;       // URL.createObjectURL handle to revoke on reset
    let _lastResult    = null;        // last server response (for copy/download)

    // ── DOM lookups (helper) ──────────────────────────────────────
    const $ = (id) => document.getElementById(id);

    // ── Public-ish init: wire DOM events. Called from page-load init. ─
    function init() {
        const zone  = $('uploadDropZone');
        const input = $('uploadFileInput');
        if (!zone || !input) return;   // upload tab markup not present (shouldn't happen)

        // Click anywhere on the drop zone -> open file picker
        zone.addEventListener('click', () => input.click());

        zone.addEventListener('dragover', (e) => {
            e.preventDefault();
            zone.classList.add('upload-drop-zone-active');
        });
        zone.addEventListener('dragleave', () => zone.classList.remove('upload-drop-zone-active'));
        zone.addEventListener('drop', (e) => {
            e.preventDefault();
            zone.classList.remove('upload-drop-zone-active');
            const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
            if (f) _handleFile(f);
        });
        input.addEventListener('change', (e) => {
            const f = e.target.files && e.target.files[0];
            if (f) _handleFile(f);
        });

        const copyBtn     = $('uploadCopyBtn');
        const downloadBtn = $('uploadDownloadBtn');
        const anotherBtn  = $('uploadAnotherBtn');
        if (copyBtn)     copyBtn.addEventListener('click', _copyParagraph);
        if (downloadBtn) downloadBtn.addEventListener('click', _downloadParagraph);
        if (anotherBtn)  anotherBtn.addEventListener('click', _reset);
    }

    // ── File entry point ──────────────────────────────────────────
    async function _handleFile(file) {
        if (!file) return;
        if (!file.type || !file.type.startsWith('video/')) {
            _showError(`Not a video file (got "${file.type || 'unknown'}").`);
            return;
        }
        if (file.size > MAX_UPLOAD_BYTES) {
            const mb = (file.size / 1024 / 1024).toFixed(1);
            _showError(`File is ${mb} MB — over the 100 MB limit. Trim the video or split into shorter clips.`);
            return;
        }
        _selectedFile = file;
        _enterProcessing(file.name);
        await _runPipeline(file);
    }

    // ── Pipeline: process-signing-clip -> construct-paragraph ─────
    async function _runPipeline(file) {
        const domain = ($('uploadDomainSelect') && $('uploadDomainSelect').value) || 'emergency';

        // ----- Step 1: segment + per-sign inference (server) -----
        _setStage(`Uploading ${(file.size/1024/1024).toFixed(1)} MB and recognizing signs…`);
        let clipResult;
        try {
            const fd = new FormData();
            fd.append('video', file, file.name);
            fd.append('domain', domain);
            fd.append('segmenter', 'motion_gate');
            fd.append('persist', 'true');
            const resp = await fetch('/api/process-signing-clip', { method: 'POST', body: fd });
            clipResult = await resp.json();
        } catch (e) {
            _showError(`Upload failed: ${e.message}`);
            return;
        }
        if (!clipResult || !clipResult.success) {
            _showError(clipResult && clipResult.error
                ? `Recognition failed: ${clipResult.error}`
                : 'Recognition failed (no details from server).');
            return;
        }
        const allGlosses     = clipResult.glosses || [];
        const confidentGlss  = allGlosses.filter(g => g.confident !== false);
        const unclearGlosses = allGlosses.filter(g => g.confident === false);
        console.log(`[Upload] clip processed: ${clipResult.segment_count} segments, ` +
                    `${confidentGlss.length} confident + ${unclearGlosses.length} unclear, ` +
                    `dur=${clipResult.duration_sec}s, total_time=${clipResult.total_time}s, ` +
                    `r2=${clipResult.r2_video_url || '(not persisted)'}`);

        if (confidentGlss.length === 0) {
            _showError(`No signs were confidently recognized (${clipResult.segment_count || 0} segments detected). ` +
                       `Try a clearer recording, or pause more between signs.`);
            return;
        }

        // ----- Step 2: build narrative paragraph (server / LLM) -----
        _setStage(`Constructing narrative from ${confidentGlss.length} sign${confidentGlss.length === 1 ? '' : 's'}…`);
        let paraResult;
        try {
            const resp = await fetch('/api/construct-paragraph', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    gloss_predictions: confidentGlss,
                    domain,
                    conversation_history: [],
                }),
            });
            paraResult = await resp.json();
        } catch (e) {
            _showError(`Sentence construction failed: ${e.message}`);
            return;
        }

        const sentences = (paraResult && paraResult.sentences && paraResult.sentences.length)
            ? paraResult.sentences
            : (paraResult && paraResult.fallback) || [];
        if (!sentences.length) {
            _showError('LLM returned no sentences.');
            return;
        }

        // CTQI v3 — GA includes unrecognized segments so dropped signs penalize
        // the score; matches the formula used by the batch path.
        const allSegs = confidentGlss.concat(unclearGlosses);
        const plausibility = (paraResult && typeof paraResult.plausibility === 'number')
            ? paraResult.plausibility : null;
        let ctqi = null;
        if (plausibility != null && allSegs.length > 0) {
            const avgConf = allSegs.reduce((s, g) => s + (g.confidence || 0), 0) / allSegs.length;
            ctqi = (avgConf * 100) * (0.5 + 0.5 * plausibility / 100);
        }

        _lastResult = {
            sentences,
            ctqi,
            plausibility,
            unclearCount: unclearGlosses.length,
            segmentCount: clipResult.segment_count,
            durationSec:  clipResult.duration_sec,
            totalTime:    clipResult.total_time,
            r2VideoUrl:   clipResult.r2_video_url,
            uploadId:     clipResult.upload_id,
            file,
        };
        _renderResult(_lastResult);
    }

    // ── UI state helpers ──────────────────────────────────────────
    function _enterProcessing(filename) {
        $('uploadDropZone').hidden = true;
        $('uploadResult').hidden   = true;
        $('uploadProgress').hidden = false;
        $('uploadFileName').textContent = filename;
        _setStage('Preparing…');
    }
    function _setStage(text) {
        const el = $('uploadStageLabel');
        if (el) el.textContent = text;
    }
    function _showError(msg) {
        $('uploadProgress').hidden = true;
        $('uploadResult').hidden   = true;
        $('uploadDropZone').hidden = false;
        const zone = $('uploadDropZone');
        // Surface inline in the drop zone area -- no toast machinery needed.
        let err = $('uploadErrorMsg');
        if (!err) {
            err = document.createElement('div');
            err.id = 'uploadErrorMsg';
            err.className = 'upload-error';
            zone.parentNode.insertBefore(err, zone);
        }
        err.textContent = '⚠ ' + msg;
        err.hidden = false;
        console.warn('[Upload]', msg);
    }
    function _clearError() {
        const err = $('uploadErrorMsg');
        if (err) err.hidden = true;
    }
    function _reset() {
        if (_previewBlobUrl) { try { URL.revokeObjectURL(_previewBlobUrl); } catch (_) {} _previewBlobUrl = null; }
        _selectedFile = null;
        _lastResult   = null;
        $('uploadProgress').hidden = true;
        $('uploadResult').hidden   = true;
        $('uploadDropZone').hidden = false;
        $('uploadFileInput').value = '';
        const vid = $('uploadVideoPreview');
        if (vid) { vid.removeAttribute('src'); vid.load && vid.load(); }
        $('uploadParagraph').innerHTML = '';
        $('uploadSummary').innerHTML   = '';
        _clearError();
    }

    // ── Result rendering ──────────────────────────────────────────
    function _renderResult(r) {
        _clearError();
        $('uploadProgress').hidden = true;
        $('uploadResult').hidden   = false;

        // Video preview: prefer the R2 URL (so the same view works on a
        // shared/demo link) but fall back to a local blob URL for the case
        // where persistence was off or the R2 upload failed.
        const vid = $('uploadVideoPreview');
        if (vid) {
            if (r.r2VideoUrl) {
                vid.src = r.r2VideoUrl;
            } else if (r.file) {
                _previewBlobUrl = URL.createObjectURL(r.file);
                vid.src = _previewBlobUrl;
            }
        }

        // Summary bar: signs, sentences, CTQI, unrecognized count.
        const sumEl = $('uploadSummary');
        if (sumEl) {
            const ctqiHtml = r.ctqi != null
                ? `<span class="${r.ctqi >= _ctqiLow() ? 'ctqi-good' : 'ctqi-bad'}">CTQI = ${r.ctqi.toFixed(0)}</span>`
                : `<span class="ctqi-missing">CTQI = ?</span>`;
            const unclearHtml = r.unclearCount > 0
                ? `<span class="upload-summary-warn">⚠ ${r.unclearCount} sign${r.unclearCount === 1 ? '' : 's'} not recognized</span>`
                : '';
            sumEl.innerHTML =
                `<span class="upload-summary-stat"><b>${r.segmentCount}</b> sign${r.segmentCount === 1 ? '' : 's'} detected</span>` +
                ` · <span class="upload-summary-stat"><b>${r.sentences.length}</b> sentence${r.sentences.length === 1 ? '' : 's'}</span>` +
                ` · ${ctqiHtml}` +
                (unclearHtml ? ` · ${unclearHtml}` : '');
        }

        // Per-sentence cards
        const paraEl = $('uploadParagraph');
        if (paraEl) {
            paraEl.innerHTML = '';
            r.sentences.forEach((sent, idx) => {
                paraEl.appendChild(_buildSentenceCard(sent, idx));
            });
        }
    }

    function _buildSentenceCard(text, idx) {
        const card = document.createElement('div');
        card.className = 'upload-sentence-card';
        card.dataset.idx = String(idx);

        const textEl = document.createElement('div');
        textEl.className = 'upload-sentence-text';
        textEl.textContent = text;
        card.appendChild(textEl);

        const actionsEl = document.createElement('div');
        actionsEl.className = 'upload-sentence-actions';
        const editBtn = document.createElement('button');
        editBtn.type = 'button';
        editBtn.className = 'caption-action';
        editBtn.title = 'Edit this sentence';
        editBtn.textContent = '✏️';
        editBtn.addEventListener('click', () => _enterEdit(card));
        actionsEl.appendChild(editBtn);
        card.appendChild(actionsEl);

        return card;
    }

    function _enterEdit(card) {
        if (card.dataset.editing === '1') return;
        card.dataset.editing = '1';
        const idx = parseInt(card.dataset.idx, 10);
        const cur = _lastResult.sentences[idx] || '';

        card.innerHTML = '';
        const input = document.createElement('textarea');
        input.className = 'upload-sentence-edit';
        input.value = cur;
        input.rows = Math.max(2, Math.ceil(cur.length / 60));
        card.appendChild(input);

        const actions = document.createElement('div');
        actions.className = 'upload-sentence-actions';
        const saveBtn = document.createElement('button');
        saveBtn.type = 'button';
        saveBtn.className = 'caption-action';
        saveBtn.title = 'Save';
        saveBtn.textContent = '✓';
        const cancelBtn = document.createElement('button');
        cancelBtn.type = 'button';
        cancelBtn.className = 'caption-action';
        cancelBtn.title = 'Cancel';
        cancelBtn.textContent = '✗';
        actions.appendChild(saveBtn);
        actions.appendChild(cancelBtn);
        card.appendChild(actions);
        input.focus();
        input.select();

        const commit = (newText) => {
            _lastResult.sentences[idx] = newText;
            const replacement = _buildSentenceCard(newText, idx);
            // Mark edited so the user sees their change is in the saved state.
            replacement.classList.add('upload-sentence-edited');
            card.parentNode.replaceChild(replacement, card);
        };
        saveBtn.addEventListener('click', () => commit((input.value || '').trim()));
        cancelBtn.addEventListener('click', () => commit(cur));
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); commit((input.value || '').trim()); }
            else if (e.key === 'Escape')                       { e.preventDefault(); commit(cur); }
        });
    }

    // ── Copy + download ───────────────────────────────────────────
    function _joinParagraph() {
        if (!_lastResult || !_lastResult.sentences) return '';
        return _lastResult.sentences.join(' ');
    }
    function _copyParagraph() {
        const text = _joinParagraph();
        if (!text) return;
        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(text).then(
                () => _flashBtn($('uploadCopyBtn'), 'Copied!'),
                () => _flashBtn($('uploadCopyBtn'), 'Copy failed'),
            );
        } else {
            // Fallback: select+execCommand
            const ta = document.createElement('textarea');
            ta.value = text;
            document.body.appendChild(ta);
            ta.select();
            try { document.execCommand('copy'); _flashBtn($('uploadCopyBtn'), 'Copied!'); }
            catch (_) { _flashBtn($('uploadCopyBtn'), 'Copy failed'); }
            document.body.removeChild(ta);
        }
    }
    function _downloadParagraph() {
        const text = _joinParagraph();
        if (!text) return;
        const blob = new Blob([text + '\n'], { type: 'text/plain;charset=utf-8' });
        const url  = URL.createObjectURL(blob);
        const a    = document.createElement('a');
        a.href     = url;
        const name = (_selectedFile && _selectedFile.name) || 'transcript';
        a.download = name.replace(/\.[a-z0-9]+$/i, '') + '.txt';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        setTimeout(() => URL.revokeObjectURL(url), 1000);
    }
    function _flashBtn(btn, msg) {
        if (!btn) return;
        const orig = btn.textContent;
        btn.textContent = msg;
        setTimeout(() => { btn.textContent = orig; }, 1500);
    }

    // ── Boot: wire as soon as the DOM is ready. ───────────────────
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        // Loaded after DOM (cache-busted script at end of body)
        init();
    }
})();
