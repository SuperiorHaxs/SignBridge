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
    let _lastGlosses   = null;        // confident-only glosses sent to the LLM
                                       // (kept around so Regenerate can re-call
                                       // /api/construct-paragraph without re-running
                                       // the heavy recognition pipeline)
    let _lastDomain    = null;
    let _regenInFlight = false;

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
        const regenBtn    = $('uploadRegenBtn');
        const editBtn     = $('uploadEditBtn');
        if (copyBtn)     copyBtn.addEventListener('click', _copyParagraph);
        if (downloadBtn) downloadBtn.addEventListener('click', _downloadParagraph);
        if (anotherBtn)  anotherBtn.addEventListener('click', _reset);
        if (regenBtn)    regenBtn.addEventListener('click', _regenerateParagraph);
        if (editBtn)     editBtn.addEventListener('click', _enterParagraphEdit);
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

        // Cache the inputs the Regenerate button needs to re-call
        // /api/construct-paragraph without re-running recognition.
        _lastGlosses = confidentGlss;
        _lastDomain  = domain;

        _lastResult = {
            sentences,
            paragraphText: sentences.join(' '),
            edited:        false,         // becomes true after manual Edit
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
        _lastGlosses  = null;
        _lastDomain   = null;
        _regenInFlight = false;
        $('uploadProgress').hidden = true;
        $('uploadResult').hidden   = true;
        $('uploadDropZone').hidden = false;
        $('uploadFileInput').value = '';
        const vid = $('uploadVideoPreview');
        if (vid) { vid.removeAttribute('src'); vid.load && vid.load(); }
        const para = $('uploadParagraph');
        if (para) { para.textContent = ''; para.hidden = false; }
        $('uploadSummary').innerHTML = '';
        // Strip any leftover edit-mode textarea + actions from a prior session.
        const block = $('uploadParagraphBlock');
        if (block) {
            block.classList.remove('upload-paragraph-edited');
            block.querySelectorAll('.upload-paragraph-edit, .upload-paragraph-edit-actions').forEach(n => n.remove());
        }
        const chips = $('uploadParagraphChips');
        if (chips) chips.hidden = false;
        _clearError();
    }

    // ── Result rendering ──────────────────────────────────────────
    // Renders the entire paragraph as a single text block plus paragraph-
    // level chips (Regenerate, Edit). No per-sentence cards -- the LLM
    // sometimes mis-segments the sentence boundaries anyway, and at this
    // stage the user just wants the whole narrative to read well.
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

        _renderSummary(r);
        _renderParagraphText(r);
    }

    function _renderSummary(r) {
        const sumEl = $('uploadSummary');
        if (!sumEl) return;
        const ctqiHtml = r.ctqi != null
            ? `<span class="${r.ctqi >= _ctqiLow() ? 'ctqi-good' : 'ctqi-bad'}">CTQI = ${r.ctqi.toFixed(0)}</span>`
            : `<span class="ctqi-missing">CTQI = ?</span>`;
        const unclearHtml = r.unclearCount > 0
            ? `<span class="upload-summary-warn">⚠ ${r.unclearCount} sign${r.unclearCount === 1 ? '' : 's'} not recognized</span>`
            : '';
        const editedHtml = r.edited
            ? `<span class="upload-summary-stat"><b>(edited)</b></span>`
            : '';
        sumEl.innerHTML =
            `<span class="upload-summary-stat"><b>${r.segmentCount}</b> sign${r.segmentCount === 1 ? '' : 's'} detected</span>` +
            ` · <span class="upload-summary-stat"><b>${r.sentences.length}</b> sentence${r.sentences.length === 1 ? '' : 's'}</span>` +
            ` · ${ctqiHtml}` +
            (unclearHtml ? ` · ${unclearHtml}` : '') +
            (editedHtml  ? ` · ${editedHtml}`  : '');
    }

    function _renderParagraphText(r) {
        const block = $('uploadParagraphBlock');
        const para  = $('uploadParagraph');
        const chips = $('uploadParagraphChips');
        if (!block || !para || !chips) return;
        block.classList.toggle('upload-paragraph-edited', !!r.edited);
        para.textContent = r.paragraphText;
        para.hidden = false;
        // Edit mode (if active) replaces these chips with Save/Cancel; restore.
        chips.hidden = false;
        // Remove any leftover edit textarea from a prior edit session.
        const leftover = block.querySelector('.upload-paragraph-edit');
        if (leftover) leftover.remove();
    }

    // ── Regenerate (paragraph-level) ──────────────────────────────
    // Re-calls /api/construct-paragraph with previous_attempt set to the
    // current paragraph, so the LLM is instructed to explore top-2/top-3
    // alternates and try a different sentence structure. Does NOT re-run
    // recognition -- the gloss list is whatever was recognized at upload time.
    async function _regenerateParagraph() {
        if (_regenInFlight) return;
        if (!_lastResult || !_lastGlosses || !_lastGlosses.length) return;
        _regenInFlight = true;

        const regenBtn = $('uploadRegenBtn');
        const editBtn  = $('uploadEditBtn');
        const para     = $('uploadParagraph');
        const prevText = para ? (para.textContent || '') : (_lastResult.paragraphText || '');

        if (regenBtn) regenBtn.disabled = true;
        if (editBtn)  editBtn.disabled  = true;
        if (para)     para.classList.add('upload-paragraph-loading');

        try {
            const resp = await fetch('/api/construct-paragraph', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    gloss_predictions:  _lastGlosses,
                    domain:             _lastDomain,
                    previous_attempt:   prevText,
                    conversation_history: [],
                }),
            });
            const data = await resp.json();
            const sentences = (data && data.sentences && data.sentences.length)
                ? data.sentences
                : (data && data.fallback) || [];
            if (!sentences.length) {
                _showError('Regenerate returned no sentences. Original paragraph kept.');
                return;
            }
            const plausibility = (data && typeof data.plausibility === 'number') ? data.plausibility : null;
            // Recompute CTQI with the new plausibility but the same gloss
            // confidences (recognition didn't change).
            const allSegs = _lastGlosses;
            let ctqi = null;
            if (plausibility != null && allSegs.length > 0) {
                const avgConf = allSegs.reduce((s, g) => s + (g.confidence || 0), 0) / allSegs.length;
                ctqi = (avgConf * 100) * (0.5 + 0.5 * plausibility / 100);
            }
            _lastResult.sentences     = sentences;
            _lastResult.paragraphText = sentences.join(' ');
            _lastResult.plausibility  = plausibility;
            _lastResult.ctqi          = ctqi;
            _lastResult.edited        = false;   // a regen replaces any prior edit
            _renderSummary(_lastResult);
            _renderParagraphText(_lastResult);
            console.log('[Upload] regenerate ok:', sentences.length, 'sentence(s), CTQI=', ctqi);
        } catch (e) {
            _showError(`Regenerate failed: ${e.message}`);
        } finally {
            if (regenBtn) regenBtn.disabled = false;
            if (editBtn)  editBtn.disabled  = false;
            if (para)     para.classList.remove('upload-paragraph-loading');
            _regenInFlight = false;
        }
    }

    // ── Edit (paragraph-level) ────────────────────────────────────
    // Swaps the paragraph text for a textarea and replaces the regen/edit
    // chips with Save / Cancel. Save commits the user's text as the new
    // paragraph (sentences is re-derived by sentence-splitting on . ! ?).
    function _enterParagraphEdit() {
        if (!_lastResult) return;
        const block = $('uploadParagraphBlock');
        const para  = $('uploadParagraph');
        const chips = $('uploadParagraphChips');
        if (!block || !para || !chips) return;
        // Already in edit mode? no-op.
        if (block.querySelector('.upload-paragraph-edit')) return;

        const cur = _lastResult.paragraphText || '';

        para.hidden = true;
        chips.hidden = true;

        const ta = document.createElement('textarea');
        ta.className = 'upload-paragraph-edit';
        ta.value = cur;
        ta.rows = Math.max(4, Math.ceil(cur.length / 70));
        block.appendChild(ta);

        const actions = document.createElement('div');
        actions.className = 'upload-paragraph-edit-actions';
        const saveBtn = document.createElement('button');
        saveBtn.type = 'button';
        saveBtn.className = 'caption-action';
        saveBtn.title = 'Save edit';
        saveBtn.textContent = '✓';
        const cancelBtn = document.createElement('button');
        cancelBtn.type = 'button';
        cancelBtn.className = 'caption-action';
        cancelBtn.title = 'Cancel';
        cancelBtn.textContent = '✗';
        actions.appendChild(saveBtn);
        actions.appendChild(cancelBtn);
        block.appendChild(actions);

        ta.focus();
        ta.select();

        const cleanup = () => {
            ta.remove();
            actions.remove();
            para.hidden = false;
            chips.hidden = false;
        };
        const commit = (newText) => {
            _lastResult.paragraphText = newText;
            // Re-derive sentences by splitting on terminal punctuation so
            // Copy / Download still get sentence-segmented output.
            const split = newText.split(/(?<=[.!?])\s+/).map(s => s.trim()).filter(Boolean);
            _lastResult.sentences = split.length ? split : [newText];
            _lastResult.edited = true;
            cleanup();
            _renderSummary(_lastResult);
            _renderParagraphText(_lastResult);
        };
        saveBtn.addEventListener('click', () => commit((ta.value || '').trim()));
        cancelBtn.addEventListener('click', () => { cleanup(); });
        ta.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); commit((ta.value || '').trim()); }
            else if (e.key === 'Escape')                       { e.preventDefault(); cleanup(); }
        });
    }

    // ── Copy + download ───────────────────────────────────────────
    function _joinParagraph() {
        if (!_lastResult) return '';
        // Prefer the live paragraph text (reflects any in-flight edit that
        // hasn't been committed yet would have been swapped back anyway).
        return _lastResult.paragraphText
            || (_lastResult.sentences ? _lastResult.sentences.join(' ') : '');
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
