"""
capture_routes.py
-----------------
Data-capture feature for SignBridge: record an isolated sign in the app and save
it as training data for that domain's isolated classifier. Isolated as a Flask
Blueprint so it doesn't touch the main app's existing routes -- app.py registers
it with a single call (see register_capture_routes).

Endpoints
    GET  /api/capture/domains   list domains the registry offers
    GET  /api/capture/glosses   a domain's active glosses (for the UI dropdown)
    POST /api/capture           save one recorded sign as a capture
    GET  /api/captures          per-gloss inventory of captures for a domain
    GET  /capture               the self-contained capture page

Flow: webcam video -> _extract_poses (83-pt, reused from app.py) -> keypoints
record -> capture_store (R2 or local) under captures/<domain>/<GLOSS>/<clip>.pkl.
Folding captures into a model + retraining is done OFFLINE by the local admin
utility kaggle-training/scripts/retrain_from_captures.py (not in the app).

The app holds the heavy deps (pose extractor, video decoder); they're injected
via register_capture_routes to avoid importing app.py back (no circular import).
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np
from flask import Blueprint, jsonify, request

import capture_store  # R2-backed (local-fallback) capture storage; shared with the retrain utility

capture_bp = Blueprint("capture", __name__)

# Injected by register_capture_routes()
_DEPS: dict = {}

_SPLITS = ("train", "val", "test")
# Glosses can be multi-word (e.g. "HIGH SCHOOL", "A LOT") -- allow spaces,
# apostrophes and hyphens, matching the augmented-pool gloss directory names.
_VALID_GLOSS = re.compile(r"^[A-Z0-9_ '\-]+$")


# ---------------------------------------------------------------------------
# POST /api/capture  -- save one recorded sign
# ---------------------------------------------------------------------------
@capture_bp.route("/api/capture", methods=["POST"])
def capture():
    if "video" not in request.files:
        return jsonify({"success": False, "error": "No video provided"}), 400
    gloss = (request.form.get("gloss") or "").strip().upper()
    domain = (request.form.get("domain") or "education").strip().lower()
    split = (request.form.get("split") or "train").strip().lower()
    signer = (request.form.get("signer_id") or "user").strip() or "user"
    if not gloss or not _VALID_GLOSS.match(gloss):
        return jsonify({"success": False, "error": f"Invalid gloss '{gloss}'"}), 400
    if split not in _SPLITS:
        return jsonify({"success": False, "error": f"split must be one of {_SPLITS}"}), 400

    video_bytes = request.files["video"].read()
    if not video_bytes:
        return jsonify({"success": False, "error": "Empty video"}), 400

    try:
        frames = _DEPS["decode_video_bytes"](video_bytes)
        if not frames:
            return jsonify({"success": False, "error": "Could not decode video"}), 400
        # Reuse the app's exact 83-point extractor + cleaning so captures match
        # the rest of the pipeline's feature space.
        pose = _DEPS["extract_poses"](frames, target_frames=min(len(frames), 90))
        if pose is None or len(pose) == 0:
            return jsonify({"success": False, "error": "No pose detected -- make sure hands are visible"}), 400
        for fn_name in ("reject_outliers", "drop_sparse_hands", "smooth_missing"):
            fn = _DEPS.get(fn_name)
            if fn is not None:
                pose = fn(pose)
        pose = np.asarray(pose, dtype=np.float32)  # (N, 83, 4)

        # Sanitize the gloss for the clip_id/filename (spaces -> _); the real
        # gloss (with spaces) is preserved in the record and the R2 key path.
        safe_gloss = re.sub(r"[^A-Z0-9]+", "_", gloss)
        clip_id = f"user_{safe_gloss}_{int(time.time() * 1000)}"
        record = {
            "keypoints": pose,            # (N, 83, 4) raw MediaPipe; ingest normalizes
            "gloss": gloss,
            "domain": domain,
            "split": split,
            "signer_id": signer,
            "clip_id": clip_id,
            "source": "user_capture",
            "captured_ms": int(time.time() * 1000),
        }
        # Store to R2 (durable across HF Space restarts) or local disk if R2 creds
        # aren't set. Same layout either way, so the retrain utility reads both.
        info = capture_store.save_capture(domain, gloss, clip_id, record, webm_bytes=video_bytes)

        return jsonify({"success": True, "clip_id": clip_id, "gloss": gloss, "domain": domain,
                        "split": split, "frames": int(len(pose)), "backend": info["backend"]})
    except Exception as e:  # noqa: BLE001
        return jsonify({"success": False, "error": f"{type(e).__name__}: {e}"}), 500


# ---------------------------------------------------------------------------
# GET /api/captures  -- inventory
# ---------------------------------------------------------------------------
@capture_bp.route("/api/captures", methods=["GET"])
def list_captures():
    domain = (request.args.get("domain") or "education").strip().lower()
    inv = capture_store.inventory(domain)
    return jsonify({"success": True, "domain": domain, **inv})


# ---------------------------------------------------------------------------
# GET /api/capture/domains  -- domains the registry offers (for the UI dropdown)
# ---------------------------------------------------------------------------
@capture_bp.route("/api/capture/domains", methods=["GET"])
def capture_domains():
    """List every domain in the registry so the capture page can offer them all
    (no hard-coded domain)."""
    pr = _DEPS["project_root"]
    reg_path = pr / "models" / "openhands-modernized" / "production-models" / "registry.json"
    try:
        reg = json.load(open(reg_path, encoding="utf-8"))
        domains = reg.get("domains", reg)
        return jsonify({"success": True, "domains": sorted(domains.keys())})
    except Exception as e:  # noqa: BLE001
        return jsonify({"success": False, "error": f"{type(e).__name__}: {e}"}), 500


@capture_bp.route("/api/capture/glosses", methods=["GET"])
def capture_glosses():
    domain = request.args.get("domain", "emergency")
    pr = _DEPS["project_root"]
    reg_path = pr / "models" / "openhands-modernized" / "production-models" / "registry.json"
    try:
        reg = json.load(open(reg_path, encoding="utf-8"))
        model_dir = reg["domains"][domain]["model_dir"]
        mc = json.load(open(reg_path.parent / model_dir / "masked_classes.json", encoding="utf-8"))
        glosses = sorted(str(g).upper() for g in (mc.get("active_glosses") or []))
        return jsonify({"success": True, "domain": domain, "glosses": glosses})
    except Exception as e:  # noqa: BLE001
        return jsonify({"success": False, "error": f"{type(e).__name__}: {e}"}), 500


# ---------------------------------------------------------------------------
# GET /capture  -- self-contained capture page (no dependency on the SPA)
# ---------------------------------------------------------------------------
@capture_bp.route("/capture", methods=["GET"])
def capture_page():
    return _CAPTURE_HTML


_CAPTURE_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>SignBridge — Data Capture</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root{--bg:#0f1116;--card:#1a1d26;--ink:#e8eaf0;--mut:#9aa0ad;--acc:#4f8cff;--ok:#2ecc71;--err:#ff5c5c}
  *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,Segoe UI,Roboto,sans-serif}
  .wrap{max-width:920px;margin:0 auto;padding:20px}
  h1{font-size:20px;margin:0 0 4px} .sub{color:var(--mut);margin:0 0 18px}
  .card{background:var(--card);border:1px solid #262a36;border-radius:12px;padding:16px;margin-bottom:16px}
  .row{display:flex;gap:14px;flex-wrap:wrap;align-items:center}
  label{color:var(--mut);font-size:13px;margin-right:6px}
  select,button{font:inherit;color:var(--ink);background:#222633;border:1px solid #333a4a;border-radius:8px;padding:9px 12px}
  button{cursor:pointer} button.primary{background:var(--acc);border-color:var(--acc);color:#fff;font-weight:600}
  button:disabled{opacity:.5;cursor:not-allowed}
  video{width:100%;max-width:560px;border-radius:10px;background:#000;transform:scaleX(-1)}
  .pill{padding:3px 9px;border-radius:999px;font-size:12px;background:#222633;border:1px solid #333a4a}
  .split label{display:inline-flex;align-items:center;gap:5px;margin-right:12px;color:var(--ink)}
  #msg{min-height:20px;font-size:14px} .ok{color:var(--ok)} .err{color:var(--err)}
  table{width:100%;border-collapse:collapse;font-size:14px} th,td{text-align:left;padding:6px 8px;border-bottom:1px solid #262a36}
  th{color:var(--mut);font-weight:500} .mut{color:var(--mut)} pre{white-space:pre-wrap;color:var(--mut);font-size:12px;max-height:160px;overflow:auto}
</style></head>
<body><div class="wrap">
  <h1>Data Capture <span class="pill" id="domainPill">domain</span></h1>
  <p class="sub">Record a sign → saved as training data for that domain's model. Retraining is run offline by the admin.</p>

  <div class="card">
    <div class="row" style="margin-bottom:10px">
      <label>Domain</label><select id="domain"></select>
    </div>

    <div class="row" style="margin-bottom:12px">
      <div><label>Gloss</label><select id="gloss"></select></div>
    </div>

    <div class="split row" style="margin-bottom:10px"><label>Split</label>
      <label><input type="radio" name="split" value="train"> train</label>
      <label><input type="radio" name="split" value="val" checked> val (held-out)</label>
      <label><input type="radio" name="split" value="test"> test (held-out)</label>
    </div>

    <video id="cam" autoplay muted playsinline></video>
    <div class="row" style="margin-top:12px">
      <button id="rec" class="primary">● Record</button>
      <button id="stop" disabled>■ Stop</button>
      <button id="submit" disabled>Submit ▶</button>
      <span id="msg"></span>
    </div>
  </div>

  <div class="card">
    <div class="row" style="justify-content:space-between">
      <b>Captured so far</b>
      <button id="refresh">↻ Refresh</button>
    </div>
    <div id="inv" class="mut" style="margin-top:8px">loading…</div>
  </div>
</div>
<script>
const $=s=>document.querySelector(s); let stream,rec,chunks=[],blob=null;
// Domain is not hard-coded: default to a ?domain= param, else the first
// registry domain. Works for any SignBridge domain.
let DOMAIN = new URLSearchParams(location.search).get('domain') || '';
function msg(t,c){ const m=$('#msg'); m.textContent=t; m.className=c||''; }
async function loadDomains(){
  const r=await (await fetch('/api/capture/domains')).json();
  const doms=(r.success&&r.domains&&r.domains.length)?r.domains:['education'];
  if(!DOMAIN || !doms.includes(DOMAIN)) DOMAIN=doms[0];
  $('#domain').innerHTML=doms.map(d=>`<option${d===DOMAIN?' selected':''}>${d}</option>`).join('');
  $('#domainPill').textContent=DOMAIN+' domain';
}
async function loadGlosses(){
  const g=await (await fetch('/api/capture/glosses?domain='+encodeURIComponent(DOMAIN))).json();
  $('#gloss').innerHTML=(g.success?g.glosses:[]).map(x=>`<option>${x}</option>`).join('');
}
async function switchDomain(){ DOMAIN=$('#domain').value; $('#domainPill').textContent=DOMAIN+' domain';
  await loadGlosses(); loadInv(); }
async function init(){
  await loadDomains();
  $('#domain').onchange=switchDomain;
  await loadGlosses();
  try{ stream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280},height:{ideal:720},facingMode:'user'},audio:false}); $('#cam').srcObject=stream;
       const tr=stream.getVideoTracks()[0]; const s=tr.getSettings&&tr.getSettings(); if(s) msg(`camera ${s.width}x${s.height}`,'ok'); }
  catch(e){ msg('Camera error: '+e.message,'err'); }
  loadInv();
}
function pickMime(){ for(const t of ['video/webm;codecs=vp9','video/webm;codecs=vp8','video/webm','video/mp4']) if(MediaRecorder.isTypeSupported(t)) return t; return ''; }
$('#rec').onclick=()=>{ if(!stream) return; chunks=[]; blob=null; rec=new MediaRecorder(stream,{mimeType:pickMime()});
  rec.ondataavailable=e=>{if(e.data.size)chunks.push(e.data)}; rec.onstop=()=>{ blob=new Blob(chunks,{type:'video/webm'}); $('#submit').disabled=false; msg('Recorded '+(blob.size/1024|0)+' KB — review & submit','ok'); };
  rec.start(); $('#rec').disabled=true; $('#stop').disabled=false; $('#submit').disabled=true; msg('Recording…'); };
$('#stop').onclick=()=>{ if(rec&&rec.state!=='inactive')rec.stop(); $('#rec').disabled=false; $('#stop').disabled=true; };
$('#submit').onclick=async()=>{ if(!blob) return; const split=document.querySelector('input[name=split]:checked').value;
  $('#submit').disabled=true; msg('Uploading…');
  const fd=new FormData(); fd.append('split',split); fd.append('domain',DOMAIN); fd.append('gloss',$('#gloss').value); fd.append('video',blob,'capture.webm');
  const r=await (await fetch('/api/capture',{method:'POST',body:fd})).json();
  if(!r.success){ msg('Error: '+r.error,'err'); $('#submit').disabled=false; return; }
  blob=null; loadInv();
  msg(`Saved ${r.gloss} (${r.split}, ${r.frames} frames, ${r.backend}) ✓`,'ok'); };
async function loadInv(){ const r=await (await fetch('/api/captures?domain='+encodeURIComponent(DOMAIN))).json(); const b=r.by_gloss||{};
  const keys=Object.keys(b).sort(); if(!keys.length){ $('#inv').textContent='No captures yet.'; return; }
  let h='<table><tr><th>Gloss</th><th>captured</th></tr>';
  for(const k of keys){ h+=`<tr><td>${k}</td><td>${b[k]}</td></tr>`; }
  h+=`</table><div class="mut" style="margin-top:6px">Total clips: ${r.total} · store: ${r.backend||'?'}</div>`; $('#inv').innerHTML=h; }
$('#refresh').onclick=loadInv;
init();
</script></body></html>"""


# ---------------------------------------------------------------------------
# GET /model-compare  -- A/B a sign through two model versions (self-contained)
# ---------------------------------------------------------------------------
@capture_bp.route("/model-compare", methods=["GET"])
def model_compare_page():
    return _COMPARE_HTML


_COMPARE_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>SignBridge — Model Compare</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root{--bg:#0f1116;--card:#1a1d26;--ink:#e8eaf0;--mut:#9aa0ad;--acc:#4f8cff;--ok:#2ecc71;--err:#ff5c5c}
  *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,Segoe UI,Roboto,sans-serif}
  .wrap{max-width:920px;margin:0 auto;padding:20px}
  h1{font-size:20px;margin:0 0 4px} .sub{color:var(--mut);margin:0 0 18px}
  .card{background:var(--card);border:1px solid #262a36;border-radius:12px;padding:16px;margin-bottom:16px}
  .row{display:flex;gap:14px;flex-wrap:wrap;align-items:center}
  label{color:var(--mut);font-size:13px;margin-right:6px}
  select,button{font:inherit;color:var(--ink);background:#222633;border:1px solid #333a4a;border-radius:8px;padding:9px 12px}
  button{cursor:pointer} button.primary{background:var(--acc);border-color:var(--acc);color:#fff;font-weight:600}
  button:disabled{opacity:.5;cursor:not-allowed}
  video{width:100%;max-width:560px;border-radius:10px;background:#000;transform:scaleX(-1)}
  .pill{padding:3px 9px;border-radius:999px;font-size:12px;background:#222633;border:1px solid #333a4a}
  #msg{min-height:20px;font-size:14px} .ok{color:var(--ok)} .err{color:var(--err)} .mut{color:var(--mut)}
  .cols{display:grid;grid-template-columns:1fr 1fr;gap:14px}
  .res{background:#222633;border:1px solid #333a4a;border-radius:10px;padding:14px;text-align:center}
  .res .g{font-size:22px;font-weight:700;margin:6px 0} .res .c{font-size:14px;color:var(--mut)}
  .agree{margin-top:10px;font-weight:600}
</style></head>
<body><div class="wrap">
  <h1>Model Compare <span class="pill" id="domainPill">domain</span></h1>
  <p class="sub">Record one sign → run it through two model versions and compare predictions.</p>

  <div class="card">
    <div class="row" style="margin-bottom:10px"><label>Domain</label><select id="domain"></select></div>
    <div class="row" style="margin-bottom:12px">
      <div><label>Version A</label><select id="verA"></select></div>
      <div><label>Version B</label><select id="verB"></select></div>
    </div>
    <video id="cam" autoplay muted playsinline></video>
    <div class="row" style="margin-top:12px">
      <button id="rec" class="primary">● Record</button>
      <button id="stop" disabled>■ Stop</button>
      <button id="go" disabled>Compare ▶</button>
      <span id="msg"></span>
    </div>
  </div>

  <div class="card">
    <div class="cols">
      <div class="res"><div class="mut" id="labA">A</div><div class="g" id="gA">—</div><div class="c" id="cA"></div></div>
      <div class="res"><div class="mut" id="labB">B</div><div class="g" id="gB">—</div><div class="c" id="cB"></div></div>
    </div>
    <div class="agree mut" id="agree"></div>
  </div>
</div>
<script>
const $=s=>document.querySelector(s); let stream,rec,chunks=[],blob=null;
let DOMAIN = new URLSearchParams(location.search).get('domain') || '';
function msg(t,c){ const m=$('#msg'); m.textContent=t; m.className=c||''; }
function verOptions(info){
  // Always allow "(active)" (empty value -> server uses active model); then each explicit version.
  const active=info.active; const vs=info.versions||[];
  let opts='<option value="">(active'+(active?' = '+active:'')+')</option>';
  for(const v of vs) opts+=`<option value="${v}">${v}</option>`;
  return opts;
}
async function loadDomains(){
  const r=await (await fetch('/api/capture/domains')).json();
  const doms=(r.success&&r.domains&&r.domains.length)?r.domains:['education'];
  if(!DOMAIN || !doms.includes(DOMAIN)) DOMAIN=doms[0];
  $('#domain').innerHTML=doms.map(d=>`<option${d===DOMAIN?' selected':''}>${d}</option>`).join('');
  $('#domainPill').textContent=DOMAIN+' domain';
}
async function loadVersions(){
  const r=await (await fetch('/api/model-versions?domain='+encodeURIComponent(DOMAIN))).json();
  const info=r.success?r:{active:null,versions:[]};
  $('#verA').innerHTML=verOptions(info); $('#verB').innerHTML=verOptions(info);
  // default B to the newest explicit version if there is one, to contrast with active
  if(info.versions&&info.versions.length){ $('#verB').value=info.versions[info.versions.length-1]; }
  if(!info.versions||!info.versions.length){ msg('Only the active model exists — run a retrain to create a version to compare.','mut'); }
}
async function switchDomain(){ DOMAIN=$('#domain').value; $('#domainPill').textContent=DOMAIN+' domain'; await loadVersions(); }
async function init(){
  await loadDomains(); $('#domain').onchange=switchDomain; await loadVersions();
  try{ stream=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280},height:{ideal:720},facingMode:'user'},audio:false}); $('#cam').srcObject=stream; }
  catch(e){ msg('Camera error: '+e.message,'err'); }
}
function pickMime(){ for(const t of ['video/webm;codecs=vp9','video/webm;codecs=vp8','video/webm','video/mp4']) if(MediaRecorder.isTypeSupported(t)) return t; return ''; }
$('#rec').onclick=()=>{ if(!stream) return; chunks=[]; blob=null; rec=new MediaRecorder(stream,{mimeType:pickMime()});
  rec.ondataavailable=e=>{if(e.data.size)chunks.push(e.data)}; rec.onstop=()=>{ blob=new Blob(chunks,{type:'video/webm'}); $('#go').disabled=false; msg('Recorded '+(blob.size/1024|0)+' KB — compare','ok'); };
  rec.start(); $('#rec').disabled=true; $('#stop').disabled=false; $('#go').disabled=true; msg('Recording…'); };
$('#stop').onclick=()=>{ if(rec&&rec.state!=='inactive')rec.stop(); $('#rec').disabled=false; $('#stop').disabled=true; };
function predict(version){
  const fd=new FormData(); fd.append('domain',DOMAIN); if(version) fd.append('version',version); fd.append('video',blob,'clip.webm');
  return fetch('/api/process-sign',{method:'POST',body:fd}).then(r=>r.json());
}
function show(side,label,r){
  $('#lab'+side).textContent = 'Version '+(label||'(active)');
  if(!r||!r.success){ $('#g'+side).textContent='error'; $('#c'+side).textContent=(r&&r.error)||''; return null; }
  $('#g'+side).textContent = r.gloss||'?';
  $('#c'+side).textContent = ((r.confidence||0)*100).toFixed(0)+'% confidence';
  return r.gloss||'?';
}
$('#go').onclick=async()=>{ if(!blob) return; $('#go').disabled=true; msg('Running both versions…');
  const vA=$('#verA').value, vB=$('#verB').value;
  const [rA,rB]=await Promise.all([predict(vA),predict(vB)]);
  const gA=show('A',$('#verA').selectedOptions[0].text,rA);
  const gB=show('B',$('#verB').selectedOptions[0].text,rB);
  if(gA&&gB){ const same=gA===gB;
    $('#agree').textContent = same ? '✓ Both agree: '+gA : '✗ Disagree: A='+gA+'  B='+gB;
    $('#agree').className = 'agree '+(same?'ok':'err'); }
  msg('Done','ok'); $('#go').disabled=false; };
init();
</script></body></html>"""


# ---------------------------------------------------------------------------
# Registration (called from app.py)
# ---------------------------------------------------------------------------
def register_capture_routes(app, *, project_root, decode_video_bytes, extract_poses,
                            reject_outliers=None, drop_sparse_hands=None, smooth_missing=None):
    """Wire the capture blueprint into the main app, injecting the heavy helpers
    so this module never imports app.py back."""
    _DEPS.update({
        "project_root": Path(project_root),
        "decode_video_bytes": decode_video_bytes,
        "extract_poses": extract_poses,
        "reject_outliers": reject_outliers,
        "drop_sparse_hands": drop_sparse_hands,
        "smooth_missing": smooth_missing,
    })
    app.register_blueprint(capture_bp)
    print("[capture] isolated data-capture routes registered (/capture, /api/capture, /api/captures, /model-compare)")
