"""
Backend LLM Service (Port 4000)
Handles Gemini API calls for sentence construction from ASL glosses
"""

import os
import sys
import time
import threading
import json
from flask import Flask, request, jsonify
import requests
from pathlib import Path
import importlib

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent / "project-utilities"))

app = Flask(__name__)

# Configuration
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'AIzaSyBuQp46-lgxrC_-R6vDoX3CsyooCWtkAPk')
FRONTEND_URL = "http://localhost:3000"

# File-based communication (non-blocking)
GLOSS_FILE = "detected_glosses.txt"
SENTENCE_FILE = "translated_sentence.txt"
CAPTION_BUFFER_SIZE = 3  # Number of glosses before auto-translate
FILE_CHECK_INTERVAL = 0.5  # Seconds between file checks

# Context Management
CONTEXT_MODE = "full_history"
MAX_CONTEXT_SENTENCES = 2
ENABLE_INCREMENTAL_MODE = True
ENABLE_DYNAMIC_RECONSTRUCTION = True
RECONSTRUCTION_MODE = "sliding_window"
RECONSTRUCTION_WINDOW_SIZE = 15
MIN_RECONSTRUCTION_DELAY = 2.0  # Minimum seconds between translations (shows intermediate sentences)

# Session state (in production, use proper session management or database)
session_data = {
    'gemini_sentences': [],
    'running_caption': '',
    'all_raw_glosses': [],
    'processed_gloss_count': 0,  # Track how many glosses we've processed
    'last_translation_time': 0  # Track when last translation happened
}

# File watcher state
file_watcher_running = False
file_watcher_lock = threading.Lock()

def get_sentence_context(all_sentences):
    """Get sentence context based on configured mode"""
    if not all_sentences:
        return []

    if CONTEXT_MODE == "full_history":
        return all_sentences
    elif CONTEXT_MODE == "rolling_window":
        if MAX_CONTEXT_SENTENCES > 0:
            return all_sentences[-MAX_CONTEXT_SENTENCES:]
        else:
            return []
    else:
        print(f"WARNING: Unknown CONTEXT_MODE '{CONTEXT_MODE}', using no context")
        return []

def should_reconstruct(new_glosses):
    """Determine if we should reconstruct caption based on new glosses"""
    clarifying_words = {
        'who', 'what', 'which', 'that', 'this', 'these', 'those',
        'mother', 'father', 'man', 'woman', 'doctor', 'cousin',
        'now', 'before', 'finish', 'year',
        'who', 'what', 'where', 'when', 'why', 'how',
        'no', 'not', 'never', 'wrong',
        'all', 'many', 'few', 'some'
    }

    for g in new_glosses:
        if isinstance(g, dict):
            gloss = g.get('gloss', '').lower()
            top_k_list = g.get('top_k_predictions', g.get('top_k', []))
            for pred in top_k_list[:3]:
                pred_gloss = pred.get('gloss', '').lower()
                if pred_gloss in clarifying_words:
                    return True
        else:
            gloss = str(g).lower()
            if gloss in clarifying_words:
                return True

    return False

def get_reconstruction_glosses(all_glosses, mode):
    """Get glosses to use for reconstruction based on mode"""
    if mode == "sliding_window":
        if RECONSTRUCTION_WINDOW_SIZE > 0:
            return all_glosses[-RECONSTRUCTION_WINDOW_SIZE:]
        else:
            return all_glosses
    elif mode == "smart":
        return all_glosses
    else:
        print(f"WARNING: Unknown RECONSTRUCTION_MODE '{mode}', using all glosses")
        return all_glosses

def construct_sentence_with_gemini(glosses, sentence_context=None, running_caption=None,
                                  is_reconstruction=False, previous_caption=None):
    """Construct sentence using Gemini API"""

    # Process glosses
    processed_glosses = []
    has_top_k = False

    for g in glosses:
        if g == "<UNKNOWN>":
            continue
        elif isinstance(g, dict):
            processed_glosses.append(g)
            has_top_k = True
        else:
            processed_glosses.append(g)

    if not processed_glosses:
        return "No valid predictions available"

    if not GEMINI_API_KEY:
        print("WARNING: No Gemini API key provided")
        fallback_glosses = []
        for g in processed_glosses:
            if isinstance(g, dict):
                fallback_glosses.append(g.get('gloss', g.get('top_prediction', 'UNKNOWN')))
            else:
                fallback_glosses.append(g)
        return " ".join(fallback_glosses)

    # Check if Gemini is available
    try:
        module_name = 'google.generativeai'
        genai = importlib.import_module(module_name)
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.5-flash')
    except ImportError:
        print("ERROR: Gemini AI library not installed")
        fallback_glosses = []
        for g in processed_glosses:
            if isinstance(g, dict):
                fallback_glosses.append(g.get('gloss', g.get('top_prediction', 'UNKNOWN')))
            else:
                fallback_glosses.append(g)
        return " ".join(fallback_glosses)

    try:
        # Build prompt based on whether we have top-k predictions
        if has_top_k:
            gloss_details = []
            for i, g in enumerate(processed_glosses, 1):
                if isinstance(g, dict):
                    detail = f"Position {i}:\n"
                    top_k_list = g.get('top_k_predictions', g.get('top_k', []))
                    if top_k_list:
                        for j, pred in enumerate(top_k_list[:3], 1):
                            detail += f"  Option {j}: '{pred['gloss']}' (confidence: {pred['confidence']:.1%})\n"
                    else:
                        detail += f"  '{g.get('gloss', 'UNKNOWN')}'\n"
                    gloss_details.append(detail)
                else:
                    gloss_details.append(f"Position {i}:\n  '{g}'")

            context_section = ""
            if ENABLE_INCREMENTAL_MODE and running_caption:
                context_section = f"\nCurrent Running Caption:\n\"{running_caption}\"\n\n"
            elif sentence_context and len(sentence_context) > 0:
                context_section = "\nPrevious Context (for conversation flow):\n"
                for i, prev_sentence in enumerate(sentence_context, 1):
                    context_section += f"  {i}. \"{prev_sentence}\"\n"
                context_section += "\n"

            if ENABLE_INCREMENTAL_MODE:
                if is_reconstruction and previous_caption is not None:
                    prompt = f"""
You are an expert in American Sign Language (ASL) and English grammar, helping create live captions.

Previous Caption (for reference only - DO NOT preserve it):
"{previous_caption}"

ALL ASL Glosses Received (in chronological order):
{''.join(gloss_details)}

Instructions - RECONSTRUCTION MODE:
1. COMPLETELY REWRITE the caption from scratch using ALL the glosses above
2. DO NOT simply append to the previous caption
3. The previous caption is ONLY a reference - you should create an entirely new interpretation if it makes sense
4. For each position, you have 3 word options - evaluate all combinations to find the most coherent overall caption
5. New glosses may completely change how earlier glosses should be interpreted
6. Optimize for:
   - Natural pronoun usage (he, she, it, they, them) instead of repeating nouns
   - Combining multiple glosses into single cohesive sentences when they're related
   - Proper sentence boundaries (only separate when ideas are truly distinct)
   - Consistent verb tenses throughout
   - Overall grammatical correctness and natural flow
7. Think: "If I had all these {len(glosses)} words at once, what's the BEST way to express this idea?"
8. Use confidence scores as a guide, but prioritize overall coherence

Return the COMPLETE reconstructed caption, nothing else.
"""
                else:
                    prompt = f"""
You are an expert in American Sign Language (ASL) and English grammar, helping create live captions.
{context_section}
New ASL Gloss Predictions to add:
{''.join(gloss_details)}

Instructions:
1. EXTEND the current running caption by adding these new words
2. For each position, you have 3 word options - evaluate combinations for best interpretation
3. Use confidence scores as a guide, but prioritize semantic and grammatical coherence
4. Add appropriate articles, prepositions, and conjunctions as needed
5. Maintain proper verb tenses and subject-verb agreement
6. The new text should flow naturally from the existing caption
7. You may start a new sentence if it makes sense, but keep the previous caption intact

Return the COMPLETE caption (existing caption + new words added), nothing else.
"""
            else:
                prompt = f"""
You are an expert in American Sign Language (ASL) and English grammar.
Given the following ASL glosses in sequential order, construct a natural, grammatically correct English sentence that conveys the intended meaning.

For each position, you are given the TOP-3 prediction options with confidence scores. Your task is to try different combinations of these word choices and determine which combination produces the most coherent, natural-sounding English sentence.
{context_section}
ASL Gloss Predictions:
{''.join(gloss_details)}

Instructions:
1. Consider that ASL has different grammar rules than English
2. For each position, you have 3 word options - evaluate different combinations to find the best sentence
3. Use confidence scores as a guide, but prioritize semantic and grammatical coherence over confidence
4. The highest confidence option is not always the correct choice - context matters more
5. Try multiple word combinations mentally to find the most natural interpretation
6. Add appropriate articles (a, an, the) where needed
7. Add appropriate prepositions and conjunctions
8. Ensure proper verb tenses and subject-verb agreement
9. Make the sentence sound natural and professional
10. If previous context is provided, ensure your sentence flows naturally from the previous sentences

Return only the constructed English sentence, nothing else.
"""
        else:
            context_section = ""
            if ENABLE_INCREMENTAL_MODE and running_caption:
                context_section = f"\nCurrent Running Caption:\n\"{running_caption}\"\n\n"
            elif sentence_context and len(sentence_context) > 0:
                context_section = "\nPrevious Context (for conversation flow):\n"
                for i, prev_sentence in enumerate(sentence_context, 1):
                    context_section += f"  {i}. \"{prev_sentence}\"\n"
                context_section += "\n"

            if ENABLE_INCREMENTAL_MODE:
                if is_reconstruction and previous_caption is not None:
                    prompt = f"""
You are an expert in American Sign Language (ASL) and English grammar, helping create live captions.

Previous Caption (for reference only - DO NOT preserve it):
"{previous_caption}"

ALL ASL Glosses Received (in chronological order):
{', '.join(processed_glosses)}

Instructions - RECONSTRUCTION MODE:
1. COMPLETELY REWRITE the caption from scratch using ALL the glosses above
2. DO NOT simply append to the previous caption
3. Create the most natural, cohesive caption possible from all {len(processed_glosses)} words
4. Optimize for natural pronoun usage and sentence flow
5. Combine related ideas into single sentences

Return the COMPLETE reconstructed caption, nothing else.
"""
                else:
                    prompt = f"""
You are an expert in American Sign Language (ASL) and English grammar, helping create live captions.
{context_section}
New ASL Glosses to add: {', '.join(processed_glosses)}

Instructions:
1. EXTEND the current running caption by adding these new words
2. Add appropriate articles, prepositions, and conjunctions as needed
3. Maintain proper verb tenses and subject-verb agreement
4. The new text should flow naturally from the existing caption
5. You may start a new sentence if it makes sense, but keep the previous caption intact

Return the COMPLETE caption (existing caption + new words added), nothing else.
"""
            else:
                prompt = f"""
You are an expert in American Sign Language (ASL) and English grammar.
Given the following ASL glosses in order, construct a natural, grammatically correct English sentence that conveys the intended meaning.
{context_section}
ASL Glosses: {', '.join(processed_glosses)}

Instructions:
1. Consider that ASL has different grammar rules than English
2. Add appropriate articles (a, an, the) where needed
3. Add appropriate prepositions and conjunctions
4. Ensure proper verb tenses and subject-verb agreement
5. Make the sentence sound natural and professional
6. If some glosses seem unclear, use context to infer the most likely meaning
7. If previous context is provided, ensure your sentence flows naturally from the previous sentences

Return only the constructed English sentence, nothing else.
"""

        response = model.generate_content(prompt)
        sentence = response.text.strip()
        return sentence

    except Exception as e:
        print(f"ERROR: Gemini API call failed: {e}")
        fallback_glosses = []
        for g in processed_glosses:
            if isinstance(g, dict):
                fallback_glosses.append(g.get('gloss', g.get('top_prediction', 'UNKNOWN')))
            else:
                fallback_glosses.append(g)
        return " ".join(fallback_glosses)


def read_glosses_from_file():
    """Read all glosses from file"""
    glosses = []
    try:
        if os.path.exists(GLOSS_FILE):
            with open(GLOSS_FILE, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            gloss_data = json.loads(line)
                            glosses.append(gloss_data)
                        except:
                            pass
    except Exception as e:
        print(f"[BACKEND] Error reading gloss file: {e}")
    return glosses


def write_sentence_to_file(sentence):
    """Write translated sentence to file"""
    try:
        with open(SENTENCE_FILE, 'w') as f:
            f.write(sentence)
        print(f"[BACKEND] ✓ Wrote sentence to file")
    except Exception as e:
        print(f"[BACKEND] Error writing sentence file: {e}")


def file_watcher_thread():
    """Background thread that watches gloss file and auto-translates"""
    global file_watcher_running

    print("[BACKEND] File watcher started - monitoring glosses...")

    while file_watcher_running:
        try:
            # Read all glosses from file
            all_glosses = read_glosses_from_file()

            # Check if we have new glosses to process
            current_count = len(all_glosses)

            with file_watcher_lock:
                processed_count = session_data['processed_gloss_count']

            # Calculate how many new glosses we have
            new_gloss_count = current_count - processed_count

            # DEBUG: Show what we found
            if new_gloss_count > 0:
                print(f"[BACKEND] Found {new_gloss_count} new gloss(es) (total: {current_count}, processed: {processed_count})")

            if new_gloss_count >= CAPTION_BUFFER_SIZE:
                # Check cooldown period
                with file_watcher_lock:
                    time_since_last = time.time() - session_data['last_translation_time']

                if time_since_last < MIN_RECONSTRUCTION_DELAY:
                    # Still in cooldown - wait before translating
                    remaining = MIN_RECONSTRUCTION_DELAY - time_since_last
                    print(f"[BACKEND] ⏳ Cooldown active - waiting {remaining:.1f}s before next translation")
                    time.sleep(FILE_CHECK_INTERVAL)
                    continue

                print(f"\n[BACKEND] ⚡ Auto-translate triggered! ({new_gloss_count} new glosses)")

                # Get the new glosses
                new_glosses = all_glosses[processed_count:]

                print(f"[BACKEND] Processing {len(new_glosses)} new glosses:")
                for i, g in enumerate(new_glosses, 1):
                    print(f"  {i}. {g.get('gloss', 'UNKNOWN')} ({g.get('confidence', 0):.1%})")

                # Add to session history
                with file_watcher_lock:
                    session_data['all_raw_glosses'].extend(new_glosses)

                    # Get context
                    context = get_sentence_context(session_data['gemini_sentences'])

                    # Determine if reconstruction is needed
                    should_recon = False
                    if ENABLE_INCREMENTAL_MODE and ENABLE_DYNAMIC_RECONSTRUCTION and session_data['running_caption']:
                        if RECONSTRUCTION_MODE == "sliding_window":
                            should_recon = True
                        elif RECONSTRUCTION_MODE == "smart":
                            should_recon = should_reconstruct(new_glosses)

                    # Construct sentence
                    if should_recon:
                        recon_glosses = get_reconstruction_glosses(session_data['all_raw_glosses'], RECONSTRUCTION_MODE)
                        print(f"[BACKEND] Reconstructing from {len(recon_glosses)} glosses")
                        sentence = construct_sentence_with_gemini(
                            recon_glosses,
                            sentence_context=context,
                            running_caption=session_data['running_caption'],
                            is_reconstruction=True,
                            previous_caption=session_data['running_caption']
                        )
                    else:
                        print(f"[BACKEND] Extending caption (incremental mode)")
                        sentence = construct_sentence_with_gemini(
                            new_glosses,
                            sentence_context=context,
                            running_caption=session_data['running_caption'] if session_data['running_caption'] else None
                        )

                    # Update session state
                    if ENABLE_INCREMENTAL_MODE:
                        session_data['running_caption'] = sentence

                    session_data['gemini_sentences'].append(sentence)
                    session_data['processed_gloss_count'] = current_count
                    session_data['last_translation_time'] = time.time()  # Record translation time

                print(f"[BACKEND] ✓ Constructed: '{sentence}'")

                # Write sentence to file (frontend will read it)
                write_sentence_to_file(sentence)

            # Sleep before next check
            time.sleep(FILE_CHECK_INTERVAL)

        except Exception as e:
            print(f"[BACKEND] File watcher error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(FILE_CHECK_INTERVAL)

    print("[BACKEND] File watcher stopped")


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'backend_llm_service'}), 200

@app.route('/translate', methods=['POST'])
def translate_glosses():
    """
    Translate glosses to English sentence

    Expected JSON payload:
    {
        "glosses": [...],  # List of gloss predictions
        "session_id": "optional-session-id"
    }
    """
    try:
        data = request.get_json()
        glosses = data.get('glosses', [])
        session_id = data.get('session_id', 'default')

        if not glosses:
            return jsonify({'error': 'No glosses provided'}), 400

        print(f"\n[BACKEND] Received {len(glosses)} glosses for translation")

        # Add to session history
        session_data['all_raw_glosses'].extend(glosses)

        # Get context
        context = get_sentence_context(session_data['gemini_sentences'])

        # Determine if reconstruction is needed
        should_recon = False
        if ENABLE_INCREMENTAL_MODE and ENABLE_DYNAMIC_RECONSTRUCTION and session_data['running_caption']:
            if RECONSTRUCTION_MODE == "sliding_window":
                should_recon = True
                print(f"[BACKEND] Reconstruction mode: sliding_window")
            elif RECONSTRUCTION_MODE == "smart":
                should_recon = should_reconstruct(glosses)
                print(f"[BACKEND] Reconstruction mode: smart - Reconstruct: {should_recon}")

        # Construct sentence
        if should_recon:
            recon_glosses = get_reconstruction_glosses(session_data['all_raw_glosses'], RECONSTRUCTION_MODE)
            print(f"[BACKEND] Reconstructing from {len(recon_glosses)} glosses")
            sentence = construct_sentence_with_gemini(
                recon_glosses,
                sentence_context=context,
                running_caption=session_data['running_caption'],
                is_reconstruction=True,
                previous_caption=session_data['running_caption']
            )
        else:
            print(f"[BACKEND] Extending caption (incremental mode)")
            sentence = construct_sentence_with_gemini(
                glosses,
                sentence_context=context,
                running_caption=session_data['running_caption'] if session_data['running_caption'] else None
            )

        # Update session state
        if ENABLE_INCREMENTAL_MODE:
            session_data['running_caption'] = sentence

        session_data['gemini_sentences'].append(sentence)

        print(f"[BACKEND] Constructed sentence: '{sentence}'")

        # Notify frontend with the sentence
        try:
            response = requests.post(
                f"{FRONTEND_URL}/sentence_update",
                json={'sentence': sentence, 'session_id': session_id},
                timeout=2
            )
            print(f"[BACKEND] Notified frontend: {response.status_code}")
        except Exception as e:
            print(f"[BACKEND] Failed to notify frontend: {e}")

        return jsonify({
            'success': True,
            'sentence': sentence,
            'running_caption': session_data['running_caption']
        }), 200

    except Exception as e:
        print(f"[BACKEND] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/reset_session', methods=['POST'])
def reset_session():
    """Reset session state"""
    with file_watcher_lock:
        session_data['gemini_sentences'] = []
        session_data['running_caption'] = ''
        session_data['all_raw_glosses'] = []
        session_data['processed_gloss_count'] = 0
        session_data['last_translation_time'] = 0

    # Clear files
    try:
        with open(GLOSS_FILE, 'w') as f:
            f.write("")
        with open(SENTENCE_FILE, 'w') as f:
            f.write("")
        print("[BACKEND] Session and files reset")
    except Exception as e:
        print(f"[BACKEND] Error resetting files: {e}")

    return jsonify({'success': True, 'message': 'Session reset'}), 200

if __name__ == '__main__':
    print("="*70)
    print("BACKEND LLM SERVICE (Port 4000)")
    print("="*70)
    print(f"Gemini API Key: {'Configured' if GEMINI_API_KEY else 'NOT CONFIGURED'}")
    print(f"Gloss File: {GLOSS_FILE}")
    print(f"Sentence File: {SENTENCE_FILE}")
    print(f"Auto-translate after: {CAPTION_BUFFER_SIZE} glosses")
    print(f"File check interval: {FILE_CHECK_INTERVAL}s")
    print(f"Min delay between translations: {MIN_RECONSTRUCTION_DELAY}s")
    print("="*70)
    print()

    # Clear files on startup for fresh session
    try:
        if os.path.exists(GLOSS_FILE):
            os.remove(GLOSS_FILE)
        if os.path.exists(SENTENCE_FILE):
            os.remove(SENTENCE_FILE)
        print("[BACKEND] Cleared old session files")

        # Reset session state to match cleared files
        session_data['gemini_sentences'] = []
        session_data['running_caption'] = ''
        session_data['all_raw_glosses'] = []
        session_data['processed_gloss_count'] = 0
        session_data['last_translation_time'] = 0
        print("[BACKEND] Reset session state")
    except Exception as e:
        print(f"[BACKEND] Warning: Could not clear old files: {e}")

    # Start file watcher thread
    def start_services():
        global file_watcher_running
        file_watcher_running = True
        watcher = threading.Thread(target=file_watcher_thread, daemon=True)
        watcher.start()
        print("[BACKEND] File watcher thread started")
        print()

    start_services()

    # Start Flask app
    try:
        app.run(host='0.0.0.0', port=4000, debug=False, threaded=True)
    finally:
        file_watcher_running = False
        print("\n[BACKEND] Shutting down...")
