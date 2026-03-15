"""
SignBridge Mobile PWA
A mobile-optimized Progressive Web App shell that reverse-proxies
the show-and-tell Flask server. Serves its own landing page + PWA assets.

Usage:
    python app.py
    # Then on iPad: open http://<laptop-ip>:8080
    # Add to Home Screen for native app experience

Requires the show-and-tell server running on port 5000.
"""

import os
import socket
import requests
from flask import Flask, render_template, request, Response, send_from_directory, redirect

app = Flask(__name__)

# The show-and-tell server we proxy to
BACKEND_URL = os.environ.get('BACKEND_URL', 'http://127.0.0.1:5000')

# ─── PWA Assets (served from our own static dir) ───

@app.route('/manifest.json')
def manifest():
    return send_from_directory(app.static_folder, 'manifest.json')


@app.route('/sw.js')
def service_worker():
    resp = send_from_directory(app.static_folder, 'sw.js')
    resp.headers['Service-Worker-Allowed'] = '/'
    return resp


@app.route('/mobile-css/<path:filename>')
def mobile_css(filename):
    return send_from_directory(os.path.join(app.static_folder, 'css'), filename)


@app.route('/mobile-icons/<path:filename>')
def mobile_icons(filename):
    return send_from_directory(os.path.join(app.static_folder, 'icons'), filename)


# ─── Landing Page vs Proxy for '/' ───
# First visit (no cookie) → show mobile landing page
# After picking a mode (cookie set) → proxy to backend's '/'

@app.route('/')
def root():
    if request.cookies.get('signbridge_mode'):
        # User already picked a mode, proxy to the show-and-tell backend
        return proxy('')
    return render_template('landing.html')


@app.route('/home')
def go_home():
    """Explicit route to return to the mobile landing page."""
    resp = redirect('/')
    resp.delete_cookie('signbridge_mode')
    return resp


# ─── Reverse Proxy ───

EXCLUDED_HEADERS = {'host', 'content-length', 'transfer-encoding', 'connection'}


def proxy(path):
    """Forward request to the show-and-tell backend."""
    url = f"{BACKEND_URL}/{path}"
    if request.query_string:
        url += f"?{request.query_string.decode()}"

    headers = {k: v for k, v in request.headers if k.lower() not in EXCLUDED_HEADERS}

    try:
        resp = requests.request(
            method=request.method,
            url=url,
            headers=headers,
            data=request.get_data(),
            cookies=request.cookies,
            allow_redirects=False,
            timeout=120,
            stream=True,
        )
    except requests.ConnectionError:
        return Response(
            "Cannot reach the SignBridge server. Make sure the show-and-tell app "
            f"is running on {BACKEND_URL}",
            status=502,
            content_type='text/plain'
        )

    excluded_resp = {'content-encoding', 'content-length', 'transfer-encoding', 'connection'}
    resp_headers = [(k, v) for k, v in resp.raw.headers.items() if k.lower() not in excluded_resp]

    return Response(
        resp.content,
        status=resp.status_code,
        headers=resp_headers,
        content_type=resp.headers.get('content-type'),
    )


# Catch-all proxy routes
@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH'])
def catch_all(path):
    return proxy(path)


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return '127.0.0.1'


if __name__ == '__main__':
    local_ip = get_local_ip()
    port = int(os.environ.get('PORT', 8080))

    print(f"\n{'='*60}")
    print(f"  SignBridge Mobile PWA")
    print(f"{'='*60}")
    print(f"  Local:   http://127.0.0.1:{port}")
    print(f"  Network: http://{local_ip}:{port}")
    print(f"  Backend: {BACKEND_URL}")
    print(f"{'='*60}")
    print(f"\n  On iPad: open http://{local_ip}:{port}")
    print(f"  Then tap Share > Add to Home Screen\n")

    app.run(host='0.0.0.0', port=port, debug=True)
