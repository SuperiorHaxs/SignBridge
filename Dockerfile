FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (required by HF Spaces)
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# Copy the full project (app imports from multiple directories)
COPY --chown=user . .

# Install PyTorch CPU-only first (separate index).
# typing-extensions is installed from PyPI FIRST: the PyTorch CPU mirror re-hosts
# it with a metadata name ('typing_extensions') that newer/stricter pip rejects,
# which forces an sdist build that then can't find flit_core on the torch-only
# index. Pre-satisfying it from PyPI means the torch install never touches that
# broken wheel, while torch itself stays CPU-only from the PyTorch index.
RUN pip install --no-cache-dir "typing-extensions>=4.10.0"
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies
RUN pip install --no-cache-dir \
    flask \
    "flask-sock>=0.7.0" \
    numpy \
    opencv-python-headless \
    "mediapipe<0.10.15" \
    pose-format \
    vidgear \
    google-generativeai \
    transformers \
    sacrebleu \
    bert-score \
    scipy \
    scikit-learn \
    openai \
    requests \
    python-dotenv

USER user

# HF Spaces expects port 7860
ENV PORT=7860
EXPOSE 7860

# Start the SignBridge app in live mode (WS streaming + WiFi camera
# + wake-word + Settings tab -- the current production flow). Show-
# and-tell remains in the repo but isn't the deployed entry point.
CMD ["python", "applications/signbridge-app/app.py", "--mode", "live", "--port", "7860"]
