FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (required by HF Spaces)
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# Copy the full project (app imports from multiple directories)
COPY --chown=user . .

# Install Python dependencies
RUN pip install --no-cache-dir \
    flask>=2.0.0 \
    torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
    numpy \
    opencv-python-headless \
    mediapipe \
    pose-format \
    vidgear \
    google-generativeai \
    transformers \
    sacrebleu \
    bert-score \
    scipy \
    scikit-learn

USER user

# HF Spaces expects port 7860
ENV PORT=7860
EXPOSE 7860

# Start the Flask app
CMD ["python", "applications/show-and-tell/app.py"]
