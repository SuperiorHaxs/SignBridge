/**
 * motion_detector.js
 *
 * Common motion detection module for ASL sign segmentation.
 * Used by both Live Mode and Closed Captions Mode.
 *
 * Provides consistent motion detection behavior across all modes.
 */

class MotionDetector {
    constructor(config = {}) {
        // Default configuration
        this.config = {
            cooldown_ms: config.cooldown_ms || 1000,
            min_sign_ms: config.min_sign_ms || 500,
            max_sign_ms: config.max_sign_ms || 5000,
            motion_threshold: config.motion_threshold || 30,
            motion_area_threshold: config.motion_area_threshold || 0.02,
            warmup_frames: config.warmup_frames || 90,
            canvas_width: config.canvas_width || 160,
            canvas_height: config.canvas_height || 120
        };

        // State
        this.isSigning = false;
        this.isProcessing = false;
        this.signStartTime = null;
        this.lastMotionTime = null;
        this.frameCount = 0;
        this.previousFrame = null;

        // Canvas for motion detection
        this.canvas = document.createElement('canvas');
        this.canvas.width = this.config.canvas_width;
        this.canvas.height = this.config.canvas_height;
        this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });

        // Callbacks
        this.onSignStart = null;
        this.onSignEnd = null;
        this.onSignTooShort = null;
        this.onWarmupProgress = null;
        this.onWarmupComplete = null;
        this.onMotionScore = null;
    }

    /**
     * Update configuration
     */
    updateConfig(config) {
        Object.assign(this.config, config);
    }

    /**
     * Reset state for new session
     */
    reset() {
        this.isSigning = false;
        this.isProcessing = false;
        this.signStartTime = null;
        this.lastMotionTime = null;
        this.frameCount = 0;
        this.previousFrame = null;
    }

    /**
     * Set processing state (called externally when processing a sign)
     */
    setProcessing(isProcessing) {
        this.isProcessing = isProcessing;
    }

    /**
     * Process a video frame for motion detection
     * @param {HTMLVideoElement} video - The video element to analyze
     * @returns {Object} - { isWarmup, motionScore, motionDetected }
     */
    processFrame(video) {
        if (!video || video.readyState < 2) {
            return { isWarmup: true, motionScore: 0, motionDetected: false };
        }

        const now = Date.now();
        this.frameCount++;

        // Warmup period - let camera stabilize
        if (this.frameCount <= this.config.warmup_frames) {
            const secondsLeft = Math.ceil((this.config.warmup_frames - this.frameCount) / 30);
            if (this.onWarmupProgress) {
                this.onWarmupProgress(secondsLeft, this.frameCount, this.config.warmup_frames);
            }
            return { isWarmup: true, motionScore: 0, motionDetected: false };
        }

        // Warmup just completed
        if (this.frameCount === this.config.warmup_frames + 1) {
            if (this.onWarmupComplete) {
                this.onWarmupComplete();
            }
        }

        // Draw current frame to detection canvas
        this.ctx.drawImage(video, 0, 0, this.canvas.width, this.canvas.height);
        const currentFrame = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);

        // Calculate motion score
        let motionScore = 0;
        if (this.previousFrame) {
            motionScore = this._calculateMotionScore(this.previousFrame.data, currentFrame.data);
        }
        this.previousFrame = currentFrame;

        // Report motion score
        if (this.onMotionScore) {
            this.onMotionScore(motionScore);
        }

        // Detect motion using area threshold
        // motionScore is a ratio (0-1) of pixels that changed
        const motionDetected = motionScore > this.config.motion_area_threshold;

        // State machine for sign detection
        this._updateSignState(motionDetected, now);

        return { isWarmup: false, motionScore, motionDetected };
    }

    /**
     * Calculate motion score between two frames
     * Returns ratio (0-1) of pixels that changed significantly
     */
    _calculateMotionScore(prevData, currData) {
        let changedPixels = 0;
        const length = prevData.length;
        const pixelThreshold = this.config.motion_threshold;

        // Sample every pixel for accuracy (i += 4 for RGBA)
        for (let i = 0; i < length; i += 4) {
            const rDiff = Math.abs(prevData[i] - currData[i]);
            const gDiff = Math.abs(prevData[i + 1] - currData[i + 1]);
            const bDiff = Math.abs(prevData[i + 2] - currData[i + 2]);

            // Use sum of differences (more sensitive, matches Live Mode behavior)
            if (rDiff + gDiff + bDiff > pixelThreshold) {
                changedPixels++;
            }
        }

        // Return ratio of changed pixels
        const totalPixels = length / 4;
        return changedPixels / totalPixels;
    }

    /**
     * Update sign detection state based on motion
     */
    _updateSignState(motionDetected, now) {
        if (motionDetected) {
            this.lastMotionTime = now;

            if (!this.isSigning && !this.isProcessing) {
                // Start signing
                this.isSigning = true;
                this.signStartTime = now;
                if (this.onSignStart) {
                    this.onSignStart();
                }
            }
        } else if (this.isSigning) {
            // Check if cooldown period has elapsed
            const timeSinceMotion = now - this.lastMotionTime;

            if (timeSinceMotion >= this.config.cooldown_ms) {
                const signDuration = now - this.signStartTime;

                if (signDuration >= this.config.min_sign_ms) {
                    // Valid sign - trigger end
                    this.isSigning = false;
                    if (this.onSignEnd) {
                        this.onSignEnd(signDuration);
                    }
                } else {
                    // Sign too short - cancel
                    this.isSigning = false;
                    if (this.onSignTooShort) {
                        this.onSignTooShort(signDuration);
                    }
                }
            }
        }

        // Check max duration
        if (this.isSigning) {
            const signDuration = now - this.signStartTime;
            if (signDuration >= this.config.max_sign_ms) {
                // Max duration reached - force end
                this.isSigning = false;
                if (this.onSignEnd) {
                    this.onSignEnd(signDuration);
                }
            }
        }
    }
}

// Export for use in other modules
window.MotionDetector = MotionDetector;
