#!/usr/bin/env python3
"""
predict_sentence.py
Predicts glosses for segmented pose files using OpenHands model

Reads pose files from a segments folder and uses the OpenHands model
to predict the gloss for each segment, then reconstructs the sentence.
"""

import os
import argparse
import glob
from pathlib import Path
import sys

# Add the current directory to Python path to import openhands_modernized
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from openhands_modernized import predict_pose_file, load_model_from_checkpoint
    print("SUCCESS: OpenHands model imported successfully")
except ImportError as e:
    print("ERROR: Failed to import OpenHands model!")
    print(f"Error: {e}")
    print("Make sure openhands_modernized.py is in the same directory")
    sys.exit(1)


class SentencePredictor:
    """Predicts sentence from segmented pose files"""

    def __init__(self, checkpoint_path=None):
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.tokenizer = None

        if checkpoint_path:
            print(f"Loading model from checkpoint: {checkpoint_path}")
            try:
                self.model, self.tokenizer = load_model_from_checkpoint(checkpoint_path)
                print("SUCCESS: Checkpoint model loaded successfully")
            except Exception as e:
                print(f"ERROR: Failed to load checkpoint: {e}")
                print("Falling back to default model")

    def find_pose_files(self, segments_dir):
        """Find all pose files in the segments directory"""

        print(f"Looking for pose files in: {segments_dir}")

        if not os.path.exists(segments_dir):
            print(f"ERROR: Segments directory not found: {segments_dir}")
            return []

        # Look for .pose files
        pose_files = glob.glob(os.path.join(segments_dir, "*.pose"))

        if not pose_files:
            print("ERROR: No .pose files found in directory")
            return []

        # Sort files to maintain order (sign_001, sign_002, etc.)
        pose_files.sort()

        print(f"Found {len(pose_files)} pose files:")
        for i, file_path in enumerate(pose_files):
            file_name = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)
            print(f"  {i+1:2d}. {file_name} ({file_size:,} bytes)")

        return pose_files

    def convert_pose_to_pickle(self, pose_file_path):
        """Convert .pose file to pickle format like the training data"""
        try:
            from pose_format import Pose
            import pickle
            import tempfile
            import numpy as np

            print(f"DEBUG: Converting {os.path.basename(pose_file_path)}")

            # Load pose file
            with open(pose_file_path, "rb") as f:
                buffer = f.read()
                pose = Pose.read(buffer)

            # Extract pose data in the same format as training
            pose_data = pose.body.data
            print(f"DEBUG: Original pose data shape: {pose_data.shape}")

            if len(pose_data.shape) == 4:
                # (frames, people, keypoints, dimensions) -> take first person
                pose_sequence = pose_data[:, 0, :, :]
            else:
                pose_sequence = pose_data

            print(f"DEBUG: Extracted pose sequence shape: {pose_sequence.shape}")
            print(f"DEBUG: Pose data sample (first frame, first 3 keypoints):")
            print(pose_sequence[0, :3, :])

            # Create temporary pickle file
            temp_pickle = tempfile.NamedTemporaryFile(suffix='.pkl', delete=False)

            # Save in the same format as training data
            pickle_data = {
                'pose': pose_sequence,
                'gloss': 'UNKNOWN'  # Placeholder since we're predicting
            }

            with open(temp_pickle.name, 'wb') as f:
                pickle.dump(pickle_data, f)

            print(f"DEBUG: Saved to temporary pickle: {temp_pickle.name}")
            return temp_pickle.name

        except Exception as e:
            print(f"ERROR: Error converting {os.path.basename(pose_file_path)} to pickle: {e}")
            return None

    def predict_single_pose(self, pose_file_path):
        """Predict gloss for a single pose file"""

        try:
            print(f"Predicting: {os.path.basename(pose_file_path)}")

            # Convert .pose to .pkl first
            temp_pickle_path = self.convert_pose_to_pickle(pose_file_path)
            if not temp_pickle_path:
                return None

            # Call the OpenHands prediction function with pickle file
            if self.model and self.tokenizer:
                # Use checkpoint model if available
                result = predict_pose_file(temp_pickle_path, model=self.model, tokenizer=self.tokenizer)
            else:
                # Use default model
                result = predict_pose_file(temp_pickle_path)

            # Clean up temporary file
            if os.path.exists(temp_pickle_path):
                os.unlink(temp_pickle_path)

            if result and len(result) == 2:
                prediction, confidence = result
                print(f"SUCCESS: Prediction: '{prediction}' (confidence: {confidence:.2%})")
                return prediction
            elif result:
                # Handle case where only prediction is returned
                prediction = result
                print(f"SUCCESS: Prediction: '{prediction}'")
                return prediction
            else:
                print(f"ERROR: No prediction returned")
                return None

        except Exception as e:
            print(f"ERROR: Error predicting {os.path.basename(pose_file_path)}: {e}")
            return None

    def predict_sentence(self, segments_dir, output_file=None):
        """Predict glosses for all segments and reconstruct sentence"""

        print("Sign Language Sentence Prediction")
        print("="*50)

        # Find all pose files
        pose_files = self.find_pose_files(segments_dir)
        if not pose_files:
            return

        # Predict each segment
        predictions = []
        successful_predictions = 0

        print(f"\nPredicting {len(pose_files)} segments...")
        print("-"*50)

        for i, pose_file in enumerate(pose_files):
            file_name = os.path.basename(pose_file)
            print(f"\nSegment {i+1}/{len(pose_files)}: {file_name}")

            prediction = self.predict_single_pose(pose_file)

            if prediction:
                predictions.append(prediction)
                successful_predictions += 1
            else:
                predictions.append("<UNKNOWN>")

        # Results summary
        print("\n" + "="*50)
        print("RESULTS: PREDICTION RESULTS:")
        print(f"   Segments processed: {len(pose_files)}")
        print(f"   Successful predictions: {successful_predictions}")
        print(f"   Failed predictions: {len(pose_files) - successful_predictions}")

        # Show individual results
        print(f"\nLIST: Individual Predictions:")
        for i, (pose_file, prediction) in enumerate(zip(pose_files, predictions)):
            file_name = os.path.basename(pose_file)
            status = "SUCCESS:" if prediction != "<UNKNOWN>" else "ERROR:"
            print(f"  {i+1:2d}. {file_name:<30} → {status} '{prediction}'")

        # Reconstruct sentence
        print(f"\nSUCCESS: PREDICTED SENTENCE:")
        sentence = " ".join([pred for pred in predictions if pred != "<UNKNOWN>"])
        if sentence:
            print(f"   '{sentence}'")
        else:
            print("   (No successful predictions)")

        # Save to file if requested
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    f.write("Sign Language Sentence Prediction Results\n")
                    f.write("="*50 + "\n\n")

                    f.write("Individual Predictions:\n")
                    for i, (pose_file, prediction) in enumerate(zip(pose_files, predictions)):
                        file_name = os.path.basename(pose_file)
                        f.write(f"{i+1:2d}. {file_name} → '{prediction}'\n")

                    f.write(f"\nPredicted Sentence:\n")
                    f.write(f"'{sentence}'\n")

                    f.write(f"\nStatistics:\n")
                    f.write(f"Segments processed: {len(pose_files)}\n")
                    f.write(f"Successful predictions: {successful_predictions}\n")
                    f.write(f"Failed predictions: {len(pose_files) - successful_predictions}\n")

                print(f"\nSAVED: Results saved to: {output_file}")
            except Exception as e:
                print(f"ERROR: Error saving results: {e}")

        return predictions, sentence

    def predict_with_confidence(self, segments_dir, min_confidence=0.5):
        """Predict with confidence scores if available"""

        print("TARGET: Sign Language Sentence Prediction (with confidence)")
        print("="*60)

        pose_files = self.find_pose_files(segments_dir)
        if not pose_files:
            return

        predictions_with_confidence = []

        print(f"\nPREDICTING: Predicting {len(pose_files)} segments with confidence...")
        print("-"*60)

        for i, pose_file in enumerate(pose_files):
            file_name = os.path.basename(pose_file)
            print(f"\nSegment {i+1}/{len(pose_files)}: {file_name}")

            try:
                # Convert .pose to .pkl first
                temp_pickle_path = self.convert_pose_to_pickle(pose_file)
                if not temp_pickle_path:
                    predictions_with_confidence.append(("<CONVERSION_ERROR>", 0.0))
                    continue

                # Try to get prediction with confidence if the function supports it
                if self.model and self.tokenizer:
                    # Use checkpoint model if available
                    result = predict_pose_file(temp_pickle_path, model=self.model, tokenizer=self.tokenizer)
                else:
                    # Use default model
                    result = predict_pose_file(temp_pickle_path)

                # Clean up temporary file
                if os.path.exists(temp_pickle_path):
                    os.unlink(temp_pickle_path)

                if isinstance(result, tuple) and len(result) == 2:
                    # Function returned (prediction, confidence)
                    prediction, confidence = result
                    print(f"SUCCESS: Prediction: '{prediction}' (confidence: {confidence:.3f})")

                    if confidence >= min_confidence:
                        predictions_with_confidence.append((prediction, confidence))
                    else:
                        print(f"WARNING:  Low confidence ({confidence:.3f} < {min_confidence}), skipping")
                        predictions_with_confidence.append(("<LOW_CONF>", confidence))

                elif isinstance(result, str):
                    # Function returned just prediction
                    prediction = result
                    print(f"SUCCESS: Prediction: '{prediction}' (no confidence info)")
                    predictions_with_confidence.append((prediction, 1.0))
                else:
                    print(f"ERROR: Unexpected result format: {result}")
                    predictions_with_confidence.append(("<UNKNOWN>", 0.0))

            except Exception as e:
                print(f"ERROR: Error: {e}")
                predictions_with_confidence.append(("<ERROR>", 0.0))

        # Results with confidence
        print("\n" + "="*60)
        print("RESULTS: CONFIDENCE-BASED RESULTS:")

        high_conf_predictions = [pred for pred, conf in predictions_with_confidence
                               if conf >= min_confidence and pred not in ["<UNKNOWN>", "<ERROR>", "<LOW_CONF>"]]

        sentence = " ".join(high_conf_predictions)

        print(f"   High confidence predictions: {len(high_conf_predictions)}")
        print(f"   Final sentence: '{sentence}'")

        return predictions_with_confidence, sentence


def main():
    """Main function"""

    parser = argparse.ArgumentParser(description="Predict sentence from segmented pose files")
    parser.add_argument("segments_dir", help="Directory containing segmented .pose files")
    parser.add_argument("-o", "--output", help="Output file to save results")
    parser.add_argument("--checkpoint", help="Path to checkpoint directory (e.g., './checkpoints/checkpoint_epoch_5')")
    parser.add_argument("--confidence", action="store_true",
                        help="Use confidence-based prediction (if supported)")
    parser.add_argument("--min-confidence", type=float, default=0.5,
                        help="Minimum confidence threshold (default: 0.5)")

    args = parser.parse_args()

    # Validate segments directory
    if not os.path.exists(args.segments_dir):
        print(f"ERROR: Segments directory not found: {args.segments_dir}")
        return

    # Create predictor (with checkpoint if provided)
    predictor = SentencePredictor(checkpoint_path=args.checkpoint)

    # Run prediction
    try:
        if args.confidence:
            predictions, sentence = predictor.predict_with_confidence(
                args.segments_dir,
                args.min_confidence
            )
        else:
            predictions, sentence = predictor.predict_sentence(
                args.segments_dir,
                args.output
            )

        if sentence:
            print(f"\nSUCCESS: SUCCESS: Predicted sentence: '{sentence}'")
        else:
            print(f"\nFAILED: No successful predictions made")

    except KeyboardInterrupt:
        print("\n\nSTOPPED:  Prediction interrupted by user")
    except Exception as e:
        print(f"\nERROR: Unexpected error: {e}")


if __name__ == "__main__":
    main()