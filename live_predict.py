"""
live_predict.py

Full end-to-end demo: opens the camera, detects hand landmarks, runs them
through the trained classifier, shows the predicted sign + confidence on
screen, speaks the predicted letter aloud, and builds up a full word by
accumulating stable letter detections. A space is automatically inserted
into the word whenever the hand leaves the frame for a sustained period.

Requires: model/sign_language_model.pkl and model/labels.json
          (created by train_model.py)

Controls:
    q -> quit
    c -> clear the current word

Word-building behavior:
    - A letter is only added to the word once it has been stable (same
      letter, confidence above threshold) for STABLE_FRAMES_REQUIRED
      consecutive frames, and only once per "hold" (won't repeat the same
      letter over and over while you keep your hand still).
    - A space is added once the hand has been missing from the frame for
      NO_HAND_FRAMES_FOR_SPACE consecutive frames (and only once per
      "hand away" period, so it won't add multiple spaces).
"""

import json
import threading
import cv2
import joblib
import numpy as np
import pyttsx3
from hand_detector import HandDetector

CONFIDENCE_THRESHOLD = 60.0       # % - below this, prediction is ignored
STABLE_FRAMES_REQUIRED = 10       # consecutive matching frames before accepting a letter
REPEAT_FRAMES_REQUIRED = 45      # extra consecutive frames needed to repeat the same letter
NO_HAND_FRAMES_FOR_SPACE = 20     # consecutive no-hand frames before inserting a space


def load_model(model_path="model/sign_language_model.pkl", labels_path="model/labels.json"):
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"ERROR: Model file not found at '{model_path}'.")
        print("Run train_model.py first to train and save a model.")
        return None, None

    with open(labels_path, "r") as f:
        labels = json.load(f)

    return model, labels


def main():
    model, labels = load_model()
    if model is None:
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not access the camera.")
        return

    detector = HandDetector()

    def speak(text):
        """Runs TTS in a background thread so the camera feed doesn't freeze."""
        def _speak():
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.say(text)
            engine.runAndWait()
            engine.stop()

        threading.Thread(target=_speak, daemon=True).start()

    # --- State for letter debouncing ---
    stable_prediction = None      # letter currently being "held"
    stable_count = 0              # how many consecutive frames it's been stable
    next_add_threshold = STABLE_FRAMES_REQUIRED  # stable_count value at which to add a letter

    # --- State for space insertion ---
    no_hand_count = 0
    space_pending = False         # True once hand is gone; prevents multiple spaces

    current_word = ""

    print("Model loaded. Press 'q' to quit, 'c' to clear the word.")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("WARNING: Failed to read frame. Stopping.")
                break

            frame = cv2.flip(frame, 1)
            frame, results = detector.find_hands(frame)
            landmarks = detector.extract_landmarks(results)

            if landmarks is not None:
                # Hand is visible - reset the "no hand" space tracking
                no_hand_count = 0
                space_pending = False

                prediction = model.predict([landmarks])[0]
                probabilities = model.predict_proba([landmarks])[0]
                confidence = np.max(probabilities) * 100

                cv2.putText(frame, f"Sign: {prediction}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                cv2.putText(frame, f"Confidence: {confidence:.1f}%", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                #  Debounce + add letter to word 
                if confidence >= CONFIDENCE_THRESHOLD:
                    if prediction == stable_prediction:
                        stable_count += 1
                    else:
                        stable_prediction = prediction
                        stable_count = 1
                        next_add_threshold = STABLE_FRAMES_REQUIRED

                    if stable_count == next_add_threshold:
                        current_word += prediction
                        speak(prediction)
                        next_add_threshold = stable_count + REPEAT_FRAMES_REQUIRED
                else:
                    stable_prediction = None
                    stable_count = 0
            else:
                cv2.putText(frame, "No Hand Detected", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                stable_prediction = None
                stable_count = 0

                # --- Space insertion logic ---
                no_hand_count += 1
                if (no_hand_count == NO_HAND_FRAMES_FOR_SPACE
                        and not space_pending
                        and current_word
                        and not current_word.endswith(" ")):
                    current_word += " "
                    space_pending = True

            # --- Display the word being built 
            cv2.rectangle(frame, (0, frame.shape[0] - 60), (frame.shape[1], frame.shape[0]),
                          (50, 50, 50), -1)
            cv2.putText(frame, f"Word: {current_word}", (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, "q:quit  c:clear  b:backspace", (10, frame.shape[0] - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("Live Sign Prediction", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                current_word = ""
                stable_prediction = None
                stable_count = 0
            elif key == ord('b'):
                current_word = current_word[:-1]
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        print("Exited cleanly.")


if __name__ == "__main__":
    main()