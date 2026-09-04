"""
live_predict.py

Full end-to-end demo: opens the camera, detects hand landmarks, runs them
through the trained classifier, shows the predicted sign + confidence on
screen, and speaks the predicted letter aloud (with debouncing so it
doesn't repeat every frame).

Requires: model/sign_language_model.pkl and model/labels.json
          (created by train_model.py)

Controls:
    q -> quit

Speaking behavior:
    - A prediction is only spoken once it has been stable (same letter,
      confidence above threshold) for STABLE_FRAMES_REQUIRED consecutive
      frames, and only if it's different from the last spoken letter.
    - This avoids the model repeating the same letter nonstop while you
      hold a sign in front of the camera.
"""

import json
import threading
import cv2
import joblib
import numpy as np
import pyttsx3
from hand_detector import HandDetector

CONFIDENCE_THRESHOLD = 60.0   # % - below this, prediction is ignored
STABLE_FRAMES_REQUIRED = 10   # consecutive matching frames before speaking


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
        """
        Runs TTS in a background thread so the camera feed doesn't freeze
        while speaking. Re-initializes the engine each call as a workaround
        for a known pyttsx3 issue on Windows where reusing one engine
        instance across multiple say()/runAndWait() calls silently fails
        after the first call.
        """
        def _speak():
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.say(text)
            engine.runAndWait()
            engine.stop()

        threading.Thread(target=_speak, daemon=True).start()

    last_spoken = None
    stable_prediction = None
    stable_count = 0

    print("Model loaded. Press 'q' to quit.")

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
                prediction = model.predict([landmarks])[0]
                probabilities = model.predict_proba([landmarks])[0]
                confidence = np.max(probabilities) * 100

                cv2.putText(frame, f"Sign: {prediction}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                cv2.putText(frame, f"Confidence: {confidence:.1f}%", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

                # --- Debounce + speak logic ---
                if confidence >= CONFIDENCE_THRESHOLD:
                    if prediction == stable_prediction:
                        stable_count += 1
                    else:
                        stable_prediction = prediction
                        stable_count = 1

                    if stable_count == STABLE_FRAMES_REQUIRED and prediction != last_spoken:
                        speak(prediction)
                        last_spoken = prediction
                else:
                    stable_prediction = None
                    stable_count = 0
            else:
                cv2.putText(frame, "No Hand Detected", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                stable_prediction = None
                stable_count = 0
                last_spoken = None  # allow re-speaking same letter after hand leaves and returns

            cv2.imshow("Live Sign Prediction", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        print("Exited cleanly.")


if __name__ == "__main__":
    main()