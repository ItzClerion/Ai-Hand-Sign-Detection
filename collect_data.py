"""
Usage:
    python collect_data.py A
    python collect_data.py B
    ...

Controls (once camera window is open):
    r  -> start/stop recording (hold toggled on, collects every frame while ON)
    q  -> quit and save

Saves data to dataset/<LABEL>.csv with rows: x1,y1,z1,x2,y2,z2,...,x21,y21,z21
"""

import sys
import os
import csv
import cv2
from hand_detector import HandDetector

SAMPLES_TARGET = 300  # aim for this many samples per letter


def main():
    if len(sys.argv) != 2:
        print("Usage: python collect_data.py <LABEL>")
        print("Example: python collect_data.py A")
        sys.exit(1)

    label = sys.argv[1].upper()

    os.makedirs("dataset", exist_ok=True)
    csv_path = os.path.join("dataset", f"{label}.csv")

    # Open in append mode so you can pause and resume collecting for the same letter
    file_exists = os.path.exists(csv_path)
    csv_file = open(csv_path, "a", newline="")
    writer = csv.writer(csv_file)

    if not file_exists:
        header = [f"{axis}{i}" for i in range(1, 22) for axis in ("x", "y", "z")]
        writer.writerow(header)

    # Count existing samples so target tracking is accurate across sessions
    existing_count = 0
    if file_exists:
        with open(csv_path, "r") as f:
            existing_count = sum(1 for _ in f) - 1  # minus header

    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    recording = False
    sample_count = existing_count

    print(f"Collecting samples for '{label}'. Target: {SAMPLES_TARGET}")
    print("Press 'r' to toggle recording ON/OFF. Press 'q' to quit and save.")

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        frame, results = detector.find_hands(frame)
        landmarks = detector.extract_landmarks(results)

        if recording and landmarks is not None:
            writer.writerow(landmarks.tolist())
            sample_count += 1

        # --- On-screen info ---
        status_color = (0, 0, 255) if recording else (200, 200, 200)
        status_text = "RECORDING" if recording else "PAUSED"
        cv2.putText(frame, f"Label: {label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, status_text, (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(frame, f"Samples: {sample_count}/{SAMPLES_TARGET}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if landmarks is None:
            cv2.putText(frame, "No hand detected", (10, 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        cv2.imshow("Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            recording = not recording
        elif key == ord('q'):
            break

        if sample_count >= SAMPLES_TARGET:
            print(f"Target reached: {sample_count} samples collected for '{label}'.")
            # Keep window open in case user wants to keep going anyway; just notify once
            recording = False

    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()
    print(f"Saved {sample_count} total samples to {csv_path}")


if __name__ == "__main__":
    main()