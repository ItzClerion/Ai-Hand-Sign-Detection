"""
test_detector.py

Standalone script to visually verify that hand detection is working.
Opens the webcam, detects a hand each frame, and shows the landmark
skeleton overlaid on the video feed.

Press 'q' to quit.
"""

import cv2
from hand_detector import HandDetector


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not access the camera.")
        print("Check that:")
        print("  - No other application (Zoom, Teams, etc.) is using the camera.")
        print("  - The correct camera index is used (try cv2.VideoCapture(1) if you have multiple cameras).")
        return

    detector = HandDetector()

    print("Camera opened. Press 'q' to quit.")

    try:
        while True:
            success, frame = cap.read()
            if not success:
                print("WARNING: Failed to read a frame from the camera. Stopping.")
                break

            frame = cv2.flip(frame, 1)  # mirror view, feels natural
            frame, results = detector.find_hands(frame)
            landmarks = detector.extract_landmarks(results)

            if landmarks is not None:
                cv2.putText(frame, "Hand Detected", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No Hand Detected", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Hand Detection Test", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        print("Camera released. Exiting cleanly.")


if __name__ == "__main__":
    main()