"""
hand_detector.py

Reusable hand detection module built on top of MediaPipe Hands.

Given a video frame, this module:
  1. Detects a hand and its 21 landmarks (wrist, finger joints, fingertips).
  2. Converts those landmarks into a flat 63-value feature vector (21 points x, y, z)
     that is translation-invariant, scale-invariant, and temporally smoothed —
     making it suitable as direct input to a classifier (e.g. Random Forest).

Usage:
    from hand_detector import HandDetector

    detector = HandDetector()
    frame, results = detector.find_hands(frame)
    landmarks = detector.extract_landmarks(results)   # np.ndarray of shape (63,) or None
"""

import cv2
import mediapipe as mp
import numpy as np


class HandDetector:
    def __init__(self, max_hands=1, detection_confidence=0.7, tracking_confidence=0.7,
                 model_complexity=1, smoothing_factor=0.5, min_handedness_confidence=0.6):
        """
        Args:
            max_hands: max number of hands to detect at once.
            detection_confidence: min confidence to consider a hand "detected".
            tracking_confidence: min confidence to keep tracking a hand across frames.
            model_complexity: 0 (faster, less accurate) or 1 (slower, more accurate).
            smoothing_factor: 0-1, how much weight the newest frame gets when
                blending with the previous frame's landmarks. Higher = less smoothing.
            min_handedness_confidence: landmarks below this confidence are discarded
                (helps filter out blurry/partial detections).
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=model_complexity,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils

        self.smoothing_factor = smoothing_factor
        self.min_handedness_confidence = min_handedness_confidence
        self.prev_landmarks = None

    def find_hands(self, frame, draw=True):
        """
        Detects hands in a BGR frame (as read by OpenCV).

        Args:
            frame: BGR image (numpy array) from cv2.VideoCapture.
            draw: if True, draws the hand skeleton on the frame for visualization.

        Returns:
            (frame, results) - the (possibly annotated) frame, and the raw
            MediaPipe results object (needed by extract_landmarks).
        """
        if frame is None:
            return frame, None

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks and draw:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
        return frame, results

    def extract_landmarks(self, results):
        """
        Converts MediaPipe results into a flat 63-length feature vector.

        Normalization applied:
          - Translation-invariant: coordinates are relative to the wrist (landmark 0),
            so hand position on screen doesn't matter.
          - Scale-invariant: coordinates are divided by the max wrist-to-landmark
            distance, so hand size / distance from camera doesn't matter.
          - Temporally smoothed: blended with the previous frame's landmarks
            (exponential moving average) to reduce frame-to-frame jitter.
          - Confidence-filtered: low-confidence detections are discarded.

        Args:
            results: the MediaPipe results object returned by find_hands().

        Returns:
            np.ndarray of shape (63,), or None if no valid hand was detected.
        """
        if results is None or not results.multi_hand_landmarks:
            self.prev_landmarks = None
            return None

        # Reject low-confidence detections (partial occlusion, motion blur, etc.)
        if results.multi_handedness:
            confidence = results.multi_handedness[0].classification[0].score
            if confidence < self.min_handedness_confidence:
                self.prev_landmarks = None
                return None

        hand = results.multi_hand_landmarks[0]  # first detected hand
        base_x, base_y, base_z = hand.landmark[0].x, hand.landmark[0].y, hand.landmark[0].z

        coords = np.array([
            [lm.x - base_x, lm.y - base_y, lm.z - base_z]
            for lm in hand.landmark
        ])

        # Scale normalization: divide by max distance from wrist to any landmark
        scale = np.max(np.linalg.norm(coords, axis=1))
        if scale > 0:
            coords = coords / scale

        landmarks = coords.flatten()  # shape (63,)

        # Temporal smoothing: blend with previous frame's landmarks
        if self.prev_landmarks is not None and self.prev_landmarks.shape == landmarks.shape:
            landmarks = (self.smoothing_factor * landmarks +
                         (1 - self.smoothing_factor) * self.prev_landmarks)

        self.prev_landmarks = landmarks
        return landmarks

    def close(self):
        """Releases MediaPipe resources. Call when done using the detector."""
        self.hands.close()