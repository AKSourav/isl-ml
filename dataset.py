import time
import cv2
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

DATA_DIR = 'isl_dataset'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

SAMPLES_PER_LABEL = 100

def extract_landmarks(results_pose, results_hands):
    # Pose landmarks (33 points × 4 features)
    if results_pose.pose_landmarks:
        pose_landmarks = []
        for lm in results_pose.pose_landmarks.landmark:
            pose_landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
    else:
        pose_landmarks = [0] * (33 * 4)

    # Hand landmarks (2 hands max × 21 points × 3 features)
    hand_landmarks = []
    if results_hands.multi_hand_landmarks:
        for hand_landmark in results_hands.multi_hand_landmarks:
            for lm in hand_landmark.landmark:
                hand_landmarks.extend([lm.x, lm.y, lm.z])
        if len(results_hands.multi_hand_landmarks) == 1:
            hand_landmarks.extend([0] * (21 * 3))  # Pad second hand
    else:
        hand_landmarks = [0] * (21 * 3 * 2)  # No hands detected

    return pose_landmarks + hand_landmarks

cap = cv2.VideoCapture(0)

while True:
    word = input("Enter the word to collect data for (or 'exit' to quit): ").strip()
    if word.lower() == 'exit':
        break

    label_dir = os.path.join(DATA_DIR, word)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    count = 0
    frame_counter = 0
    print(f"Starting data collection for '{word}'. Press 'q' to stop early.")
    time.sleep(2)

    while count < SAMPLES_PER_LABEL:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_counter += 1
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_pose = pose.process(image_rgb)
        results_hands = hands.process(image_rgb)

        landmarks = extract_landmarks(results_pose, results_hands)

        # Save every 3rd frame with valid landmarks
        if any(landmarks) and (frame_counter % 3 == 0):
            np.save(os.path.join(label_dir, f'{count}.npy'), np.array(landmarks))
            count += 1
            print(f'Collected sample {count}/{SAMPLES_PER_LABEL} for "{word}"')

        cv2.putText(frame, f'Collecting "{word}": {count}/{SAMPLES_PER_LABEL}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('ISL Data Collection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('Early stop!')
            break

cap.release()
cv2.destroyAllWindows()
hands.close()
pose.close()
