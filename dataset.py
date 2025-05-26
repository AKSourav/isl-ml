import cv2
import numpy as np
import os
import mediapipe as mp
import time

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7)

DATA_DIR = 'isl_dataset_holistic'
os.makedirs(DATA_DIR, exist_ok=True)
SAMPLES_PER_LABEL = 100

def normalize_landmarks(landmarks, origin_index=0):
    coords = np.array(landmarks).reshape(-1, 3)
    origin = coords[origin_index]
    coords -= origin
    max_dist = np.max(np.linalg.norm(coords, axis=1))
    if max_dist > 0:
        coords /= max_dist
    return coords.flatten().tolist()

def extract_landmarks(results):
    pose = []
    if results.pose_landmarks:
        pose_coords = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
        pose = normalize_landmarks(pose_coords, origin_index=mp_holistic.PoseLandmark.LEFT_HIP.value)

    left_hand = []
    if results.left_hand_landmarks:
        left_hand_coords = [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]
        left_hand = normalize_landmarks(left_hand_coords)

    right_hand = []
    if results.right_hand_landmarks:
        right_hand_coords = [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]
        right_hand = normalize_landmarks(right_hand_coords)

    # Pad missing parts
    pose = pose if pose else [0] * (33 * 3)
    left_hand = left_hand if left_hand else [0] * (21 * 3)
    right_hand = right_hand if right_hand else [0] * (21 * 3)

    return pose + left_hand + right_hand

cap = cv2.VideoCapture(0)

while True:
    word = input("Enter word for data collection (or 'exit'): ").strip()
    if word.lower() == 'exit':
        break

    label_dir = os.path.join(DATA_DIR, word)
    os.makedirs(label_dir, exist_ok=True)

    count = 0
    frame_counter = 0
    print(f"Starting collection for '{word}'...")
    time.sleep(2)

    while count < SAMPLES_PER_LABEL:
        ret, frame = cap.read()
        if not ret:
            continue

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)
        frame_counter += 1

        landmarks = extract_landmarks(results)

        if any(landmarks) and frame_counter % 3 == 0:
            np.save(os.path.join(label_dir, f'{count}.npy'), np.array(landmarks))
            count += 1
            print(f'Sample {count}/{SAMPLES_PER_LABEL} collected for "{word}"')

        cv2.putText(frame, f'Collecting "{word}": {count}/{SAMPLES_PER_LABEL}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.imshow('Data Collection (Holistic)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Early exit.")
            break

cap.release()
cv2.destroyAllWindows()
holistic.close()
