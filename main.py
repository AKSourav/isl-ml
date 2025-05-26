import cv2
import numpy as np
import mediapipe as mp
import joblib

# Load trained model
model = joblib.load('sign_language_classifier.joblib')

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(static_image_mode=False,
                                min_detection_confidence=0.7,
                                min_tracking_confidence=0.7)

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
    ret, frame = cap.read()
    if not ret:
        break

    # Process unflipped frame for correct landmark order
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)

    # Extract landmarks & predict
    landmarks = extract_landmarks(results)
    if any(landmarks):
        landmarks_np = np.array(landmarks).reshape(1, -1)
        prediction = model.predict(landmarks_np)[0]
    else:
        prediction = "No hands/pose detected"

    # Draw landmarks on the *unflipped* frame
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    # Flip frame horizontally for user-friendly mirror view
    frame = cv2.flip(frame, 1)

    # Show prediction text
    cv2.putText(frame, f'Prediction: {prediction}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Sign Language Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
holistic.close()
