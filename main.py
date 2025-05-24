import cv2
import mediapipe as mp
import numpy as np
import joblib

clf = joblib.load('isl_rf_model.pkl')
le = joblib.load('isl_label_encoder.pkl')

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)


def normalize_pose_landmarks(pose_landmarks):
    coords = np.array(pose_landmarks).reshape(33, 4)[:, :3]

    left_hip = coords[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = coords[mp_pose.PoseLandmark.RIGHT_HIP.value]
    origin = (left_hip + right_hip) / 2

    coords -= origin

    left_shoulder = coords[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = coords[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    torso_size = np.linalg.norm(left_shoulder - right_shoulder) + np.linalg.norm(left_hip - right_hip)

    if torso_size > 0:
        coords /= torso_size

    visibility = np.array(pose_landmarks).reshape(33, 4)[:, 3]
    normalized = np.hstack([coords, visibility.reshape(-1, 1)]).flatten()
    return normalized


def normalize_hand_landmarks(hand_landmarks):
    coords = np.array(hand_landmarks).reshape(21, 3)

    origin = coords[0]
    coords -= origin

    max_dist = np.max(np.linalg.norm(coords, axis=1))
    if max_dist > 0:
        coords /= max_dist

    return coords.flatten()


def extract_landmarks(results_pose, results_hands):
    if results_pose.pose_landmarks:
        pose_landmarks = [val for lm in results_pose.pose_landmarks.landmark for val in (lm.x, lm.y, lm.z, lm.visibility)]
    else:
        pose_landmarks = [0] * (33 * 4)

    if results_hands.multi_hand_landmarks:
        hand_landmarks = []
        for hand_landmark in results_hands.multi_hand_landmarks:
            for lm in hand_landmark.landmark:
                hand_landmarks.extend([lm.x, lm.y, lm.z])
        if len(results_hands.multi_hand_landmarks) == 1:
            hand_landmarks.extend([0] * (21 * 3))  # pad second hand
    else:
        hand_landmarks = [0] * (21 * 3 * 2)

    return pose_landmarks + hand_landmarks


def normalize_landmarks(landmarks):
    # landmarks is full vector: 33*4 + 2*21*3 = 258 features
    pose_part = landmarks[:132]
    hands_part = landmarks[132:]

    norm_pose = normalize_pose_landmarks(pose_part)

    hand1 = hands_part[:63]
    hand2 = hands_part[63:]

    norm_hand1 = normalize_hand_landmarks(hand1)
    norm_hand2 = normalize_hand_landmarks(hand2)

    return np.concatenate([norm_pose, norm_hand1, norm_hand2])


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_pose = pose.process(image_rgb)
    results_hands = hands.process(image_rgb)

    landmarks = extract_landmarks(results_pose, results_hands)

    if any(landmarks):
        landmarks = normalize_landmarks(np.array(landmarks))
        X = landmarks.reshape(1, -1)

        probs = clf.predict_proba(X)[0]
        max_prob = max(probs)
        pred = clf.predict(X)[0]

        CONFIDENCE_THRESHOLD = 0.6
        if max_prob < CONFIDENCE_THRESHOLD:
            word = "null"
        else:
            word = le.inverse_transform([pred])[0]
    else:
        word = "No hands/pose detected"

    cv2.putText(frame, f'Prediction: {word}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('ISL Live Prediction', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
pose.close()
