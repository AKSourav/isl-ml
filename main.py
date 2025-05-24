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
        X = np.array(landmarks).reshape(1, -1)
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
