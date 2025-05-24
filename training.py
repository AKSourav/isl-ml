import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import mediapipe as mp

mp_pose = mp.solutions.pose

DATA_DIR = 'isl_dataset'


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


def normalize_sample(sample):
    # sample is 258 features: 132 pose + 126 hands
    pose_part = sample[:132]
    hands_part = sample[132:]

    norm_pose = normalize_pose_landmarks(pose_part)
    
    # split hands: 2 hands × 63 features (21 points * 3 coords)
    hand1 = hands_part[:63]
    hand2 = hands_part[63:]

    norm_hand1 = normalize_hand_landmarks(hand1)
    norm_hand2 = normalize_hand_landmarks(hand2)

    return np.concatenate([norm_pose, norm_hand1, norm_hand2])


X = []
y = []

for label in os.listdir(DATA_DIR):
    label_dir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_dir):
        continue

    files = [f for f in os.listdir(label_dir) if f.endswith('.npy')]
    for file in files:
        features = np.load(os.path.join(label_dir, file))

        # Normalize each sample here
        norm_features = normalize_sample(features)

        X.append(norm_features)
        y.append(label)

X = np.array(X)
y = np.array(y)

print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features each.")

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

joblib.dump(clf, 'isl_rf_model.pkl')
joblib.dump(le, 'isl_label_encoder.pkl')

print("Model and label encoder saved.")
