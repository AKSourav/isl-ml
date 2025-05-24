import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

DATA_DIR = 'isl_dataset'

X = []
y = []

for label in os.listdir(DATA_DIR):
    label_dir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_dir):
        continue

    files = [f for f in os.listdir(label_dir) if f.endswith('.npy')]
    for file in files:
        features = np.load(os.path.join(label_dir, file))
        X.append(features)
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
