import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

DATA_DIR = 'isl_dataset_holistic'

# Load data
X, y = [], []
for label in os.listdir(DATA_DIR):
    label_dir = os.path.join(DATA_DIR, label)
    if not os.path.isdir(label_dir):
        continue
    for file in os.listdir(label_dir):
        if file.endswith('.npy'):
            data = np.load(os.path.join(label_dir, file))
            X.append(data)
            y.append(label)

X = np.array(X)
y = np.array(y)

print(f"Loaded {len(X)} samples from {len(set(y))} classes.")

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(clf, 'sign_language_classifier.joblib')
print("Model saved as sign_language_classifier.joblib")
