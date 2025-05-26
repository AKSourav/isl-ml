import joblib
from sklearn.preprocessing import LabelEncoder
import json

le = joblib.load('isl_label_encoder.pkl')

# Create mapping: index -> label
index_to_label = {int(idx): label for idx, label in enumerate(le.classes_)}

# Convert dict to JSON string suitable for JS
js_object_str = json.dumps(index_to_label, indent=2)

print("const LABELS = " + js_object_str + ";")
