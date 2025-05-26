import joblib
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes

# Load your trained model
clf = joblib.load('sign_language_classifier.joblib')

# Input shape (1 sample, feature size)
initial_type = [('input', FloatTensorType([None, clf.n_features_in_]))]

# Conversion options: disable zipmap so output probabilities are raw tensors
options = {id(clf): {'zipmap': False}}

# Convert model
onnx_model = convert_sklearn(clf, initial_types=initial_type, options=options)

# Save ONNX model
with open("sign_language_classifier.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("ONNX model saved as sign_language_classifier.onnx")
