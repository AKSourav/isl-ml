import pickle
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Load pickle file
model_path = 'modelislv17.p'
with open(model_path, 'rb') as file:
    content = pickle.load(file)

# Check if the loaded object is a model or a dictionary
if isinstance(content, dict):
    # Adjust this if your model is stored under a different key
    clf = content.get('model')  # or content['model'] if you're sure
else:
    clf = content

# Validate clf is a scikit-learn estimator
if not hasattr(clf, "n_features_in_"):
    raise ValueError("Loaded object does not have n_features_in_ — likely not a valid scikit-learn model.")

# Automatically extract input feature count
n_features = clf.n_features_in_
initial_type = [('float_input', FloatTensorType([None, n_features]))]

# Convert model to ONNX
onx = convert_sklearn(
    clf,
    initial_types=initial_type,
    options={id(clf): {'zipmap': False}}
)

# Print inputs/outputs
print("Inputs:")
for inp in onx.graph.input:
    dims = [dim.dim_value if dim.HasField("dim_value") else "None" for dim in inp.type.tensor_type.shape.dim]
    print(f"  {inp.name}: shape={dims}")

print("Outputs:")
for out in onx.graph.output:
    dims = [dim.dim_value if dim.HasField("dim_value") else "None" for dim in out.type.tensor_type.shape.dim]
    print(f"  {out.name}: shape={dims}")

# Save to ONNX file
with open("isl_rf_model_dual_output.onnx", "wb") as f:
    f.write(onx.SerializeToString())
