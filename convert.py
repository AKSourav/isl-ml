import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

model = joblib.load("isl_rf_model.pkl")

initial_type = [("float_input", FloatTensorType([None, 258]))]  # 258 features

onnx_model = convert_sklearn(model, initial_types=initial_type)

with open("isl_rf_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())


# delete one output
import onnx

model = onnx.load("isl_rf_model.onnx")

# Remove output_probability from model.graph.output
model.graph.output.pop(1)  # assuming output_probability is second

onnx.save(model, "isl_rf_model_single_output.onnx")
