import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

initial_type = [('float_input', FloatTensorType([None, 258]))]  # input size matching your features
clf = joblib.load("isl_rf_model.pkl")
onx = convert_sklearn(
    clf, 
    initial_types=initial_type,
    options={id(clf): {'zipmap': False}}  # disable zipmap to get probabilities as tensor
)

with open("isl_rf_model_single_output.onnx", "wb") as f:
    f.write(onx.SerializeToString())
