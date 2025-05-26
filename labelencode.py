import os

DATA_DIR = 'isl_dataset_holistic'
labels = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])

# Create a JS object mapping index -> label string
js_lines = ["const LABELS = {"]
for i, label in enumerate(labels):
    js_lines.append(f'  {i}: "{label}",')
js_lines.append("};\n")
js_lines.append("export default LABELS;")  # optional for ES module export

js_code = "\n".join(js_lines)

# Save to labels.js
with open("labels.js", "w") as f:
    f.write(js_code)

print("JavaScript LABELS object saved to labels.js")
print(js_code)
