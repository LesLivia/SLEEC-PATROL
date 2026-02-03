from textx import metamodel_from_str

from verifier.sleec2marabou_parser import MarabouGenerator

# ==========================================
# 1. LOAD TEXTX GRAMMAR
# ==========================================
grammar_file = "LEGOS-SLEEC/sleec-gramar.tx"
with open(grammar_file, 'r') as f:
    grammar = f.read()

# ==========================================
# 2. RUNNING THE PARSER
# ==========================================
sleec_input_file = "resources/requirements/pima_indians.sleec"
# sleec_input_file = "resources/requirements/dressassist_nn.sleec"
with open(sleec_input_file, 'r') as f:
    dsl_snippet = f.read()

# Parse
meta = metamodel_from_str(grammar)
model = meta.model_from_str(dsl_snippet)

# Generate Code
gen = MarabouGenerator(model)
with open("verifier/generated_verify_model.py", 'w') as f:
    f.write(gen.generate())

# ==========================================
# 3. RUNNING THE VERIFIER
# ==========================================
from verifier.generated_verify_model import check_property

ONNX_MODEL_FILENAME = "verifier/resources/models/pima_classifier/pima_marabou.onnx"
# ONNX_MODEL_FILENAME = "verifier/resources/models/dressassist_carebot/carebot_decision_model_v2.onnx"
check_property(ONNX_MODEL_FILENAME)
