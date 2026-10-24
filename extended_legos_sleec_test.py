from textx import metamodel_from_str

from verifier.generate_mutants import get_mutants
from verifier.sleec2marabou_parser import MarabouGenerator

# ==========================================
# 1. LOAD TEXTX GRAMMAR
# ==========================================
grammar_file = "legos_sleec/sleec-grammar.txt"
with open(grammar_file, 'r') as f:
    grammar = f.read()

# ==========================================
# 2. RUNNING THE PARSER
# ==========================================
# sleec_input_file = "resources/requirements/pima_indians.sleec"
sleec_input_file = "resources/requirements/lane_changer.sleec"
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

# ONNX_MODEL_FILENAME = "verifier/resources/models/pima_classifier/pima_marabou.onnx"
ONNX_MODEL_FILENAME = "verifier/resources/models/ngsim_lanechange_marabou.onnx"
# ONNX_MODEL_FILENAME = "verifier/resources/models/dressassist_carebot/carebot_decision_model_v2.onnx"
outcome = check_property(ONNX_MODEL_FILENAME)
print(outcome)

mutated_requirements = get_mutants()
verified_mutants = []
for mutations, req in mutated_requirements:
    print(req)
    with open(sleec_input_file, "r") as original_f:
        lines = original_f.readlines()
        new_lines = []
        for line in lines:
            if line.startswith("    r1"):
                new_lines.append(f"{req}\n")
            else:
                new_lines.append(line)
        with open(sleec_input_file.replace(".sleec", "_mutated.sleec"), "w") as mutated_f:
            mutated_f.writelines(new_lines)

    with open(sleec_input_file.replace(".sleec", "_mutated.sleec"), 'r') as f:
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

    # ONNX_MODEL_FILENAME = "verifier/resources/models/pima_classifier/pima_marabou.onnx"
    ONNX_MODEL_FILENAME = "verifier/resources/models/ngsim_lanechange_marabou.onnx"
    # ONNX_MODEL_FILENAME = "verifier/resources/models/dressassist_carebot/carebot_decision_model_v2.onnx"
    outcome = check_property(ONNX_MODEL_FILENAME)

    if all(verified for verified, _, _ in outcome["results"].values()):
        verified_mutants.append(req)

print("The NN verifies:")
for req in verified_mutants:
    print(req)