from textx import metamodel_from_str

# ==========================================
# 1. THE TEXTX GRAMMAR
# ==========================================
grammar_file = "../LEGOS-SLEEC/sleec-gramar.tx"
with open(grammar_file, 'r') as f:
    grammar = f.read()


# ==========================================
# 2. THE GENERATOR LOGIC
# ==========================================
class MarabouGenerator:
    def __init__(self, model):
        self.model = model
        self.constants = {}
        self.scales = {}
        self.measures = {}  # Map measure name to index
        self.covered_inputs = []
        self.outputs = {}  # Map output event name to index

        # Populate symbol tables
        self._scan_definitions()

    def _scan_definitions(self):
        # 1. Load Constants
        for item in self.model.definitions:
            if item.__class__.__name__ == 'Constant':
                self.constants[item.name] = item.value

        # 2. Map Inputs (Measures) to Indices (0, 1, 2...)
        input_idx = 0
        for item in self.model.definitions:
            if 'Measure' in item.__class__.__name__:
                self.measures[item.name] = input_idx
                input_idx += 1
            if hasattr(item, "type") and 'Scale' in item.type.__class__.__name__:
                self.scales[item.name] = [p.name for p in item.type.scaleParams]

        # 3. Map Output Events to Indices
        # TODO: For now, we assume output events have 'output' in their name
        out_idx = 0
        for item in self.model.definitions:
            if item.__class__.__name__ == 'Event' and 'output' in item.name:
                self.outputs[item.name] = out_idx
                out_idx += 1

    def _resolve_value(self, val_name, bound="LB"):
        """Resolves a value name (e.g., 'mediumDistress') to a number."""
        # Check if it's a defined constant (e.g., mediumDistressLB)
        # In a real compiler, we would check for specific suffixes or exact matches.
        # Here we look for exact match or match + "LB" based on snippet conventions.
        if val_name in self.constants and not bound:
            return self.constants[val_name].value
        if f"{val_name}LB" in self.constants and bound == "LB":
            return self.constants[f"{val_name}LB"].value
        if f"{val_name}UB" in self.constants and bound == "UB":
            return self.constants[f"{val_name}UB"].value

        # If it's an enum (like noUserCmd), usually mapped to 0, 1, 2.
        for scale in self.scales.values():
            if val_name in scale:
                return scale.index(val_name)

        # Since the snippet doesn't define enum integers, we default to 0.0 for the example
        return 0.0  # Default placeholder

    def generate(self):
        output = []
        output.append("from maraboupy import Marabou, MarabouCore, MarabouUtils")
        output.append("\ndef check_property(onnx_path):")

        for rule in self.model.ruleBlock.rules:
            output.append(self._generate_rule_check(rule))
            self.covered_inputs = []

        return "\n".join([line for line in output if line is not None])

    def _generate_rule_check(self, rule):
        # TODO: no need to generate code for the fact that outputs are mutually exclusive
        # (these must be named o1, o2, ...)
        if rule.name[0] == 'o':
            return None

        code = []
        code.append(f"\n    # --- Checking Rule: {rule.name} ---")

        # Generate Case A: The Normal 'Then'
        code.append(f"    print('Checking Normal Case for {rule.name}...')")
        code.append(f"    net = Marabou.read_onnx(onnx_path)")
        code.append(self._apply_condition(rule.condition, "Normal"))

        # If there is an 'unless', we must negate it for the Normal case
        if len(rule.response.defeater) > 0:
            code.append(self._apply_condition(rule.response.defeater[0].expr, "Negate"))

        vars_to_cover = list(set(self.measures.values()).difference(set(self.covered_inputs)))
        code.append(self._add_remaining_conditions(vars_to_cover))

        code.append(self._apply_output_check(rule.response.occ.event.event))

        # Generate Case B: The Exception 'Unless'
        if len(rule.response.defeater) > 0:
            self.covered_inputs = []
            code.append(f"\n    print('Checking Exception Case for {rule.name}...')")
            code.append(f"    net = Marabou.read_onnx(onnx_path)")
            code.append(self._apply_condition(rule.condition, "Normal"))  # Precondition holds
            code.append(self._apply_condition(rule.response.defeater[0].expr, "Normal"))  # Exception triggers
            vars_to_cover = list(set(self.measures.values()).difference(set(self.covered_inputs)))
            code.append(self._add_remaining_conditions(vars_to_cover))
            if rule.response.defeater[0].response is None:
                code.append(self._apply_output_check(rule.response.occ.event.event, mode="Negate"))
            else:
                code.append(self._apply_output_check(rule.response.defeater[0].response.occ.event.event))

        return "\n".join(code)

    def _apply_condition(self, condition, mode):
        # TODO: add the 'or' case
        if condition.op == 'and':
            return self._apply_condition(condition.lhs, mode) + "\n" + self._apply_condition(condition.rhs, mode)

        # variable, operator, value
        c = condition.op
        var_idx = self.measures[condition.lhs.ID.name]
        val = self._resolve_value(condition.rhs.ID.name)
        op = c

        lines = []

        # Logic to handle Marabou Bounds based on Operator and Mode (Normal vs Negate)
        eq_constraint = op == '='
        is_greater = '>' in op

        if mode == "Negate":
            # Invert logic: NOT (> 50) becomes <= 50
            is_greater = not is_greater

        if eq_constraint:
            if mode == "Negate":
                val_lb = self._resolve_value(condition.lhs.ID.name, bound="LB")
                val_ub = self._resolve_value(condition.lhs.ID.name, bound="UB")

                lines.append(f"    net.setLowerBound({var_idx}, {val_lb})")
                lines.append(f"    net.setUpperBound({var_idx}, {val_ub})")

                # TODO: make sure this works with more than one equality constraint
                lines.append(f"    eq1 = MarabouCore.Equation(MarabouCore.Equation.LE)\n" +
                             f"    eq1.addAddend(1, net.inputVars[0][0][{var_idx}])\n" +
                             f"    eq1.setScalar({val - 1})")
                lines.append(f"    eq2 = MarabouCore.Equation(MarabouCore.Equation.GE)\n" +
                             f"    eq2.addAddend(1, net.inputVars[0][0][{var_idx}])\n" +
                             f"    eq2.setScalar({val + 1})")
                lines.append(f"    net.addDisjunctionConstraint([[eq1], [eq2]])")
            else:
                lines.append(f"    net.setLowerBound({var_idx}, {val})")
                lines.append(f"    net.setUpperBound({var_idx}, {val})")
        elif is_greater:
            # Lower bound constraint
            lines.append(f"    net.setLowerBound({var_idx}, {val}) # {condition.lhs.ID.name} >= {val}")
            val_ub = self._resolve_value(condition.lhs.ID.name, bound="UB")
            lines.append(f"    net.setUpperBound({var_idx}, {val_ub}) # {condition.lhs.ID.name} < {val_ub}")
        else:
            lines.append(f"    net.setUpperBound({var_idx}, {val}) # {condition.lhs.ID.name} <= {val}")
            val_lb = self._resolve_value(condition.lhs.ID.name, bound="LB")
            lines.append(f"    net.setLowerBound({var_idx}, {val_lb}) # {condition.lhs.ID.name} > {val_lb}")

        self.covered_inputs.append(var_idx)

        return "\n".join(lines)

    def _add_remaining_conditions(self, vars_to_cover):
        lines = []
        for var in vars_to_cover:
            var_idx = [id for id in self.measures if self.measures[id] == var][0]
            val_lb = self._resolve_value(var_idx, "LB")
            val_ub = self._resolve_value(var_idx, "UB")
            lines.append(f"    net.setLowerBound({var}, {val_lb})")
            lines.append(f"    net.setUpperBound({var}, {val_ub})")
        return "\n".join(lines)

    def _apply_output_check(self, target_action, mode="Normal"):
        target_idx = self.outputs[target_action.name]
        lines = []

        if mode == "Normal":
            lines.append(f"    # Verify output is {target_action.name} (Index {target_idx})")
            lines.append(f"    # Assert Negation: exists other output >= {target_action.name}")
        else:
            lines.append(f"    # Verify output is NOT {target_action.name} (Index {target_idx})")
            lines.append(f"    # Assert Negation:  {target_action.name} >= all other outputs")

        lines.append(f"    outputVars = net.outputVars[0][0]")
        lines.append(f"    equations = []")

        for name, idx in self.outputs.items():
            if idx == target_idx: continue
            if mode == "Normal":
                lines.append(f"    # {name} >= {target_action}")
                lines.append(f"    eq = MarabouCore.Equation(MarabouCore.Equation.GE)")
                lines.append(f"    eq.addAddend(1, outputVars[{idx}])")
                lines.append(f"    eq.addAddend(-1, outputVars[{target_idx}])")
                lines.append(f"    equations.append([eq])")
            else:
                lines.append(f"    # {name} <= {target_action}")
                lines.append(f"    eq = MarabouUtils.Equation(MarabouCore.Equation.GE)")
                lines.append(f"    eq.addAddend(1, outputVars[{target_idx}])")
                lines.append(f"    eq.addAddend(-1, outputVars[{idx}])")
                lines.append(f"    eq.setScalar(0)")
                lines.append(f"    net.addEquation(eq)")

        if mode == "Normal":
            lines.append(f"    net.addDisjunctionConstraint(equations)")
        lines.append(f"    result = net.solve()")
        lines.append(f"    if result[0] == 'sat': print('  -> FAILED: Counter-example found')")
        lines.append(f"    else: print('  -> VERIFIED')")
        return "\n".join(lines)


# ==========================================
# 3. RUNNING THE PARSER
# ==========================================

sleec_input_file = "../resources/requirements/dressassist_nn.sleec"
with open(sleec_input_file, 'r') as f:
    dsl_snippet = f.read()

# Parse
meta = metamodel_from_str(grammar)
model = meta.model_from_str(dsl_snippet)

# Generate Code
gen = MarabouGenerator(model)
with open("generated_verify_model.py", 'w') as f:
    f.write(gen.generate())
