# 3. Translate the model to Marabou API calls (simplified example)
def translate_to_marabou(model):
    print(f"# --- Translating property: {model.name} ---")

    # The model object directly has the structure we need
    input_constraints = model.input_constraints
    output_constraint = model.output_constraint

    num_inputs = len(input_constraints)
    print(f"marabou_network.setNumberOfInputs({num_inputs})")
    print("# Setting input constraints:")
    for i, constraint in enumerate(input_constraints):
        # We can access attributes directly by name!
        print(f"marabou_network.setLowerBound({i}, {constraint.lower_bound}) # Var: {constraint.variable}")
        print(f"marabou_network.setUpperBound({i}, {constraint.upper_bound})")

    print("\n# Setting output constraint:")
    # The output constraint is also a rich object
    output_idx = output_constraint.index
    op = output_constraint.op
    val = output_constraint.value

    # This logic maps our DSL operator to Marabou's equation format
    if op == '>':
        # Marabou represents y > C as -y < -C
        print(f"marabou_network.addInequality([{output_idx}], [-1.0], {-val})")
    # ... (add logic for other operators)

    print("# --- End of translation ---")