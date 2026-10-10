# ==========================================
# 2. THE GENERATOR LOGIC
# ==========================================

# Generation options. Override per run via MarabouGenerator(model, config={...}).
#   target_output_ub: extra upper bound on the output under test, or None for no
#       bound. Only narrows the search, so it can hide counterexamples; 0.0
#       reproduces the original behaviour.
#   output_semantics: "exclusive" for an argmax/softmax head, where an action is
#       taken iff it beats every other output; "independent" for per-output
#       sigmoids, where it must also clear decision_threshold. Under
#       "independent" several actions may be taken at once, or none.
#   decision_threshold: the threshold under "independent". Export the head
#       without its sigmoid and leave this at 0.0: sigmoid(z) >= 0.5 <=> z >= 0.
DEFAULT_CONFIG = {
    "target_output_ub": None,
    "output_semantics": "exclusive",
    "decision_threshold": 0.0,
}

# Operator seen under a 'not'.
NEGATED_OP = {'>': '<=', '<': '>=', '>=': '<', '<=': '>', '=': '<>', '<>': '='}


class _SleecNode:
    """Stand-in for a textX node, so normalization can build 'and' / 'not'."""

    def __init__(self, kind, **fields):
        self.__class__ = type(kind, (_SleecNode,), {})
        self.__dict__.update(fields)


def _and(lhs, rhs):
    if lhs is None:
        return rhs
    return _SleecNode('BoolBinaryOp', lhs=lhs, op='and', rhs=rhs)


def _not(expr):
    return _SleecNode('Negation', expr=expr, op=None)


class NormalizedRule:
    """A rule with no defeaters: when <condition> then <occ>."""

    def __init__(self, name, origin, condition, occ):
        self.name = name
        self.origin = origin
        self.condition = condition
        self.occ = occ


class MarabouGenerator:
    def __init__(self, model, config=None):
        self.model = model
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.constants = {}
        self.scales = {}
        self.measures = {}  # Map measure name to index
        self.measure_kinds = {}  # Map measure name to boolean / scale / numeric
        self.outputs = {}  # Map output event name to index
        self.trace = {}  # normalized rule name -> original rule name

        unknown = set(self.config) - set(DEFAULT_CONFIG)
        if unknown:
            raise ValueError(f"Unknown config option(s): {sorted(unknown)}")

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
            cls_name = item.__class__.__name__
            if 'Measure' in cls_name:
                self.measures[item.name] = input_idx
                self.measure_kinds[item.name] = (
                    'boolean' if cls_name == 'BoolMeasure'
                    else 'scale' if cls_name == 'ScalarMeasure' else 'numeric'
                )
                input_idx += 1
            if hasattr(item, "type") and 'Scale' in item.type.__class__.__name__:
                self.scales[item.name] = [p.name for p in item.type.scaleParams]

        # 3. Map output events to network output indices.
        #    Language rule: NN inputs are the measures, in declaration order;
        #    NN outputs are the events named output<Name>, in declaration order.
        #    Both orders must match the network's actual input/output layers.
        out_idx = 0
        for item in self.model.definitions:
            if item.__class__.__name__ == 'Event' and item.name.startswith('output'):
                self.outputs[item.name] = out_idx
                out_idx += 1

        if not self.outputs:
            raise ValueError(
                "No output events found. NN outputs must be declared as "
                "'event output<Name>', in the network's output order."
            )

    def _resolve_value(self, val_name, bound="LB"):
        """Resolve a value name used on the right of a comparison to a number.

        Exactly two cases, and no guessing between them:

        A scale label names a band of a real quantity, so its numeric edge lives
        in a separate constant: 'mediumGap' -> 'mediumGapLB' = 15. Which edge is
        the caller's choice via 'bound'; every spec so far wants the lower one,
        which is why that is the default. A label of a pure enum has no edge
        constant at all, and resolves to its ordinal instead.

        Anything else is a plain constant and must be named exactly. No suffix is
        tried for it: a name is either declared or it is a typo, and silently
        defaulting a typo builds the query over the wrong input box -- which, when
        it empties the box, makes Marabou answer unsat and reports the rule
        VERIFIED without it ever having been tested.
        """
        for scale in self.scales.values():
            if val_name in scale:
                edge = f"{val_name}{bound}"
                if edge in self.constants:
                    return self.constants[edge].value
                return scale.index(val_name)  # pure enum: the label IS the value

        if val_name in self.constants:
            # The old generator could only read '<name>LB', so specs declared a
            # dummy '<name> = 0' beside the real '<name>LB'. Both now resolve, so
            # the pair is ambiguous; collapse it rather than pick one silently.
            if f"{val_name}{bound}" in self.constants:
                raise ValueError(
                    f"'{val_name}' is ambiguous: both '{val_name}' "
                    f"(= {self.constants[val_name].value}) and '{val_name}{bound}' "
                    f"(= {self.constants[f'{val_name}{bound}'].value}) are declared. "
                    f"Keep one constant holding the real value."
                )
            return self.constants[val_name].value

        raise ValueError(
            f"cannot resolve '{val_name}' to a number: it is neither a declared "
            f"constant nor a label of a scale measure. Check the spelling."
        )

    # ---------------- normalization ----------------

    def normalize_rules(self):
        """Erase defeaters: one sub-rule for the main case, one per 'unless'.

        when A then B unless C1 then D1 unless C2
            -> <r>_main     when A and not C1 and not C2  then B
               <r>_unless1  when A and C1                 then D1

        Every defeater removes its case from the main rule. Only a defeater with
        a 'then' adds a rule of its own: a bare 'unless C' withdraws the
        obligation over C, it does not require the opposite, so there is nothing
        to check there. A rule with no defeater keeps its name. Defeaters are
        assumed disjoint. Fills self.trace with {normalized: original}.
        """
        normalized = []
        self.trace = {}
        for rule in self.model.ruleBlock.rules:
            # Rules triggered by an NN output state that two outputs cannot occur
            # together; argmax picks exactly one, so there is nothing to check.
            if rule.trigger.event.name in self.outputs:
                continue

            defeaters = list(rule.response.defeater or [])
            if not defeaters:
                normalized.append(
                    NormalizedRule(rule.name, rule.name, rule.condition, rule.response.occ)
                )
                continue

            condition = rule.condition
            for defeater in defeaters:
                condition = _and(condition, _not(defeater.expr))
            normalized.append(
                NormalizedRule(f"{rule.name}_main", rule.name, condition, rule.response.occ)
            )
            for i, defeater in enumerate(defeaters, start=1):
                if defeater.response is None:
                    continue  # obligation withdrawn, nothing to verify
                normalized.append(
                    NormalizedRule(f"{rule.name}_unless{i}", rule.name,
                                   _and(rule.condition, defeater.expr),
                                   defeater.response.occ)
                )

        for nrule in normalized:
            self.trace[nrule.name] = nrule.origin
        if len(self.trace) != len(normalized):
            raise ValueError("normalized rule name collision; rename the clashing rule")
        return normalized

    # ---------------- code generation ----------------

    def generate(self):
        normalized = self.normalize_rules()
        output = ["from maraboupy import Marabou, MarabouCore, MarabouUtils"]
        output.append(f"\n# normalized rule -> original rule\nTRACE = {self.trace!r}")
        output.append("\ndef check_property(onnx_path):")
        output.append("\n    results = {}")
        for nrule in normalized:
            output.append(self._generate_rule_check(nrule))
        output.append("\n    return results")
        return "\n".join(output)

    def _generate_rule_check(self, nrule):
        target = nrule.occ.event.event
        demanded = f"not {target.name}" if nrule.occ.neg else target.name
        code = [
            f"\n    # --- {nrule.name} (from {nrule.origin}): then {demanded} ---",
            f"    print('Checking {nrule.name}...')",
            f"    net = Marabou.read_onnx(onnx_path)",
            self._apply_condition(nrule.condition),
            self._apply_output_check(nrule.occ),
            self._record_result(nrule.name),
        ]
        return "\n".join(code)

    def _record_result(self, rule_name):
        """Store (verified, counterexample, exit_code). Only 'unsat' proves a rule.

        The counterexample keeps the inputs and outputs, looked up by the variable
        numbers the network reports rather than by position. solve() returns a
        value for EVERY variable in the query -- for a 25-256-256-5 net that is
        ~1054 of them, with the hidden layers numbered between the inputs and the
        outputs -- so a positional slice would take hidden neurons and miss the
        outputs. Marabou returns no assignment on unsat, hence the membership test.
        """
        return (
            f"    cex_vars = ([v.item() for v in net.inputVars[0][0].flatten()]\n"
            f"                + [v.item() for v in net.outputVars[0][0].flatten()])\n"
            f"    results[\"{rule_name}\"] = (\n"
            f"        result[0] == 'unsat',\n"
            f"        {{v: result[1][v] for v in cex_vars if v in result[1]}},\n"
            f"        result[0],\n"
            f"    )"
        )

    # ---------------- conditions ----------------

    def _term(self, term):
        """A comparison operand -> ('measure', name) or ('const', number)."""
        ref = getattr(term, 'ID', None)
        if ref is not None:
            if 'Measure' in ref.__class__.__name__:
                return ('measure', ref.name)
            return ('const', self._resolve_value(ref.name))  # Constant reference
        val = getattr(term, 'value', None)
        if val is None:
            raise ValueError("uninterpretable operand in condition")
        if val.__class__.__name__ == 'Value':  # literal INT, or "-> constant"
            return ('const', val.value if val.value is not None
                    else val.constant.value.value)
        return ('const', self._resolve_value(val.name))  # ScaleParam label

    def _literal(self, node, negate):
        """A comparison, or a bare boolean measure, -> (name, op, value)."""
        if node.__class__.__name__ == 'BoolTerminal':  # {m} means m == 1
            return (node.ID.name, '=', 0 if negate else 1)
        lhs, rhs = self._term(node.lhs), self._term(node.rhs)
        if lhs[0] != 'measure':
            raise ValueError("left side of a comparison must be a {measure}")
        if rhs[0] == 'measure':
            raise ValueError(
                f"comparing two measures ({lhs[1]} vs {rhs[1]}) is not supported"
            )
        name, val = lhs[1], rhs[1]
        op = NEGATED_OP[node.op] if negate else node.op
        # Marabou has only closed bounds, so '>' would be emitted as '>=' and admit
        # v itself. On an integer-valued measure v is a value the measure really
        # takes; the step of 1 makes the strict comparison exact instead.
        if op in ('>', '<') and self._is_ordinal(name):
            return (name, '>=', val + 1) if op == '>' else (name, '<=', val - 1)
        return (name, op, val)

    def _is_ordinal(self, name):
        """True when the measure's values really are the integers 0, 1, 2, ...

        'scale' covers two things, told apart by whether the measure declares
        LB/UB constants:
          enum   -- no LB/UB. The labels already fix the range 0..n-1 and each
                    resolves to its index (robotCommand -> pullOverHead = 2).
          bands  -- LB/UB declared. The measure is a real quantity and each label
                    carries its own LB threshold (glucose -> normalGlucose = 70).
        Only the enum has an integer step, so only it can encode '<>', '<' and '>'
        exactly.
        """
        kind = self.measure_kinds[name]
        if kind == 'boolean':
            return True
        if kind != 'scale':
            return False
        declared = f"{name}LB" in self.constants and f"{name}UB" in self.constants
        labelled = any(f"{label}LB" in self.constants for label in self.scales[name])
        if declared != labelled:
            raise ValueError(
                f"scale measure '{name}' is inconsistent: it "
                f"{'declares' if declared else 'omits'} {name}LB/{name}UB but its "
                f"labels {'carry no' if declared else 'carry'} LB constants. An enum "
                f"declares neither; a real quantity with named bands declares both."
            )
        return not declared

    def _not_equal_branches(self, name, val):
        """'m <> v' -> the branches that fall inside m's range.

        Only integer-valued measures have an exact encoding: the step of 1 skips
        no value, so "<= v-1 or >= v+1" is equivalent to "!= v". At an endpoint
        one branch leaves the range and drops, giving a plain bound.
        """
        if not self._is_ordinal(name):
            raise ValueError(
                f"'{name} <> {val}': {name} is a real quantity with named bands "
                f"(range {self._measure_range(name)}), so '<>' has no exact encoding. "
                f"Marabou has only closed inequalities and cannot exclude a single "
                f"point, and any band around {val} would drop legitimate values. "
                f"Write '< {val} or > {val}' if that is what you mean."
            )
        self._check_threshold(name, val)
        lb, ub = self._measure_range(name)
        branches = []
        if val - 1 >= lb:
            branches.append((name, '<=', val - 1))
        if val + 1 <= ub:
            branches.append((name, '>=', val + 1))
        if not branches:
            raise ValueError(
                f"'{name} <> {val}': {val} is the only value in range [{lb}, {ub}], "
                f"so the condition can never hold"
            )
        return branches

    def _and_parts(self, node, negate=False):
        """Condition -> [('lit', literal) | ('or', disjuncts)], all conjoined.

        A top-level 'and' costs nothing: it just yields more parts. Each literal
        becomes a bound, intersected with the others in _emit_literal; each 'or'
        becomes one disjunction, which Marabou conjoins with the rest.
        """
        if node is None:
            return []  # no condition: no extra constraints
        if node.__class__.__name__ == 'Negation':
            return self._and_parts(node.expr, not negate)
        op = getattr(node, 'op', None)
        if op in ('and', 'or'):
            effective = op if not negate else ('or' if op == 'and' else 'and')
            if effective == 'and':
                return (self._and_parts(node.lhs, negate)
                        + self._and_parts(node.rhs, negate))
            return [('or', self._disjuncts(node, negate))]
        name, lit_op, val = self._literal(node, negate)
        if lit_op == '<>':
            branches = self._not_equal_branches(name, val)
            if len(branches) == 1:
                return [('lit', branches[0])]
            return [('or', [[b] for b in branches])]
        return [('lit', (name, lit_op, val))]

    def _disjuncts(self, node, negate=False):
        """An 'or' subtree -> [[literal, ...], ...]: disjuncts, each a conjunction.

        A Marabou disjunct is a flat list of equations, so an 'and' met inside an
        'or' has to be distributed. That product stays inside this subtree.
        """
        if node.__class__.__name__ == 'Negation':
            return self._disjuncts(node.expr, not negate)
        op = getattr(node, 'op', None)
        if op in ('and', 'or'):
            effective = op if not negate else ('or' if op == 'and' else 'and')
            left = self._disjuncts(node.lhs, negate)
            right = self._disjuncts(node.rhs, negate)
            if effective == 'or':
                return left + right
            return [l + r for l in left for r in right]
        name, lit_op, val = self._literal(node, negate)
        if lit_op == '<>':
            return [[b] for b in self._not_equal_branches(name, val)]
        return [[(name, lit_op, val)]]

    def _measure_range(self, name):
        """(lower, upper) for an input: the LB/UB constants, else from its type."""
        if f"{name}LB" in self.constants and f"{name}UB" in self.constants:
            return (self.constants[f"{name}LB"].value, self.constants[f"{name}UB"].value)
        kind = self.measure_kinds[name]
        if kind == 'boolean':
            return (0, 1)
        if kind == 'scale':
            return (0, len(self.scales[name]) - 1)  # scale labels are ordinals
        raise ValueError(
            f"numeric measure '{name}' needs both {name}LB and {name}UB constants"
        )

    def _check_threshold(self, name, val):
        lb, ub = self._measure_range(name)
        if not (lb <= val <= ub):
            raise ValueError(f"threshold {val} for '{name}' is outside its range [{lb}, {ub}]")

    def _emit_full_ranges(self):
        """Every input must be bounded, so start each query from the declared ranges."""
        lines = []
        for name, idx in self.measures.items():
            lb, ub = self._measure_range(name)
            lines.append(f"    net.setLowerBound({idx}, {lb})")
            lines.append(f"    net.setUpperBound({idx}, {ub})")
        return lines

    def _emit_literal(self, lit, bounds):
        """One comparison -> a bound, emitted only where it narrows the current one.

        Marabou keeps one lower and one upper bound per variable, and
        setLowerBound / setUpperBound replace the stored value instead of
        intersecting with it. A comparison looser than a bound already set would
        widen the box and undo that bound, so it is skipped: conjoined, the two
        reduce to the tighter one. 'bounds' maps each input to the [lower, upper]
        currently in force and is updated in place.
        """
        name, op, val = lit
        if op not in ('=', '<', '<=', '>', '>='):
            raise ValueError(f"unsupported operator '{op}'")
        idx = self.measures[name]
        self._check_threshold(name, val)
        bound = bounds[name]
        calls = []
        if op in ('=', '>', '>=') and val > bound[0]:
            bound[0] = val
            calls.append(f"net.setLowerBound({idx}, {val})")
        if op in ('=', '<', '<=') and val < bound[1]:
            bound[1] = val
            calls.append(f"net.setUpperBound({idx}, {val})")
        if not calls:
            return [f"    # {name} {op} {val}: skipped, bounds already "
                    f"[{bound[0]}, {bound[1]}]"]
        return ([f"    {calls[0]}  # {name} {op} {val}"]
                + [f"    {call}" for call in calls[1:]])

    def _emit_or(self, disjuncts, tag):
        """One 'or' -> one Marabou disjunction.

        Equations in the same disjunct are AND-ed by the solver, so a conjunctive
        disjunct needs no special handling. 'tag' keeps the generated locals
        distinct when a condition holds more than one 'or'.
        """
        lines = [f"    disjuncts{tag} = []"]
        for n, conjunction in enumerate(disjuncts):
            lines.append(f"    d{tag}_{n} = []")
            for name, op, val in conjunction:
                self._check_threshold(name, val)
                rel = 'EQ' if op == '=' else ('LE' if op in ('<', '<=') else 'GE')
                lines += [
                    f"    e = MarabouUtils.Equation(MarabouCore.Equation.{rel})"
                    f"  # {name} {op} {val}",
                    f"    e.addAddend(1, net.inputVars[0][0][{self.measures[name]}])",
                    f"    e.setScalar({val})",
                    f"    d{tag}_{n}.append(e)",
                ]
            lines.append(f"    disjuncts{tag}.append(d{tag}_{n})")
        lines.append(f"    net.addDisjunctionConstraint(disjuncts{tag})")
        return lines

    def _apply_condition(self, condition):
        """Every input bounded by its range, then one emit per conjoined part.

        The bounds in force start at the declared ranges and only ever narrow.
        """
        bounds = {name: list(self._measure_range(name)) for name in self.measures}
        lines = self._emit_full_ranges()
        for tag, (kind, payload) in enumerate(self._and_parts(condition)):
            lines += (self._emit_literal(payload, bounds) if kind == 'lit'
                      else self._emit_or(payload, tag))
        return "\n".join(lines)

    def _apply_output_check(self, occ):
        """occ.neg decides the property: 'then B' vs 'then not B'."""
        target_action = occ.event.event
        target_idx = self.outputs[target_action.name]
        mode = "Negate" if occ.neg else "Normal"
        lines = []

        independent = self.config["output_semantics"] == "independent"
        threshold = self.config["decision_threshold"]
        # The threshold is itself a disjunct, so one output still needs the list.
        use_disjunction = mode == "Normal" and (len(self.outputs) > 1 or independent)

        # With one output there is nothing to compare against, so the query would
        # carry no output constraint at all and be trivially satisfiable.
        # "independent" is exempt: the output carries its own threshold.
        if (not independent and len(self.outputs) == 1
                and self.config["target_output_ub"] is None):
            raise ValueError(
                f"Spec declares a single output ({target_action.name}), so the property "
                "needs a decision threshold: pass config={'target_output_ub': <value>}."
            )

        if mode == "Normal":
            lines.append(f"    # Verify output is {target_action.name} (Index {target_idx})")
            if independent:
                lines.append(f"    # Assert Negation: {target_action.name} <= {threshold} "
                             f"OR exists other output >= {target_action.name}")
            else:
                lines.append(f"    # Assert Negation: exists other output >= {target_action.name}")
        else:
            lines.append(f"    # Verify output is NOT {target_action.name} (Index {target_idx})")
            if independent:
                lines.append(f"    # Assert Negation:  {target_action.name} >= {threshold} "
                             f"AND >= all other outputs")
            else:
                lines.append(f"    # Assert Negation:  {target_action.name} >= all other outputs")

        lines.append(f"    outputVars = net.outputVars[0][0]")
        if use_disjunction:
            lines.append(f"    equations = []")

        for name, idx in self.outputs.items():
            if idx == target_idx:
                # Optional: the comparisons below are the property on their own.
                if self.config["target_output_ub"] is not None:
                    lines.append(f"    # optional bound on {name}")
                    lines.append(f"    net.setUpperBound(outputVars[{idx}], "
                                 f"{self.config['target_output_ub']})")
            else:
                if mode == "Normal":
                    lines.append(f"    # {name} >= {target_action.name}")
                    # Marabou 2.0 reads EquationType/addendList/scalar off this object,
                    # so it must be a MarabouUtils.Equation with an explicit scalar.
                    lines.append(f"    eq = MarabouUtils.Equation(MarabouCore.Equation.GE)")
                    lines.append(f"    eq.addAddend(1, outputVars[{idx}])")
                    lines.append(f"    eq.addAddend(-1, outputVars[{target_idx}])")
                    lines.append(f"    eq.setScalar(0)")
                    lines.append(f"    equations.append([eq])")
                else:
                    lines.append(f"    # {name} <= {target_action.name}")
                    lines.append(f"    eq = MarabouUtils.Equation(MarabouCore.Equation.GE)")
                    lines.append(f"    eq.addAddend(1, outputVars[{target_idx}])")
                    lines.append(f"    eq.addAddend(-1, outputVars[{idx}])")
                    lines.append(f"    eq.setScalar(0)")
                    lines.append(f"    net.addEquation(eq)")

        # Clearing the threshold is one more way the property can hold, so its
        # negation gains a disjunct under "Normal" and a conjunct under "Negate".
        if independent:
            if mode == "Normal":
                lines.append(f"    # {target_action.name} not permitted")
                lines.append(f"    eq = MarabouUtils.Equation(MarabouCore.Equation.LE)")
                lines.append(f"    eq.addAddend(1, outputVars[{target_idx}])")
                lines.append(f"    eq.setScalar({threshold})")
                lines.append(f"    equations.append([eq])")
            else:
                lines.append(f"    # {target_action.name} permitted")
                lines.append(f"    eq = MarabouUtils.Equation(MarabouCore.Equation.GE)")
                lines.append(f"    eq.addAddend(1, outputVars[{target_idx}])")
                lines.append(f"    eq.setScalar({threshold})")
                lines.append(f"    net.addEquation(eq)")

        if use_disjunction:
            lines.append(f"    net.addDisjunctionConstraint(equations)")
        lines.append(f"    result = net.solve()")
        lines.append(f"    if result[0] == 'sat': print('  -> VIOLATED: counter-example found')")
        lines.append(f"    elif result[0] == 'unsat': print('  -> VERIFIED')")
        lines.append(f"    else: print('  -> UNKNOWN: solver returned ' + str(result[0]))")
        return "\n".join(lines)
