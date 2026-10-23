# ==========================================
# 2. THE GENERATOR LOGIC
# ==========================================

import itertools

# Generation options. Override per run via MarabouGenerator(model, config={...}).
#   target_output_ub: extra upper bound on the output under test, or None for no
#       bound. Only narrows the search, so it can hide counterexamples; 0.0
#       reproduces the original behaviour.
#   output_semantics: "exclusive" for an argmax/softmax head, where an action may
#       be taken when no other output is above it, and a rule must hold for every
#       action that may be taken; "independent" for per-output sigmoids, where an
#       action must also clear decision_threshold. Under "independent" several
#       actions may be taken at once, or none. strict_marabou_verif decides how a
#       tie is read.
#   decision_threshold: the threshold under "independent". Export the head
#       without its sigmoid and leave this at 0.0: sigmoid(z) >= 0.5 <=> z >= 0.
#   strict_input_comparisons: how '>' and '<' in a rule's condition are encoded
#       on a real quantity, which a closed bound cannot express. "epsilon" moves
#       the bound inside the threshold, so a witness really satisfies it; "closed"
#       leaves it on the threshold and admits the boundary, as every run before
#       this option did.
#   strict_marabou_verif: how the output comparisons are encoded: one output
#       against another, and against decision_threshold. The spec writes none of
#       them, so this decides whether a tie at the top counts against a rule.
#       "epsilon" leaves the tie out of the search, so a counterexample really
#       takes the action it is reported with, and a rule is unproven only within
#       one step of a tie; "closed" keeps the tie in, as every run before this
#       option did, and a counterexample may sit on the tie.
#   splitting_strategy: how Marabou branches inside a query. "largest-interval"
#       splits the input box, which is fast on networks with many ReLUs; None
#       leaves Marabou's own choice. It changes only the search, not what is checked.
#   snc_workers: Marabou's parallel split-and-conquer mode with this many
#       workers; 0 turns it off. It changes only the search, not what is checked.
DEFAULT_CONFIG = {
    "target_output_ub": None,
    "output_semantics": "exclusive",
    "decision_threshold": 0.0,
    "strict_input_comparisons": "epsilon",
    "strict_marabou_verif": "epsilon",
    "splitting_strategy": "largest-interval",
    "snc_workers": 8,
}

# Operator seen under a 'not'.
NEGATED_OP = {'>': '<=', '<': '>=', '>=': '<', '<=': '>', '=': '<>', '<>': '='}

# Marabou accepts a value that misses a bound by up to ADDITIVE + MULTIPLICATIVE *
# |bound| (Tableau::computeBasicStatus), and reports sat on that same relaxation
# (Engine::solve). A strict comparison survives only if it clears this.
MARABOU_BOUND_SLACK = (1e-7, 1e-10)


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


def _or(lhs, rhs):
    return _SleecNode('BoolBinaryOp', lhs=lhs, op='or', rhs=rhs)


def _number(value):
    """A textX Value -> its number, following '-> constant' references.

    Value is 'INT | -> constant'. textX fills the INT field with 0 when the
    reference form is used, so the reference is checked first. A loop such as
    'constant a = -> a' never reaches a number and is refused.
    """
    seen = []
    while value.constant is not None:
        if value.constant.name in seen:
            raise ValueError("constant reference loop: "
                             + " -> ".join(seen + [value.constant.name]))
        seen.append(value.constant.name)
        value = value.constant.value
    return value.value


# A measure relation of the relation block -> the condition it states.
MEASURE_RELATIONS = {
    'imply':           lambda a, b: _or(_not(a), b),
    'iff':             lambda a, b: _or(_and(a, b), _and(_not(a), _not(b))),
    'mutualExclusive': lambda a, b: _not(_and(a, b)),
    'opposite':        lambda a, b: _or(_and(a, _not(b)), _and(_not(a), b)),
}


class NormalizedRule:
    """A rule with no defeaters: when <condition> then <occs[0]> else <occs[1]> ...

    occs holds the actions the rule allows, one per 'else' branch; a rule with
    no 'else' has a single one.
    """

    def __init__(self, name, origin, condition, occs):
        self.name = name
        self.origin = origin
        self.condition = condition
        self.occs = occs


class MarabouGenerator:
    def __init__(self, model, config=None):
        self.model = model
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.constants = {}  # constant name -> its number
        self.scales = {}
        self.scale_kinds = {}  # scale measure name -> 'enum' or 'bands'
        self.measures = {}  # Map measure name to index
        self.measure_kinds = {}  # Map measure name to boolean / scale / numeric
        self.outputs = {}  # Map output event name to index
        self.trace = {}  # normalized rule name -> original rule name
        self.rule_text = {}  # original rule name -> its text in the spec
        self.inputs = {}  # normalized rule name -> inputs its condition uses
        self.relations = None  # relation block's measure relations, as one condition

        unknown = set(self.config) - set(DEFAULT_CONFIG)
        if unknown:
            raise ValueError(f"Unknown config option(s): {sorted(unknown)}")

        # Populate symbol tables
        self._scan_definitions()
        self._scan_relations()

    def _scan_definitions(self):
        # 1. Load Constants
        for item in self.model.definitions:
            if item.__class__.__name__ == 'Constant':
                self.constants[item.name] = _number(item.value)

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

        #    A scale measure is either a pure enum (no constants: each label is its
        #    position) or named bands of a real quantity (its own LB/UB and an LB
        #    for every label). Its kind is decided here, once; anything in between
        #    would be guessed, so it is refused.
        for name, labels in self.scales.items():
            needed = [f"{name}LB", f"{name}UB"] + [f"{label}LB" for label in labels]
            present = [c for c in needed if c in self.constants]
            if not present:
                self.scale_kinds[name] = 'enum'
            elif len(present) == len(needed):
                self.scale_kinds[name] = 'bands'
            else:
                missing = [c for c in needed if c not in self.constants]
                raise ValueError(
                    f"scale measure '{name}' is neither an enum (no constants) nor named "
                    f"bands (all of {', '.join(needed)}): missing {', '.join(missing)}"
                )

        #    A boolean takes only 0 or 1, so it declares no range of its own. A
        #    fixed value is a fact about the data and belongs in a measure invariant.
        for name, kind in self.measure_kinds.items():
            if kind == 'boolean' and (f"{name}LB" in self.constants or f"{name}UB" in self.constants):
                raise ValueError(
                    f"boolean measure '{name}' declares {name}LB/{name}UB, but a boolean's "
                    f"range is {{0, 1}}; state a fixed value as 'measure invariant' instead"
                )

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

    def _scan_relations(self):
        """The relation block's dataset constraints, conjoined into self.relations.

        Only measure relations constrain a single network evaluation: 'measure
        invariant E' states E, and 'measure <rel> A B' states MEASURE_RELATIONS[rel].
        The others (event relations, causation, includes, forbid, the timed ones)
        relate events over time, which a single evaluation does not have, so they
        are skipped.
        """
        if self.model.relBlock is None:
            return
        for relation in self.model.relBlock.relations:
            kind = relation.__class__.__name__
            if kind == 'MeasureInv':
                self.relations = _and(self.relations, relation.expr)
            elif kind == 'MeasureRel':
                self.relations = _and(self.relations,
                                      MEASURE_RELATIONS[relation.rel](relation.lhs, relation.rhs))

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
        COMPLIANT without it ever having been tested.
        """
        for measure, labels in self.scales.items():
            if val_name in labels:
                if self.scale_kinds[measure] == 'enum':
                    return labels.index(val_name)  # pure enum: the label IS the value
                return self.constants[f"{val_name}{bound}"]

        if val_name in self.constants:
            # The old generator could only read '<name>LB', so specs declared a
            # dummy '<name> = 0' beside the real '<name>LB'. Both now resolve, so
            # the pair is ambiguous; collapse it rather than pick one silently.
            if f"{val_name}{bound}" in self.constants:
                raise ValueError(
                    f"'{val_name}' is ambiguous: both '{val_name}' "
                    f"(= {self.constants[val_name]}) and '{val_name}{bound}' "
                    f"(= {self.constants[f'{val_name}{bound}']}) are declared. "
                    f"Keep one constant holding the real value."
                )
            return self.constants[val_name]

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
        assumed disjoint. A 'then' may be an 'else' chain: it stays one
        obligation, and its sub-rule carries it whole (see _branches). Fills
        self.trace with {normalized: original}, and self.rule_text with
        {original: its text}, skipped rules included.
        """
        normalized = []
        self.trace = {}
        self.rule_text = {}
        source = self.model._tx_parser.input
        for rule in self.model.ruleBlock.rules:
            if rule.name in self.rule_text:
                raise ValueError(f"two rules are named '{rule.name}'; rule names must be unique")
            self.rule_text[rule.name] = source[rule._tx_position:rule._tx_position_end]
            # Rules triggered by an NN output state that two outputs cannot occur
            # together; argmax picks exactly one, so there is nothing to check.
            # They get no sub-rule, which the rule table reports as IGNORED.
            if rule.trigger.event.name in self.outputs:
                continue

            occs = self._branches(rule.name, rule.response)
            defeaters = list(rule.response.defeater or [])
            if not defeaters:
                normalized.append(
                    NormalizedRule(rule.name, rule.name, rule.condition, occs)
                )
                continue

            condition = rule.condition
            for defeater in defeaters:
                condition = _and(condition, _not(defeater.expr))
            normalized.append(
                NormalizedRule(f"{rule.name}_main", rule.name, condition, occs)
            )
            for i, defeater in enumerate(defeaters, start=1):
                if defeater.response is None:
                    continue  # obligation withdrawn, nothing to verify
                normalized.append(
                    NormalizedRule(f"{rule.name}_unless{i}", rule.name,
                                   _and(rule.condition, defeater.expr),
                                   self._branches(rule.name, defeater.response))
                )

        for nrule in normalized:
            self.trace[nrule.name] = nrule.origin
        if len(self.trace) != len(normalized):
            raise ValueError("normalized rule name collision; rename the clashing rule")
        return normalized

    def _branches(self, rule_name, response):
        """'then A else B else C' -> [A, B, C]: the response holds if one of them does.

        Each 'else' holds a response of its own, which may have its own 'else',
        so the chain is followed to its end. A response with no 'else' gives
        [A], and 'then not A' keeps its 'not'.

        Only an action at the top may be taken (under "independent", one that
        also clears decision_threshold), which rules some chains out: a 'not'
        in a chain collapses it ('A else not B' is just 'not B') or makes it
        hold everywhere except where A and B tie on top ('not A else not B'),
        and under "exclusive" a chain naming every output can never fail.
        Each says something other than what it seems to, so it is refused, as
        is a repeated action. Under "independent" a chain naming every output
        still fails when no output clears the threshold, so it is kept. An
        'unless' inside an 'else' branch is refused too; it can always be
        written on the whole rule.
        """
        occs = [response.occ]
        nd = response.nd
        while nd is not None:
            if nd.response.defeater:
                raise ValueError(
                    f"rule '{rule_name}': 'unless' inside an 'else' branch is not "
                    f"supported. Write it on the whole rule instead: 'then A else "
                    f"{{B unless C}}' is 'then A else B unless C', and 'then A else "
                    f"{{B unless C then D}}' is 'then A else B unless C then A else D'.")
            occs.append(nd.response.occ)
            nd = nd.response.nd
        if len(occs) == 1:
            return occs

        names = [occ.event.event.name for occ in occs]
        chain = " else ".join(names)
        if any(occ.neg for occ in occs):
            raise ValueError(
                f"rule '{rule_name}': 'not' inside an 'else' chain. Only an action at "
                f"the top may be taken, so 'then A else not B' means 'then not B', and "
                f"'then not A else not B' holds everywhere except where A and B tie on "
                f"top; write the rule without 'else'."
            )
        if len(set(names)) < len(names):
            raise ValueError(f"rule '{rule_name}': 'then {chain}' names an action twice")
        if self.config["output_semantics"] == "exclusive" and set(names) == set(self.outputs):
            raise ValueError(f"rule '{rule_name}': 'then {chain}' allows every output, "
                             f"so it can never fail")
        return occs

    # ---------------- code generation ----------------

    def generate(self):
        """The verification module. Its check_property(onnx_path) returns

            {"results": {normalized rule: (verified, counterexample, exit_code)},
             "trace":   {normalized rule: original rule},
             "rules":   {original rule: {"text", "subrules", "verdict"}},
             "inputs":  {normalized rule: [{"name", "index"}]}}

        with "rules" in spec order; rule_verdicts() documents the verdicts.
        """
        normalized = self.normalize_rules()
        self.inputs = {}
        checks = [self._generate_rule_check(nrule) for nrule in normalized]  # fills self.inputs
        output = ["from maraboupy import Marabou, MarabouCore, MarabouUtils"]
        output.append(f"\n# normalized rule -> original rule\nTRACE = {self.trace!r}")
        output.append(f"\n# original rule -> its text in the spec\nRULES = {self.rule_text!r}")
        output.append(f"\n# normalized rule -> inputs its condition uses\nINPUTS = {self.inputs!r}")
        output.append("\n" + self._emit_rule_verdicts())
        output.append("\n\ndef check_property(onnx_path):")
        output.append("\n    results = {}")
        output += checks
        output.append('\n    return {"results": results, "trace": TRACE, '
                      '"rules": rule_verdicts(results), "inputs": INPUTS}')
        return "\n".join(output)

    def _emit_rule_verdicts(self):
        """The generated module's rule_verdicts(): one verdict per original rule,
        from the results of its normalized rules. The generated module has no
        access to the parsed spec, so it reads the RULES and TRACE constants
        generated beside it."""
        lines = [
            "def rule_verdicts(results):",
            '    """One verdict per original rule, from the results of its normalized rules.',
            "",
            "    COMPLIANT   every sub-rule is unsat",
            "    VIOLATED    at least one sub-rule is sat",
            "    UNVERIFIED  none is sat, and at least one ended otherwise (timeout, error)",
            "    IGNORED     no sub-rule: rules triggered by an NN output are not checked",
            '    """',
            "    rules = {}",
            "    for name, text in RULES.items():",
            "        subrules = [n for n, origin in TRACE.items() if origin == name]",
            "        codes = [results[n][2] for n in subrules]",
            "        if not subrules:",
            '            verdict = "IGNORED"',
            '        elif "sat" in codes:',
            '            verdict = "VIOLATED"',
            '        elif all(code == "unsat" for code in codes):',
            '            verdict = "COMPLIANT"',
            "        else:",
            '            verdict = "UNVERIFIED"',
            '        rules[name] = {"text": text, "subrules": subrules, "verdict": verdict}',
            "    return rules",
        ]
        return "\n".join(lines)

    def _generate_rule_check(self, nrule):
        demanded = " else ".join(f"not {occ.event.event.name}" if occ.neg
                                 else occ.event.event.name for occ in nrule.occs)
        condition_code, self.inputs[nrule.name] = self._apply_condition(nrule.condition,
                                                                        self.relations)
        # One action keeps the original check; an 'else' chain has its own.
        if len(nrule.occs) == 1:
            output_code = self._apply_output_check(nrule.occs[0])
        elif self.config["output_semantics"] == "independent":
            output_code = self._apply_independent_chain_check(nrule.occs)
        else:
            output_code = self._apply_chain_check(nrule.occs)
        code = [
            f"\n    # --- {nrule.name} (from {nrule.origin}): then {demanded} ---",
            f"    print('Checking {nrule.name}...')",
            f"    net = Marabou.read_onnx(onnx_path)",
            condition_code,
            output_code,
            self._emit_solve(),
            self._record_result(nrule.name),
        ]
        return "\n".join(code)

    def _emit_solve(self):
        """Solve the query and print its outcome; _record_result stores it.

        Only the options that are switched on are passed; with none of them the
        call is a plain net.solve().
        """
        options = []
        if self.config["snc_workers"]:
            options.append(f"snc=True, numWorkers={self.config['snc_workers']}")
        if self.config["splitting_strategy"] is not None:
            options.append(f"splittingStrategy={self.config['splitting_strategy']!r}")
        if options:
            solve = [f"    options = Marabou.createOptions({', '.join(options)})",
                     f"    result = net.solve(options=options)"]
        else:
            solve = [f"    result = net.solve()"]
        return "\n".join(solve + [
            f"    if result[0] == 'sat': print('  -> VIOLATED: counter-example found')",
            f"    elif result[0] == 'unsat': print('  -> COMPLIANT')",
            f"    else: print('  -> UNVERIFIED: solver returned ' + str(result[0]))",
        ])

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
            return ('const', _number(val))
        return ('const', self._resolve_value(val.name))  # ScaleParam label

    def _literal(self, node, negate):
        """A comparison, or a bare boolean measure, -> (name, op, value)."""
        if node.__class__.__name__ == 'BoolTerminal':  # {m} means m == 1
            if node.ID is None:
                raise ValueError(f"'{node.value}' is not supported in conditions; remove it")
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

        'scale' covers two things, decided once per measure by _scan_definitions
        and kept in self.scale_kinds:
          enum   -- no constants. The labels already fix the range 0..n-1 and each
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
        return self.scale_kinds[name] == 'enum'

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
        """(lower, upper) for an input: an enum's positions, else its LB/UB
        constants, else a boolean's {0, 1}."""
        kind = self.measure_kinds[name]
        if kind == 'scale' and self.scale_kinds[name] == 'enum':
            return (0, len(self.scales[name]) - 1)  # an enum's values are its positions
        if f"{name}LB" in self.constants and f"{name}UB" in self.constants:
            return (self.constants[f"{name}LB"], self.constants[f"{name}UB"])
        if kind == 'boolean':
            return (0, 1)
        raise ValueError(
            f"numeric measure '{name}' needs both {name}LB and {name}UB constants"
        )

    def _strict_step(self, val):
        """How far inside the threshold a strict bound has to sit.

        Ten times each of Marabou's two slack terms, so it clears them at any
        magnitude. Not configurable: a value set by hand that fell below the slack
        would silently restore the behaviour this is here to fix.
        """
        additive, relative = MARABOU_BOUND_SLACK
        return max(10 * additive, 10 * relative * abs(val))

    def _strict_as_closed(self, op, val):
        """A comparison as a bound Marabou can hold: op in '=', '<=', '>='.

        _literal already stepped an ordinal measure, exactly. What is still strict
        here is a real quantity, which has no exact encoding, so the bound moves
        inside the threshold by enough that a point on it still compares strictly.
        """
        if op not in ('>', '<') or self.config["strict_input_comparisons"] != "epsilon":
            return op, val
        step = self._strict_step(val)
        return ('>=', val + step) if op == '>' else ('<=', val - step)

    def _output_step(self, val=0):
        """How far an output comparison moves off the tie: 0 under "closed".

        val is what the comparison is made against: 0 between two outputs,
        decision_threshold against the threshold. The step is the input side's,
        since it has the same slack to clear.
        """
        if self.config["strict_marabou_verif"] != "epsilon":
            return 0
        return self._strict_step(val)

    def _check_threshold(self, name, val):
        lb, ub = self._measure_range(name)
        if not (lb <= val <= ub):
            raise ValueError(f"threshold {val} for '{name}' is outside its range [{lb}, {ub}]")

    def _check_shifted(self, name, op, val, written):
        """A strict bound moved inside the range still has to land in it."""
        lb, ub = self._measure_range(name)
        if not (lb <= val <= ub):
            side = 'above' if op == '>=' else 'below'
            raise ValueError(
                f"'{written}' cannot be satisfied: {name} ranges over [{lb}, {ub}], so "
                f"no value lies strictly {side} that threshold inside its range.")

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
        self._check_threshold(name, val)        # the threshold as the rule states it
        written = f"{name} {op} {val}"
        op, val = self._strict_as_closed(op, val)
        self._check_shifted(name, op, val, written)
        # The comment carries the comparison as written, and the step where one was
        # taken, so the query says what it encodes.
        note = written if f"{val}" in written else f"{written} -> {op} {val}"
        bound = bounds[name]
        calls = []
        if op in ('=', '>', '>=') and val > bound[0]:
            bound[0] = val
            calls.append(f"net.setLowerBound({idx}, {val})")
        if op in ('=', '<', '<=') and val < bound[1]:
            bound[1] = val
            calls.append(f"net.setUpperBound({idx}, {val})")
        if not calls:
            return [f"    # {note}: skipped, bounds already "
                    f"[{bound[0]}, {bound[1]}]"]
        return ([f"    {calls[0]}  # {note}"]
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
                written = f"{name} {op} {val}"
                op, val = self._strict_as_closed(op, val)
                self._check_shifted(name, op, val, written)
                note = written if f"{val}" in written else f"{written} -> {op} {val}"
                rel = 'EQ' if op == '=' else ('LE' if op in ('<', '<=') else 'GE')
                lines += [
                    f"    e = MarabouUtils.Equation(MarabouCore.Equation.{rel})"
                    f"  # {note}",
                    f"    e.addAddend(1, net.inputVars[0][0][{self.measures[name]}])",
                    f"    e.setScalar({val})",
                    f"    d{tag}_{n}.append(e)",
                ]
            lines.append(f"    disjuncts{tag}.append(d{tag}_{n})")
        lines.append(f"    net.addDisjunctionConstraint(disjuncts{tag})")
        return lines

    def _emit_part(self, kind, payload, bounds, tag):
        """One conjoined part of _and_parts(): a comparison or an 'or'."""
        if kind == 'lit':
            return self._emit_literal(payload, bounds)
        return self._emit_or(payload, tag)

    def _apply_condition(self, condition, relations=None):
        """Every input bounded by its range, then one emit per conjoined part.

        The rule's condition comes first, then the relation block's constraints,
        as in the spec. Both share the bounds in force, which start at the
        declared ranges and only ever narrow, so a constraint can never undo a
        bound of the rule. Last, every boolean input is split into 'm = 0 or
        m = 1': Marabou has only real variables, and a boolean takes no value in
        between. Returns the code, and the inputs the rule's condition compares,
        inside an 'or' or not, as [{"name", "index"}] in input order. The relation
        block's inputs are left out: its constraints are the same for every rule.
        """
        bounds = {name: list(self._measure_range(name)) for name in self.measures}
        lines = self._emit_full_ranges()
        tags = itertools.count()  # one per part, so every 'or' gets its own locals
        used = set()
        for kind, payload in self._and_parts(condition):
            lines += self._emit_part(kind, payload, bounds, next(tags))
            if kind == 'lit':
                used.add(payload[0])
            else:
                used.update(name for conjunction in payload for name, _, _ in conjunction)
        relation_parts = self._and_parts(relations)
        if relation_parts:
            lines.append("    # relation block")
        for kind, payload in relation_parts:
            lines += self._emit_part(kind, payload, bounds, next(tags))
        booleans = [name for name, kind in self.measure_kinds.items() if kind == 'boolean']
        if booleans:
            lines.append("    # boolean inputs take only 0 or 1")
        for name in booleans:
            lines += self._emit_or([[(name, '=', 0)], [(name, '=', 1)]], next(tags))
        inputs = [{"name": name, "index": idx}
                  for name, idx in self.measures.items() if name in used]
        return "\n".join(lines), inputs

    def _apply_output_check(self, occ):
        """occ.neg decides the property: 'then B' vs 'then not B'.

        Every comparison sits _output_step off the tie. Under "epsilon" a
        counterexample then has another output strictly above B ('then B'), or B
        strictly above every other ('then not B'), and under "independent" B
        strictly on the searched side of the threshold.
        """
        target_action = occ.event.event
        target_idx = self.outputs[target_action.name]
        mode = "Negate" if occ.neg else "Normal"
        lines = []

        independent = self.config["output_semantics"] == "independent"
        threshold = self.config["decision_threshold"]
        # Two outputs are compared through their difference, set against 0; one
        # output against the threshold is set against the threshold, moved to the
        # side the negation searches: below it for "Normal", above it for "Negate".
        step = self._output_step()
        shifted_threshold = (threshold - self._output_step(threshold)
                             if mode == "Normal"
                             else threshold + self._output_step(threshold))
        plus = f" + {step}" if step else ""
        minus = f" - {step}" if step else ""
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
                lines.append(f"    # Assert Negation: {target_action.name} <= "
                             f"{shifted_threshold} "
                             f"OR exists other output >= {target_action.name}{plus}")
            else:
                lines.append(f"    # Assert Negation: exists other output >= "
                             f"{target_action.name}{plus}")
        else:
            lines.append(f"    # Verify output is NOT {target_action.name} (Index {target_idx})")
            if independent:
                lines.append(f"    # Assert Negation:  {target_action.name} >= "
                             f"{shifted_threshold} "
                             f"AND >= all other outputs{plus}")
            else:
                lines.append(f"    # Assert Negation:  {target_action.name} >= "
                             f"all other outputs{plus}")

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
                    lines.append(f"    # {name} >= {target_action.name}{plus}")
                    # Marabou 2.0 reads EquationType/addendList/scalar off this object,
                    # so it must be a MarabouUtils.Equation with an explicit scalar.
                    lines.append(f"    eq = MarabouUtils.Equation(MarabouCore.Equation.GE)")
                    lines.append(f"    eq.addAddend(1, outputVars[{idx}])")
                    lines.append(f"    eq.addAddend(-1, outputVars[{target_idx}])")
                    lines.append(f"    eq.setScalar({step})")
                    lines.append(f"    equations.append([eq])")
                else:
                    lines.append(f"    # {name} <= {target_action.name}{minus}")
                    lines.append(f"    eq = MarabouUtils.Equation(MarabouCore.Equation.GE)")
                    lines.append(f"    eq.addAddend(1, outputVars[{target_idx}])")
                    lines.append(f"    eq.addAddend(-1, outputVars[{idx}])")
                    lines.append(f"    eq.setScalar({step})")
                    lines.append(f"    net.addEquation(eq)")

        # Clearing the threshold is one more way the property can hold, so its
        # negation gains a disjunct under "Normal" and a conjunct under "Negate".
        if independent:
            if mode == "Normal":
                lines.append(f"    # {target_action.name} not permitted")
                lines.append(f"    eq = MarabouUtils.Equation(MarabouCore.Equation.LE)")
                lines.append(f"    eq.addAddend(1, outputVars[{target_idx}])")
                lines.append(f"    eq.setScalar({shifted_threshold})")
                lines.append(f"    equations.append([eq])")
            else:
                lines.append(f"    # {target_action.name} permitted")
                lines.append(f"    eq = MarabouUtils.Equation(MarabouCore.Equation.GE)")
                lines.append(f"    eq.addAddend(1, outputVars[{target_idx}])")
                lines.append(f"    eq.setScalar({shifted_threshold})")
                lines.append(f"    net.addEquation(eq)")

        if use_disjunction:
            lines.append(f"    net.addDisjunctionConstraint(equations)")
        return "\n".join(lines)

    def _apply_chain_check(self, occs):
        """'then A else B ...' holds iff each action that may be taken is in the chain.

        Its negation is "some output outside the chain is >= every action in
        it", which is exactly when an outside output may be taken; a tie
        between two actions of the chain is fine, as either may be taken. One
        disjunction, one option per output outside the chain, each option one
        equation per action in it. _branches lets only plain actions under
        "exclusive" into a chain, and never every output, so there is always at
        least one option. Under "closed" every comparison is exact: an outside
        output may be taken as soon as it reaches the chain's best. Under
        "epsilon" it has to beat each action of the chain by _output_step, so a
        tie with the chain counts for the rule.
        """
        names = [occ.event.event.name for occ in occs]
        allowed = [self.outputs[name] for name in names]
        step = self._output_step()
        each = f", each + {step}" if step else ""
        lines = [
            f"    # Verify output is {' or '.join(names)} "
            f"(Index {', '.join(map(str, allowed))})",
            f"    # Assert Negation: exists other output >= {' and '.join(names)}{each}",
            f"    outputVars = net.outputVars[0][0]",
            f"    equations = []",
        ]
        for name, idx in self.outputs.items():
            if idx in allowed:
                # Optional: the comparisons below are the property on their own.
                if self.config["target_output_ub"] is not None:
                    lines.append(f"    # optional bound on {name}")
                    lines.append(f"    net.setUpperBound(outputVars[{idx}], "
                                 f"{self.config['target_output_ub']})")
            else:
                # This output wins over the chain only if it beats every action in it.
                lines.append(f"    # {name} >= {' and '.join(names)}{each}")
                lines.append(f"    option = []")
                for target in allowed:
                    lines += [
                        f"    eq = MarabouUtils.Equation(MarabouCore.Equation.GE)",
                        f"    eq.addAddend(1, outputVars[{idx}])",
                        f"    eq.addAddend(-1, outputVars[{target}])",
                        f"    eq.setScalar({step})",
                        f"    option.append(eq)",
                    ]
                lines.append(f"    equations.append(option)")
        lines.append(f"    net.addDisjunctionConstraint(equations)")
        return "\n".join(lines)

    def _apply_independent_chain_check(self, occs):
        """'then A else B ...' under "independent": the action taken is one in the chain.

        The action taken is the output at the top, if it also clears
        decision_threshold, so an action X of the chain is not taken when
        "X <= threshold, or some output outside the chain is >= X". The
        negation of the chain is that for every X in it: one disjunction per
        action, all conjoined. A tie between two actions of the chain is fine,
        as either may be taken, so they are not compared with each other. A
        chain naming every output has no outside output: it is violated
        exactly when every output is <= threshold, i.e. no action is taken.
        Under "epsilon" both comparisons sit _output_step off the tie: X below
        the threshold, or an outside output above X, by at least that step.
        """
        threshold = self.config["decision_threshold"]
        shifted_threshold = threshold - self._output_step(threshold)
        step = self._output_step()
        plus = f" + {step}" if step else ""
        names = [occ.event.event.name for occ in occs]
        allowed = [self.outputs[name] for name in names]
        lines = [
            f"    # Verify output is {' or '.join(names)} "
            f"(Index {', '.join(map(str, allowed))})",
            f"    # Assert Negation: each of {', '.join(names)} is <= {shifted_threshold} "
            f"OR some output outside the chain >= it{plus}",
            f"    outputVars = net.outputVars[0][0]",
        ]
        for name, idx in self.outputs.items():
            # Optional: the comparisons below are the property on their own.
            if idx in allowed and self.config["target_output_ub"] is not None:
                lines.append(f"    # optional bound on {name}")
                lines.append(f"    net.setUpperBound(outputVars[{idx}], "
                             f"{self.config['target_output_ub']})")
        for name, target in zip(names, allowed):
            lines += [
                f"    # {name} <= {shifted_threshold} "
                f"OR some output outside the chain >= {name}{plus}",
                f"    equations = []",
                f"    eq = MarabouUtils.Equation(MarabouCore.Equation.LE)",
                f"    eq.addAddend(1, outputVars[{target}])",
                f"    eq.setScalar({shifted_threshold})",
                f"    equations.append([eq])",
            ]
            for idx in self.outputs.values():
                if idx not in allowed:
                    lines += [
                        f"    eq = MarabouUtils.Equation(MarabouCore.Equation.GE)",
                        f"    eq.addAddend(1, outputVars[{idx}])",
                        f"    eq.addAddend(-1, outputVars[{target}])",
                        f"    eq.setScalar({step})",
                        f"    equations.append([eq])",
                    ]
            lines.append(f"    net.addDisjunctionConstraint(equations)")
        return "\n".join(lines)
