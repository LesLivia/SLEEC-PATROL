"""Verification logic behind the web UI, with no web framework involved.

Kept apart from app.py so it can be exercised without FastAPI, and - except for
solve_module() - without maraboupy, so parsing, code generation and report building
stay testable on any machine.

The two phases are separate calls on purpose: the UI shows a parser error verbatim,
while a failure inside Marabou gets a generic message with the detail tucked away.
"""

import html
import importlib.util
import re
import sys
import uuid
from pathlib import Path

from textx import metamodel_from_str

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from verifier.sleec2marabou_parser import MarabouGenerator  # noqa: E402
from webapp import ruletext, scenes  # noqa: E402

DEFAULT_GRAMMAR = ROOT / "legos_sleec" / "sleec-grammar.txt"


def build_generator(spec_text, grammar_text=None, target_output_ub=None):
    """Parse the spec and return a MarabouGenerator for it."""
    grammar_text = grammar_text or DEFAULT_GRAMMAR.read_text(encoding="utf-8")
    model = metamodel_from_str(grammar_text).model_from_str(spec_text)
    return MarabouGenerator(model, config={"target_output_ub": target_output_ub})


def generate_module(spec_text, workdir, grammar_text=None, target_output_ub=None):
    """Phase 1: parse the spec and write the Marabou query module. No solver involved.

    Returns (generator, code, module_path). Anything raised here is a specification or
    code-generation problem, which the UI reports verbatim.
    """
    generator = build_generator(spec_text, grammar_text, target_output_ub)
    code = generator.generate()
    module_path = Path(workdir) / "generated_verify_model.py"
    module_path.write_text(code, encoding="utf-8")
    return generator, code, module_path


def solve_module(module_path, onnx_path):
    """Phase 2: run the generated module. Needs maraboupy and the ONNX file.

    Returns check_property()'s dict: {"results", "trace", "rules"}.
    """
    spec = importlib.util.spec_from_file_location(
        f"generated_{uuid.uuid4().hex[:6]}", str(module_path)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.check_property(str(onnx_path))


def network_interface(onnx_path):
    """How many inputs and outputs a model has -> (inputs, outputs).

    Read with the solver's own loader, and flattened the way the generated query reads
    them, so the counts are the ones that query will index into.
    """
    from maraboupy import Marabou
    net = Marabou.read_onnx(str(onnx_path))
    return (len(net.inputVars[0][0].flatten()),
            len(net.outputVars[0][0].flatten()))


def interface_mismatch(onnx_path, generator):
    """What about this model does not match the specification, in words, or None.

    The generated query refers to the network by position - measure 0 is input 0, and so
    on - so a model with a different shape fails somewhere inside it, as an index error
    about an array. Said here instead, while the two files can still be named.
    """
    try:
        inputs, outputs = network_interface(onnx_path)
    except Exception:
        return None            # unreadable here: let the solver say why, in its own words

    def count(n, word):
        return f"{n} {word}{'' if n == 1 else 's'}"

    wrong = []
    if inputs != len(generator.measures):
        wrong.append(f"The network has {count(inputs, 'input')}, but the specification "
                     f"declares {count(len(generator.measures), 'measure')}.")
    if outputs != len(generator.outputs):
        wrong.append(f"The network has {count(outputs, 'output')}, but the "
                     f"specification declares "
                     f"{count(len(generator.outputs), 'output event')}.")
    if not wrong:
        return None
    wrong.append("A model's inputs must be the measures, in declaration order, and its "
                 "outputs the output events.")
    return " ".join(wrong)


def _show(value, kind):
    """A boolean measure takes only 0 or 1, so it reads better as True or False.

    Anything else is shown as the number it is - including a boolean that somehow came
    back between the two, which is worth seeing rather than rounding away.

    Ten significant digits, not four. A strict comparison is encoded by moving the bound
    just inside the threshold, by as little as 1e-6, so at four digits a value sitting
    inside the bound prints as the threshold itself: 1.999999 reads as 2. That hides the
    difference between a counter-example that proves something and one that only touches
    the boundary, where the rule never applied.
    """
    if kind == "boolean":
        if abs(value) <= 1e-6:
            return "False"
        if abs(value - 1.0) <= 1e-6:
            return "True"
    return f"{value:.10g}"


def label_counterexample(cex, measures, outputs, used_names, kinds):
    """Marabou reports {variable number: value}; attach names where the count matches.

    _record_result() in the parser collects input variables first, then outputs, and a
    dict comprehension keeps that insertion order, so zipping against declaration order
    labels them. The parser's language rule is that measures are the network inputs in
    declaration order and output events its outputs, so the orders agree. A variable
    missing from the assignment changes the count: then the raw variable numbers are
    shown rather than a guess that could mislabel every value.

    'used_names' are the inputs the sub-rule's condition compares, from the generated
    module. Those rows are flagged related, and so is every output: the verdict turns on
    which output wins, so the outputs are always part of reading the result.

    'kinds' is the generator's measure_kinds, which says which measures are booleans.

    'output' says which side of the network a row belongs to, so the report can show what
    it reads apart from what it answers. Without names there is no telling the two apart,
    so it is None there rather than a guess.
    """
    names = ([n for n, _ in sorted(measures.items(), key=lambda kv: kv[1])]
             + [n for n, _ in sorted(outputs.items(), key=lambda kv: kv[1])])
    items = list(cex.items())
    if len(items) != len(names):
        return [{"name": f"variable {v}", "var": v, "value": f"{val:.4g}",
                 "raw": float(val), "related": True, "output": None}
                for v, val in items]

    related = set(used_names) | set(outputs)
    # 'raw' is the number a scene renderer draws from; 'value' is what the table shows.
    return [{"name": n, "var": v, "value": _show(val, kinds.get(n)), "raw": float(val),
             "related": n in related, "output": n in outputs}
            for n, (v, val) in zip(names, items)]


# The words the grammar reserves, as it lists them. A rule can only hold some of these,
# but marking by the language's own list is what keeps the two in step.
KEYWORDS = ("and", "boolean", "causation", "constant", "days", "else", "equal", "event",
            "eventually", "exists", "false", "for", "forbid", "happenBefore", "hours",
            "iff", "imply", "includes", "invariant", "meanwhile", "measure", "minutes",
            "mutualExclusive", "not", "numeric", "opposite", "or", "otherwise", "scale",
            "seconds", "then", "true", "unless", "until", "when", "while", "within",
            "witness")


def highlight_rule(text, measures, outputs, constants=()):
    """The rule as written, with its keywords, what it reads, demands and compares.

    A measure is read in braces, so {v1X} is marked whole; an output event and a constant
    are bare words and are marked wherever they stand alone. Everything is escaped first,
    so the rule's own text can never introduce markup - and the keywords are marked before
    anything else, while the text is still only text.
    """
    marked = html.escape(text or "")
    marked = re.sub(r"\b(" + "|".join(KEYWORDS) + r")\b",
                    r'<b class="kw">\1</b>', marked)
    for names, cls in ((measures, "measure"), (outputs, "output"),
                       (constants, "constant")):
        if not names:
            continue
        pattern = "|".join(map(re.escape, sorted(names, key=len, reverse=True)))
        # These passes read text the keyword pass has already marked up, so a name that
        # happened to be 'b', 'kw' or 'class' could otherwise be matched inside a tag:
        # reaching '>' without crossing '<' means the match is inside one.
        outside_tags = r"(?![^<]*>)"
        if cls == "measure":
            marked = re.sub(r"\{(" + pattern + r")\}" + outside_tags,
                            r'<b class="measure">{\1}</b>', marked)
        else:
            marked = re.sub(r"\b(" + pattern + r")\b" + outside_tags,
                            rf'<b class="{cls}">\1</b>', marked)
    return marked


def rule_sections(text):
    """A rule as written -> its sections: the main case, then one per defeater.

    A defeater opens with 'unless' at the top level of the rule. Every condition is
    parenthesised and a measure is read inside braces, so an 'unless' at depth zero can
    only be the keyword. Cutting the text rather than the tree keeps the sections aligned
    with what the reader sees, character for character.
    """
    cuts, depth, i, end = [], 0, 0, len(text)
    while i < end:
        char = text[i]
        if char.isalpha() or char == "_":
            start = i
            while i < end and (text[i].isalnum() or text[i] == "_"):
                i += 1
            if depth == 0 and text[start:i] == "unless":
                cuts.append(start)
            continue
        if char in "({":
            depth += 1
        elif char in ")}":
            depth -= 1
        i += 1
    bounds = [0] + cuts + [end]
    return [text[a:b].strip() for a, b in zip(bounds, bounds[1:])]


def section_owners(name, sections, defeaters, subrules):
    """Each written section -> the sub-rule that checked it, or None.

    The main case is checked by '<r>_main', or by '<r>' itself when the rule has no
    defeater. The parser numbers defeaters by their position in the text, so the i-th
    'unless' section maps straight onto '<r>_unless<i>'. A defeater with no 'then' has no
    sub-rule of its own - its 'not C' is a conjunct of the main condition, and that is the
    only place it is ever checked - so it belongs to the main case.

    If the text and the tree disagree on how many defeaters there are, no section is
    claimed: a rule coloured by the wrong check would read worse than one not coloured.

    One answer, used twice: to colour the rule as written by each section's verdict, and
    to show a sub-rule which part of that rule it is.
    """
    main = (f"{name}_main" if f"{name}_main" in subrules
            else name if name in subrules else None)
    if len(sections) != defeaters + 1:
        return [None] * len(sections)
    return [main] + [(f"{name}_unless{i}" if f"{name}_unless{i}" in subrules else main)
                     for i in range(1, defeaters + 1)]


def chosen_output(rows):
    """The output event the network picked on this assignment: its largest output.

    The argmax head takes exactly one action, and the report names it in the
    specification's own words - the names its rules use - rather than in those of
    whatever environment the model was trained in.
    """
    outputs = [row for row in rows if row.get("output")]
    return max(outputs, key=lambda row: row["raw"])["name"] if outputs else None


def scene_renderer(generator):
    """The scene renderer for this specification's interface, or None if we know none."""
    return scenes.pick_renderer(generator.measures, generator.outputs)


def build_report(outcome, generator):
    """Shape check_property()'s dict into one panel per original rule, in spec order.

    Nothing is recomputed: the grouping, the rule text and the per-rule verdict all come
    from the generated module. Only presentation is derived here - the per-sub-rule label,
    the names of the counter-example variables and the drawing of the scene.
    """
    results = outcome["results"]
    renderer = scene_renderer(generator)
    # The sub-rules exist only as trees; printing them back gives the report something
    # to show beside the rule as written.
    normalized = {n.name: n for n in generator.normalize_rules()}
    # How many defeaters each rule was written with, so the sections cut out of its text
    # can be checked against the tree before they are coloured by a verdict.
    defeaters = {rule.name: len(rule.response.defeater or [])
                 for rule in generator.model.ruleBlock.rules}

    def marked(text):
        """Rule text with what it reads, what it demands and what it compares marked."""
        return highlight_rule(text, generator.measures, generator.outputs,
                              generator.constants)

    def constants_of(text):
        """The constants that text names, with their values."""
        return ruletext.constants_used(text, generator.constants)

    groups = []
    for name, info in outcome["rules"].items():
        text = (info["text"] or "").strip()
        # The panel already carries the rule's name, so the text starts at 'when'.
        if text.startswith(name):
            text = text[len(name):].lstrip()
        # The rule as written, cut at its defeaters, and which sub-rule checked each cut.
        # A sub-rule is shown as its own part of this text rather than as the normalized
        # form it was verified in: the rule is what the reader knows.
        sections = rule_sections(text)
        owners = section_owners(name, sections, defeaters.get(name, 0),
                                set(info["subrules"]))
        marks = [marked(section) for section in sections]
        constants = constants_of(text)

        parts = []
        for sub in info["subrules"]:
            verified, cex, exit_code = results[sub]
            if verified:
                status = "COMPLIANT"
            elif exit_code == "sat":
                status = "VIOLATED"
            else:
                # Neither proved nor broken: the solver stopped without deciding. Its own
                # word for it lives in exit_code, which the report does not show.
                status = "UNVERIFIED"
            # The inputs this sub-rule's condition compares; the rest of the 25 were only
            # bounded by their declared ranges and say nothing about why the rule failed.
            used = [field["name"] for field in outcome["inputs"][sub]]
            violated = status == "VIOLATED"
            rows = (label_counterexample(cex, generator.measures, generator.outputs,
                                         used, generator.measure_kinds)
                    if violated else [])
            # What the rule demanded here, in its own words: 'outputSlower', or
            # 'not outputSlower' where the rule forbids rather than requires, or
            # 'outputHalt else outputSlow' where it chains the actions it accepts.
            # 'allowed' is the same thing as names: a violation is the network taking
            # none of them, so every one of them is what the reader is pointed at.
            occs = normalized[sub].occs if sub in normalized else []
            expected = ruletext.response_text(occs) if occs else None
            allowed = [occ.event.event.name for occ in occs]
            # What the model did instead, named as the specification names it.
            action = chosen_output(rows) if violated else None
            # The measures this part of the rule names itself. Narrower than 'used': a
            # normalized sub-rule also carries the conditions it was given to rule the
            # other alternatives out, so it compares inputs its own text never mentions.
            own_text = " ".join(text for text, owner in zip(sections, owners)
                                if owner == sub)
            names_here = set(ruletext.names_used(own_text, generator.measures))
            # Which rows the reader is being pointed at: the measures this part names,
            # every action it would have accepted, and the one the network took instead.
            # The last wins where a rule forbids an output and the network produced it,
            # so the same row is both demanded against and taken.
            for row in rows:
                if row["output"]:
                    row["role"] = ("got" if row["name"] == action
                                   else "wanted" if row["name"] in allowed else None)
                else:
                    row["role"] = "used" if row["name"] in names_here else None
            parts.append({
                "name": sub,
                "status": status,
                "text": text,
                # The rule as written, with this sub-rule's own part of it coloured and
                # the rest left plain: which part of the rule this check was about.
                # Inside a sub-rule's own diagnosis there is nothing left to open, so no
                # section carries an owner to open it with.
                "sections": [{"html": mark, "owner": None,
                              "status": status if owner == sub else None}
                             for mark, owner in zip(marks, owners)],
                "constants": constants,
                "expected": expected,
                # The actions accepted, so the report can say whether it is naming one
                # demanded response or a chain of alternatives.
                "allowed": allowed,
                "related_rows": [row for row in rows if row["related"]],
                "all_rows": rows,
                "action": action,
                # Only a violation has an assignment to draw. scenes.render() refuses
                # empty rows anyway; this says so where the decision is made.
                "scene": scenes.render(renderer, rows) if violated else None,
            })
        groups.append({
            "name": name,
            "text": text,
            # The text cut at its defeaters: each section marked up, carrying the verdict
            # of the check that covered it and the name of that check - which is what the
            # report opens the section's own diagnosis by. A section no check covered
            # carries neither. A bare defeater is covered by the main case, so its
            # section names that one: there is no separate check to show for it.
            "sections": [{"html": mark,
                          "status": {p["name"]: p["status"] for p in parts}.get(owner),
                          "owner": owner if owner in {p["name"] for p in parts}
                          else None}
                         for mark, owner in zip(marks, owners)],
            "constants": constants,
            "verdict": info["verdict"],
            "parts": parts,
        })
    return groups
