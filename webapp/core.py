"""Verification logic behind the web UI, with no web framework involved.

Kept apart from app.py so it can be exercised without FastAPI, and - except for
solve_module() - without maraboupy, so parsing, code generation and report building
stay testable on any machine.

The two phases are separate calls on purpose: the UI shows a parser error verbatim,
while a failure inside Marabou gets a generic message with the detail tucked away.
"""

import importlib.util
import sys
import uuid
from pathlib import Path

from textx import metamodel_from_str

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from verifier.sleec2marabou_parser import MarabouGenerator  # noqa: E402
from webapp import scenes  # noqa: E402

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


def label_counterexample(cex, measures, outputs, used_names):
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
    """
    names = ([n for n, _ in sorted(measures.items(), key=lambda kv: kv[1])]
             + [n for n, _ in sorted(outputs.items(), key=lambda kv: kv[1])])
    items = list(cex.items())
    if len(items) != len(names):
        return [{"name": f"variable {v}", "var": v, "value": f"{val:.4g}",
                 "raw": float(val), "related": True} for v, val in items]

    related = set(used_names) | set(outputs)
    # 'raw' is the number a scene renderer draws from; 'value' is what the table shows.
    return [{"name": n, "var": v, "value": f"{val:.4g}", "raw": float(val),
             "related": n in related}
            for n, (v, val) in zip(names, items)]


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
    groups = []
    for name, info in outcome["rules"].items():
        parts = []
        for sub in info["subrules"]:
            verified, cex, exit_code = results[sub]
            if verified:
                status = "VERIFIED"
            elif exit_code == "sat":
                status = "VIOLATED"
            else:
                status = f"UNKNOWN ({exit_code})"
            # The inputs this sub-rule's condition compares; the rest of the 25 were only
            # bounded by their declared ranges and say nothing about why the rule failed.
            used = [field["name"] for field in outcome["inputs"][sub]]
            violated = status == "VIOLATED"
            rows = (label_counterexample(cex, generator.measures, generator.outputs, used)
                    if violated else [])
            parts.append({
                "name": sub,
                "status": status,
                "related_rows": [row for row in rows if row["related"]],
                "all_rows": rows,
                # Only a violation has an assignment to draw. scenes.render() refuses
                # empty rows anyway; this says so where the decision is made.
                "scene": scenes.render(renderer, rows) if violated else None,
            })
        groups.append({
            "name": name,
            "text": (info["text"] or "").strip(),
            "verdict": info["verdict"],
            "parts": parts,
        })
    return groups
