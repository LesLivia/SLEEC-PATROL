"""Verification logic behind the web UI, with no web framework involved.

Kept apart from app.py so it can be exercised without FastAPI, and (up to the
Marabou call) without maraboupy: run_verification() is the only part that needs
the solver, so parsing, code generation and result labelling stay testable on any
machine.
"""

import importlib.util
import sys
import uuid
from pathlib import Path

from textx import metamodel_from_str

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from verifier.sleec2marabou_parser import MarabouGenerator  # noqa: E402

DEFAULT_GRAMMAR = ROOT / "legos_sleec" / "sleec-grammar.txt"


def build_generator(spec_text, grammar_text=None, target_output_ub=None):
    """Parse the spec and return a MarabouGenerator for it."""
    grammar_text = grammar_text or DEFAULT_GRAMMAR.read_text(encoding="utf-8")
    model = metamodel_from_str(grammar_text).model_from_str(spec_text)
    return MarabouGenerator(model, config={"target_output_ub": target_output_ub})


def import_generated(path):
    """Import a generated module from disk under a unique name (so runs never collide)."""
    spec = importlib.util.spec_from_file_location(f"generated_{uuid.uuid4().hex[:6]}", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def label_counterexample(cex, measures, outputs):
    """Marabou reports {variable number: value}; attach names where the count matches.

    _record_result() in the parser collects input variables first, then outputs, and
    a dict comprehension keeps that insertion order, so zipping against declaration
    order labels them. The parser's language rule is that measures are the network
    inputs in declaration order and output events its outputs, so the orders agree.
    A variable missing from the assignment changes the count: then the raw variable
    numbers are shown rather than a guess that could mislabel every value.
    """
    names = ([n for n, _ in sorted(measures.items(), key=lambda kv: kv[1])]
             + [n for n, _ in sorted(outputs.items(), key=lambda kv: kv[1])])
    items = list(cex.items())
    if len(items) == len(names):
        return [{"name": n, "var": v, "value": f"{val:.4g}"} for n, (v, val) in zip(names, items)]
    return [{"name": f"variable {v}", "var": v, "value": f"{val:.4g}"} for v, val in items]


def summarize(outcome, generator):
    """Turn check_property()'s {rule: (verified, cex, exit_code)} into rows for the page.

    Only 'unsat' proves a rule; 'sat' means Marabou found a counter-example; anything
    else (timeout, error) is reported as-is rather than being read as either verdict.
    """
    rows = []
    for name, (verified, cex, exit_code) in outcome.items():
        if verified:
            status = "VERIFIED"
        elif exit_code == "sat":
            status = "VIOLATED"
        else:
            status = f"UNKNOWN ({exit_code})"
        rows.append({
            "name": name,
            "origin": generator.trace.get(name, name),
            "status": status,
            "counterexample": (label_counterexample(cex, generator.measures, generator.outputs)
                               if status == "VIOLATED" else []),
        })
    return rows


def run_verification(spec_text, onnx_path, workdir, grammar_text=None, target_output_ub=None):
    """Parse, generate, solve every rule, and return (rows, generated_code, generator).

    Needs maraboupy, which the generated module imports.
    """
    generator = build_generator(spec_text, grammar_text, target_output_ub)
    code = generator.generate()
    code_path = Path(workdir) / "generated_verify_model.py"
    code_path.write_text(code, encoding="utf-8")
    outcome = import_generated(code_path).check_property(str(onnx_path))
    return summarize(outcome, generator), code, generator
