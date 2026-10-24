"""Run a generated module's check_property() in a process of its own (core.solve_module).

    python webapp/solve_run.py <module> <onnx> <answer.pickle>
"""
import importlib.util
import pickle
import sys

module_path, onnx_path, answer = sys.argv[1:]
spec = importlib.util.spec_from_file_location("generated_verify_model", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
with open(answer, "wb") as f:
    pickle.dump(module.check_property(onnx_path), f)
