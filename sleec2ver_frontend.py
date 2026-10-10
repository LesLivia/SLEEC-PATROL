import importlib
import os
import sys
from os.path import dirname, join

sys.path.append(join(dirname(__file__), "analyzer"))

import idlelib.colorizer as ic
import idlelib.percolator as ip
import re
import tkinter as tk
from tkinter import ttk, messagebox
from textx import metamodel_from_str

from verifier.sleec2marabou_parser import MarabouGenerator

from legos_sleec.sleecParser import (
    check_input_red,
    check_input_conflict,
    check_input_concerns,
    check_input_purpose,
)
from legos_sleec.SleecNorm import check_situational_conflict

# -------------------------------------------------------------------
# Paths and utilities
# -------------------------------------------------------------------

REQ_DIR = "resources/requirements"
MODEL_DIR = "verifier/resources/models"


def list_files(directory, extensions=None):
    if not os.path.isdir(directory):
        return []
    out = []
    for f in os.listdir(directory):
        if extensions is None or f.endswith(extensions):
            out.append(f)
    return sorted(out)


def read_text_file(file_path):
    with open(file_path, "r") as file:
        return file.read()


# -------------------------------------------------------------------
# Output helpers
# -------------------------------------------------------------------

def output_clear():
    output_text.configure(state="normal")
    output_text.delete("1.0", tk.END)
    output_text.configure(state="disabled")


def output_write(msg: str):
    output_text.configure(state="normal")
    output_text.insert(tk.END, msg + "\n")
    output_text.see(tk.END)
    output_text.configure(state="disabled")


def output_header(title: str):
    output_write("=" * 72)
    output_write(title)
    output_write("=" * 72)


# -------------------------------------------------------------------
# Highlighted response rendering (keeps your existing right pane)
# -------------------------------------------------------------------

def _render_highlighted_response(result, response, hl):
    new_text.delete("1.0", "end-1c")
    if result and hl:
        cur_index = 0
        sorted_hl = sorted(hl, key=lambda k: k[0], reverse=True)
        while sorted_hl:
            cur_start, cur_end = sorted_hl.pop()
            if cur_start >= cur_index:
                new_text.insert(tk.INSERT, response[cur_index:cur_start])
                new_text.insert(tk.INSERT, response[cur_start:cur_end], "hl")
                cur_index = cur_end
        if cur_index < len(response):
            new_text.insert(tk.INSERT, response[cur_index:])
    else:
        new_text.insert(tk.INSERT, response)


# -------------------------------------------------------------------
# Check callbacks
# -------------------------------------------------------------------

def check_redundancy():
    output_header("SMT CHECK: Redundancy")
    cur_text = aText.get("1.0", "end-1c")
    result, response, hl = check_input_red(cur_text)
    output_write(f"Status: {'ISSUE FOUND' if result else 'OK'}")
    _render_highlighted_response(result, response, hl)


def check_conflict():
    output_header("SMT CHECK: Conflicts")
    cur_text = aText.get("1.0", "end-1c")
    result, response, hl = check_input_conflict(cur_text)
    output_write(f"Status: {'ISSUE FOUND' if result else 'OK'}")
    _render_highlighted_response(result, response, hl)


def check_concern():
    output_header("SMT CHECK: Concern")
    cur_text = aText.get("1.0", "end-1c")
    result, response, hl = check_input_concerns(cur_text)
    output_write(f"Status: {'ISSUE FOUND' if result else 'OK'}")
    _render_highlighted_response(result, response, hl)


def check_purpose():
    output_header("SMT CHECK: Purpose")
    cur_text = aText.get("1.0", "end-1c")
    result, response, hl = check_input_purpose(cur_text)
    output_write(f"Status: {'ISSUE FOUND' if result else 'OK'}")
    _render_highlighted_response(result, response, hl)


def check_situational():
    output_header("SMT CHECK: Situational Conflict")
    cur_text = aText.get("1.0", "end-1c")
    result, response, hl = check_situational_conflict(cur_text)
    output_write(f"Status: {'ISSUE FOUND' if result else 'OK'}")
    _render_highlighted_response(result, response, hl)


# -------------------------------------------------------------------
# Verification callback (HOOK POINT)
# -------------------------------------------------------------------

def verify_model():
    if not selected_req.get():
        messagebox.showerror("Verification error", "No requirements file selected")
        return
    if not selected_model.get():
        messagebox.showerror("Verification error", "No model selected")
        return

    output_header("NN VERIFICATION")
    spec_text = aText.get("1.0", "end-1c")
    model_path = os.path.join(MODEL_DIR, selected_model.get())

    output_write(f"Requirements: {selected_req.get()}")
    output_write(f"Model: {selected_model.get()}")
    output_write("")

    output_write("Running verification...")
    grammar_file = "legos_sleec/sleec-grammar.txt"
    with open(grammar_file, 'r') as f:
        grammar = f.read()
    selected_req_path = os.path.join(REQ_DIR, selected_req.get())
    with open(selected_req_path, 'r') as f:
        dsl_snippet = f.read()
    meta = metamodel_from_str(grammar)
    model = meta.model_from_str(dsl_snippet)
    try:
        ub = float(ub_value.get()) if require_ub.get() else None
    except ValueError:
        messagebox.showerror("Verification error", f"Not a number: {ub_value.get()}")
        return
    gen = MarabouGenerator(model, config={"target_output_ub": ub})
    output_write(f"Config: {gen.config}")
    with open("verifier/generated_verify_model.py", 'w') as f:
        f.write(gen.generate())
    # reload: the module stays cached between VERIFY clicks
    from verifier import generated_verify_model
    importlib.reload(generated_verify_model)
    outcome = generated_verify_model.check_property(model_path)

    new_text.delete("1.0", tk.END)
    new_text.insert(
        tk.INSERT,
        "NN Verification Report\n\n"
        f"Model: {model_path}\n"
    )
    for res, (verified, counterexample, exit_code) in outcome.items():
        if verified:
            output_write(f"Result: PROPERTY {res} VERIFIED")
            detail = f"Property: {res}\nOutcome: verified\n"
        elif exit_code == "sat":
            output_write(f"Result: PROPERTY {res} VIOLATED")
            detail = (f"Property: {res}\nOutcome: violated\n"
                      f"Counterexample: {counterexample}\n")
        else:
            output_write(f"Result: PROPERTY {res} UNKNOWN (marabou: {exit_code})")
            detail = f"Property: {res}\nOutcome: unknown ({exit_code})\n"
        new_text.insert(tk.END, "\n" + detail)


# -------------------------------------------------------------------
# Tkinter setup
# -------------------------------------------------------------------

lord = tk.Tk()
lord.title("SLEEC-D")

width = lord.winfo_screenwidth()
height = lord.winfo_screenheight()
lord.geometry(f"{width * 2 // 3}x{height}")

# -------------------------------------------------------------------
# Top bar: dropdown selectors
# -------------------------------------------------------------------

top_frame = tk.Frame(lord)
top_frame.pack(fill=tk.X, padx=6, pady=6)

selected_req = tk.StringVar()
selected_model = tk.StringVar()

tk.Label(top_frame, text="Requirements:").grid(row=0, column=0, sticky="w")
req_menu = ttk.Combobox(
    top_frame,
    textvariable=selected_req,
    values=list_files(REQ_DIR, (".sleec",)),
    state="readonly",
    width=45,
)
req_menu.grid(row=0, column=1, padx=6)


def on_req_selected(event):
    path = os.path.join(REQ_DIR, selected_req.get())
    aText.delete("1.0", tk.END)
    aText.insert(tk.INSERT, read_text_file(path))
    output_header("Loaded requirements file")
    output_write(path)


req_menu.bind("<<ComboboxSelected>>", on_req_selected)

tk.Label(top_frame, text="Model:").grid(row=0, column=2, sticky="w")
model_menu = ttk.Combobox(
    top_frame,
    textvariable=selected_model,
    values=list_files(MODEL_DIR, (".onnx",)),
    state="readonly",
    width=45,
)
model_menu.grid(row=0, column=3, padx=6)

# -------------------------------------------------------------------
# Action bar: moved BELOW dropdowns, at the top
# -------------------------------------------------------------------

action_frame = tk.Frame(lord)
action_frame.pack(fill=tk.X, padx=6, pady=(0, 6))

tk.Button(action_frame, text="check redundancy", command=check_redundancy).pack(side=tk.LEFT, padx=2)
tk.Button(action_frame, text="check conflicts", command=check_conflict).pack(side=tk.LEFT, padx=2)
tk.Button(action_frame, text="check concern", command=check_concern).pack(side=tk.LEFT, padx=2)
tk.Button(action_frame, text="check purpose", command=check_purpose).pack(side=tk.LEFT, padx=2)
tk.Button(action_frame, text="check situational conflict", command=check_situational).pack(side=tk.LEFT, padx=2)

tk.Button(
    action_frame,
    text="VERIFY",
    command=verify_model,
    bg="#2ecc71",
    font=("Helvetica", 12, "bold"),
).pack(side=tk.RIGHT, padx=5)

# Optional extra bound on the output under test; off by default.
require_ub = tk.BooleanVar(value=False)
ub_value = tk.StringVar(value="0")
tk.Entry(action_frame, textvariable=ub_value, width=6).pack(side=tk.RIGHT)
tk.Checkbutton(action_frame, text="bound target output <=",
               variable=require_ub).pack(side=tk.RIGHT, padx=(12, 0))

# -------------------------------------------------------------------
# Output box: dedicated panel for SMT + NN verification output
# -------------------------------------------------------------------

output_frame = tk.LabelFrame(lord, text="Output (SMT + NN Verification)", padx=6, pady=6)
output_frame.pack(fill=tk.X, padx=6, pady=(0, 6))

output_scroll = tk.Scrollbar(output_frame)
output_scroll.pack(side=tk.RIGHT, fill=tk.Y)

output_text = tk.Text(output_frame, height=8, font=("Consolas", 12), yscrollcommand=output_scroll.set)
output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
output_scroll.config(command=output_text.yview)
output_text.configure(state="disabled")

# -------------------------------------------------------------------
# Editor panes (left: spec editor, right: highlighted report)
# -------------------------------------------------------------------

text_frame = tk.Frame(lord)
text_frame.pack(expand=True, fill=tk.BOTH, padx=6, pady=6)

aText = tk.Text(text_frame, font=("Georgia", 14))
aText.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=(0, 6))

new_text = tk.Text(text_frame, font=("Georgia", 14))
new_text.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
new_text.tag_config("hl", background="yellow")

# -------------------------------------------------------------------
# Syntax highlighting (keep your DSL keywords)
# -------------------------------------------------------------------

KEYWORD = r"\b(?P<KEYWORD>event|measure|constant|when|then|within|minutes|hours|seconds|unless|otherwise|boolean|numeric|scale|not|and|or|def_start|def_end|rule_end|rule_start|concern_start|concern_end|purpose_start|purpose_end|True|False|eventually|else|exists|while|meanwhile|invariant|witness|coincide|conflict|happenBefore|relation_start|relation_end|negate|imply|iff|negate|causation|effect|until|for|includes|forbid|mutualExclusive)\b"
COMMENT = r"(?P<COMMENT>//[^\n]*)"
NUMBER = r"\b(?P<NUMBER>((\d*\.)?\d+))\b"
PROG = rf"{KEYWORD}|{COMMENT}|{NUMBER}"

TAGDEFS = {
    "COMMENT": {"foreground": "#013220", "background": None, "font": "Helvetica 14 bold"},
    "NUMBER": {"foreground": "#ffa500", "font": "Helvetica 14 bold"},
    "KEYWORD": {"foreground": "#8e44ad", "font": "Helvetica 14 bold"},
}

cd = ic.ColorDelegator()
cd.prog = re.compile(PROG, re.S | re.M)
cd.tagdefs = {**cd.tagdefs, **TAGDEFS}
ip.Percolator(aText).insertfilter(cd)

cd1 = ic.ColorDelegator()
cd1.prog = re.compile(PROG, re.S | re.M)
cd1.tagdefs = {**cd1.tagdefs, **TAGDEFS}
ip.Percolator(new_text).insertfilter(cd1)

# -------------------------------------------------------------------
# Startup: preload an example requirements file if available
# -------------------------------------------------------------------

default_reqs = os.path.join(REQ_DIR, "smart_mobility.sleec")
if os.path.isfile(default_reqs):
    aText.insert(tk.INSERT, read_text_file(default_reqs))
    selected_req.set("pima_indians.sleec")
    output_header("Loaded default requirements file")
    output_write(default_reqs)
else:
    aText.insert(tk.INSERT, "// Select a requirements file from the dropdown\n")

output_header("Ready")
output_write("Select a requirements file and model, then run checks or VERIFY MODEL.")

# -------------------------------------------------------------------
# Main loop
# -------------------------------------------------------------------

lord.mainloop()
