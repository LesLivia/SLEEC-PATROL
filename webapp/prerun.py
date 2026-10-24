"""Solve the bundled setups once, as the site would, and keep each answer beside them.

Run it from the repository root, in the environment the site runs in (on Windows, inside
WSL, since maraboupy has no Windows build):
    python -m webapp.prerun              every bundled setup
    python -m webapp.prerun sddoia       only the setups named

Each setup goes through the calls verify() makes for it - its own specification, model,
semantics and threshold, so the same generated query - and solve_module()'s answer is
written to the setup's folder as prerun.json. The site uses that answer only while its own
query and the model are still the ones it answers (core.load_prerun), so after a change to
a setup, its settings or the parser, run this again; until then the setup is solved live.
"""

import sys
import tempfile
import time
from pathlib import Path

from webapp import core
from webapp.app import examples


def prerun(setup):
    """Solve one setup as verify() does for it, and store the answer.

    Returns (where it was stored, the answer).
    """
    spec_text = setup["spec"].read_text(encoding="utf-8")
    workdir = Path(tempfile.mkdtemp(prefix="sleec2ver_prerun_"))
    # The setup's semantics and threshold as setup.json gives them, as verify() passes them.
    generator, code, module_path = core.generate_module(
        spec_text, workdir, output_semantics=setup["semantics"],
        decision_threshold=setup["threshold"])
    mismatch = core.interface_mismatch(setup["model"], generator)
    if mismatch:
        raise ValueError(f"{setup['name']}: {mismatch}")
    outcome = core.solve_module(module_path, setup["model"])
    path = setup["spec"].parent / core.PRERUN
    core.save_prerun(path, code, setup["model"], outcome)
    return path, outcome


def main(names):
    setups = examples()
    unknown = sorted(set(names) - {s["name"] for s in setups})
    if unknown:
        sys.exit(f"No bundled setup named {', '.join(unknown)}. "
                 f"The setups are: {', '.join(s['name'] for s in setups)}.")
    for setup in setups:
        if names and setup["name"] not in names:
            continue
        print(f"--- {setup['name']} ({setup['semantics']})", flush=True)
        start = time.time()
        path, outcome = prerun(setup)
        for rule, info in outcome["rules"].items():
            print(f"    {rule:14s} {info['verdict']}")
        print(f"    solved in {time.time() - start:.0f} s, stored in {path}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
