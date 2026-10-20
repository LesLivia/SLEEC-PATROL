"""Minimal FastAPI UI for SLEEC -> Marabou verification.

Page 1 (GET /)           what the tool is, how it works, and a way in.
Page 2 (GET /upload)     upload a .sleec specification and an .onnx network.
POST /verify             parse + generate the query, then solve every rule.
Page 3 (GET /results/x)  one panel per rule as written, holding its normalized
                         sub-rules; a violated sub-rule expands to its counter-example.

Verification logic lives in core.py; the verdicts and grouping come from the generated
module, so nothing is recomputed here. Uploads go to a temp folder, never into the repo.

Run it inside WSL, where maraboupy lives:
    pip install fastapi uvicorn jinja2 python-multipart
    cd /mnt/d/SLEEC_verif/SLEEC2Ver && python -m uvicorn webapp.app:app --port 8000
then open http://localhost:8000 in Windows.
"""

import re
import tempfile
import traceback
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, Request, Response, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from webapp import core

ROOT = Path(__file__).resolve().parent.parent
app = FastAPI(title="SLEEC-PATROL")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
ARCHITECTURE = Path(__file__).parent / "static" / "architecture.png"

# Single-user local tool: finished runs live in memory, keyed by a short id.
RUNS = {}

# The verdicts rule_verdicts() returns, which are also the words the report shows.
VERDICTS = ("COMPLIANT", "VIOLATED", "UNVERIFIED", "IGNORED")

# Set this once the walkthrough is recorded. Until then the home page names the video in
# its closing sentence as plain words: there is nothing to link them to yet.
VIDEO_URL = ""

ONNX_GUIDE = ("https://docs.pytorch.org/tutorials/beginner/onnx/"
              "export_simple_model_to_onnx_tutorial.html")


def _without_comments(text):
    """The example specification as rules only: // comments carry the reasoning, and the
    instructions explain the form, so showing them again would bury the shape of it.

    A comment-only line leaves a blank behind it, which is wanted between declarations and
    unwanted where it opens a gap just inside a block, so those are dropped.
    """
    kept = []
    for line in text.splitlines():
        code = re.sub(r"//.*$", "", line).rstrip()
        if code.strip():
            if kept and kept[-1] == "" and code.strip().endswith("_end"):
                kept[-1] = code                    # a blank sitting just before a closer
            else:
                kept.append(code)
        elif kept and kept[-1] != "" and not kept[-1].strip().endswith("_start"):
            kept.append("")                        # ...and none just after an opener
    return "\n".join(kept).strip() + "\n"


EXAMPLE_SPEC_PATH = ROOT / "resources" / "requirements" / "highway_behaviourplanner.sleec"
EXAMPLE_SPEC = (_without_comments(EXAMPLE_SPEC_PATH.read_text(encoding="utf-8"))
                if EXAMPLE_SPEC_PATH.exists() else "")


EXAMPLES = Path(__file__).parent / "examples"


def examples():
    """The setups that ship with the tool: one folder each, holding its own two files.

    A folder counts only if it holds exactly one specification and one model, so a
    half-filled one is passed over rather than offered and then failing at verification.
    Adding a setup is adding a folder; nothing here needs to know about it.
    """
    found = []
    for folder in sorted(p for p in EXAMPLES.glob("*") if p.is_dir()):
        specs, models = sorted(folder.glob("*.sleec")), sorted(folder.glob("*.onnx"))
        if len(specs) == 1 and len(models) == 1:
            found.append({"name": folder.name, "spec": specs[0], "model": models[0]})
    return found


def _upload_context(**extra):
    """Everything the upload page needs besides its own state.

    'error' is named here rather than left out when there is none, so the page never
    reads a name the context does not carry: an error page passes its own through
    'extra', which takes precedence.
    """
    return {"example_spec": EXAMPLE_SPEC, "onnx_guide": ONNX_GUIDE, "error": None,
            # What the page was holding. A run that fails comes back here, and rules
            # someone typed are not theirs to lose over a missing 'then'.
            "spec_text": "", "source": "ide", "setup": "",
            # Each setup carries its rules with it, so the page can show them without
            # asking for them again.
            "setups": [dict(e, text=e["spec"].read_text(encoding="utf-8"))
                       for e in examples()],
            **extra}


def _error_page(request, headline, detail, collapse, **state):
    """Back to the upload page with a message, and with the page as it was.

    'collapse' hides the detail behind a toggle. 'state' is what was being given when it
    failed - the rules in the editor, the side they were given on - so a run that does not
    start costs nothing but the attempt.
    """
    return templates.TemplateResponse(
        request, "upload.html",
        _upload_context(error={"headline": headline, "detail": detail,
                               "collapse": collapse}, **state),
        status_code=400,
    )


@app.get("/", response_class=HTMLResponse)
def home_page(request: Request):
    # Starlette >= 0.29 takes the request first; the old (name, context) order is gone.
    return templates.TemplateResponse(request, "home.html", {"video_url": VIDEO_URL})


@app.get("/figures/architecture.png")
def architecture_figure():
    """The diagram on the home page. A route, not a mount, so a missing file 404s."""
    return FileResponse(ARCHITECTURE, media_type="image/png")


@app.get("/upload", response_class=HTMLResponse)
def upload_page(request: Request):
    return templates.TemplateResponse(request, "upload.html", _upload_context())


@app.post("/verify")
async def verify(request: Request,
                 network: Optional[UploadFile] = File(None),
                 spec: Optional[UploadFile] = File(None),
                 spec_text: str = Form(""),
                 source: str = Form("ide"),
                 setup: str = Form("")):
    """Either the inputs given on the page, or a setup that ships with the tool.

    On the page, the network arrives as a file and the specification as the editor's
    text: a file loaded there is read into the editor, so 'spec_text' is what was seen
    and is what is verified. That same file input is still posted, which is the only way
    a specification can arrive when the page's scripting did not run - then the editor is
    empty and the file is read here instead.

    'source' says which. A setup is named, never pathed: the name has to be one of the
    folders found under webapp/examples, so nothing a form can say reaches another file.
    """
    workdir = Path(tempfile.mkdtemp(prefix="sleec2ver_"))
    # What to put the page back to if this run does not start. The editor's text is kept
    # as it was typed; a file read on the server is not, since the file is still there.
    was = {"spec_text": spec_text, "source": source, "setup": setup}

    if source == "existing":
        chosen = next((e for e in examples() if e["name"] == setup), None)
        if chosen is None:
            return _error_page(request, "That setup is not available",
                               "Choose one of the setups listed on the form, or give the "
                               "specification and the model directly.", collapse=False, **was)
        spec_text = chosen["spec"].read_text(encoding="utf-8")
        spec_name, onnx_path = chosen["spec"].name, chosen["model"]
        network_name = chosen["model"].name
    else:
        if network is None or not network.filename:
            return _error_page(request, "No neural network was chosen",
                               "Choose the .onnx file of the model to verify, or pick a "
                               "setup that comes with the tool.", collapse=False, **was)
        onnx_path = workdir / Path(network.filename).name
        onnx_path.write_bytes(await network.read())
        network_name = network.filename

        if spec_text.strip():
            # Whatever was loaded, the editor is where it was read, edited and verified
            # from - not a file whose contents may since differ.
            spec_name = "SLEEC rules from IDE"
        elif spec is not None and spec.filename:
            spec_text = (await spec.read()).decode("utf-8")
            spec_name = spec.filename
        if not spec_text.strip():
            return _error_page(request, "The specification is empty",
                               "Write the rules in the editor, or load a .sleec file "
                               "into it.", collapse=False, **was)

    # Phase 1: the spec and the generated query. The parser's messages name the exact
    # constant, measure or rule at fault, so they are shown as they are.
    try:
        generator, code, module_path = core.generate_module(spec_text, workdir)
    except Exception as exc:
        return _error_page(request, "The specification could not be turned into a query",
                           f"{type(exc).__name__}: {exc}", collapse=False, **was)

    # Between the two phases: does this model have the interface the specification
    # describes? Asked here, while both files can still be named - inside the query it
    # would only be an index error about an array.
    mismatch = core.interface_mismatch(onnx_path, generator)
    if mismatch:
        return _error_page(request, "The model does not match the specification",
                           mismatch, collapse=False, **was)

    # Phase 2: Marabou. Failures here are about the network or the solver, not the spec,
    # so the headline stays generic and the detail sits under a toggle.
    try:
        outcome = core.solve_module(module_path, onnx_path)
        groups = core.build_report(outcome, generator)
    except Exception as exc:
        # The page gets the exception; the terminal gets the traceback, which is the only
        # place anything can be worked out from when the failure is not the obvious one.
        traceback.print_exc()
        return _error_page(request, "An error occurred during Marabou verification",
                           f"{type(exc).__name__}: {exc}", collapse=True, **was)

    run_id = uuid.uuid4().hex[:8]
    RUNS[run_id] = {
        "spec_name": spec_name,
        # Kept so the report can show what was actually verified, typed or uploaded.
        "spec_text": spec_text,
        "network_name": network_name,
        "groups": groups,
        "code": code,
        # Said once at the top of the report rather than under every counter-example.
        "has_scene": core.scene_renderer(generator) is not None,
    }
    return RedirectResponse(f"/results/{run_id}", status_code=303)


@app.get("/results/{run_id}", response_class=HTMLResponse)
def results_page(request: Request, run_id: str):
    run = RUNS.get(run_id)
    if run is None:
        return RedirectResponse("/upload", status_code=303)   # an expired run id
    counts = {v: sum(1 for g in run["groups"] if g["verdict"] == v) for v in VERDICTS}
    checks = sum(len(g["parts"]) for g in run["groups"])
    return templates.TemplateResponse(
        request, "results.html",
        {"run": run, "run_id": run_id, "counts": counts, "checks": checks},
    )


@app.get("/results/{run_id}/code")
def verification_code(run_id: str):
    """The generated Marabou query, as a file.

    The report links to it instead of printing it: it is the longest thing a run produces,
    and it is read in an editor rather than in a browser. An expired run id goes back to
    the form, as the report itself does.
    """
    run = RUNS.get(run_id)
    if run is None:
        return RedirectResponse("/upload", status_code=303)
    filename = f"marabou_query_{run_id}.py"
    return Response(
        run["code"], media_type="text/x-python",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
