"""Minimal FastAPI UI for SLEEC -> Marabou verification.

Page 1 (GET /)           upload a .sleec specification and an .onnx network.
POST /verify             parse + generate the query, then solve every rule.
Page 2 (GET /results/x)  one panel per rule as written, holding its normalized
                         sub-rules; a violated sub-rule expands to its counter-example.

Verification logic lives in core.py; the verdicts and grouping come from the generated
module, so nothing is recomputed here. Uploads go to a temp folder, never into the repo.

Run it inside WSL, where maraboupy lives:
    pip install fastapi uvicorn jinja2 python-multipart
    cd /mnt/d/SLEEC_verif/SLEEC2Ver && python -m uvicorn webapp.app:app --port 8000
then open http://localhost:8000 in Windows.
"""

import tempfile
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from webapp import core

app = FastAPI(title="SLEEC2Ver")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# Single-user local tool: finished runs live in memory, keyed by a short id.
RUNS = {}

VERDICTS = ("VERIFIED", "VIOLATED", "UNKNOWN", "IGNORED")


def _error_page(request, headline, detail, collapse):
    """Back to the upload page with a message. 'collapse' hides the detail behind a toggle."""
    return templates.TemplateResponse(
        request, "upload.html",
        {"error": {"headline": headline, "detail": detail, "collapse": collapse}},
        status_code=400,
    )


@app.get("/", response_class=HTMLResponse)
def upload_page(request: Request):
    # Starlette >= 0.29 takes the request first; the old (name, context) order is gone.
    return templates.TemplateResponse(request, "upload.html")


@app.post("/verify")
async def verify(request: Request,
                 spec: UploadFile = File(...),
                 network: UploadFile = File(...)):
    workdir = Path(tempfile.mkdtemp(prefix="sleec2ver_"))
    onnx_path = workdir / Path(network.filename or "model.onnx").name
    onnx_path.write_bytes(await network.read())
    spec_text = (await spec.read()).decode("utf-8")

    # Phase 1: the spec and the generated query. The parser's messages name the exact
    # constant, measure or rule at fault, so they are shown as they are.
    try:
        generator, code, module_path = core.generate_module(spec_text, workdir)
    except Exception as exc:
        return _error_page(request, "The specification could not be turned into a query",
                           f"{type(exc).__name__}: {exc}", collapse=False)

    # Phase 2: Marabou. Failures here are about the network or the solver, not the spec,
    # so the headline stays generic and the detail sits under a toggle.
    try:
        outcome = core.solve_module(module_path, onnx_path)
        groups = core.build_report(outcome, generator)
    except Exception as exc:
        return _error_page(request, "An error occurred during Marabou verification",
                           f"{type(exc).__name__}: {exc}", collapse=True)

    run_id = uuid.uuid4().hex[:8]
    RUNS[run_id] = {
        "spec_name": spec.filename,
        "network_name": network.filename,
        "config": generator.config,
        "groups": groups,
        "code": code,
    }
    return RedirectResponse(f"/results/{run_id}", status_code=303)


@app.get("/results/{run_id}", response_class=HTMLResponse)
def results_page(request: Request, run_id: str):
    run = RUNS.get(run_id)
    if run is None:
        return RedirectResponse("/", status_code=303)
    counts = {v: sum(1 for g in run["groups"] if g["verdict"] == v) for v in VERDICTS}
    checks = sum(len(g["parts"]) for g in run["groups"])
    return templates.TemplateResponse(
        request, "results.html",
        {"run": run, "run_id": run_id, "counts": counts, "checks": checks},
    )
