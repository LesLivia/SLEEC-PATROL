"""Minimal FastAPI UI for SLEEC -> Marabou verification.

Page 1 (GET /)           upload a .sleec specification and an .onnx network.
POST /verify             parse, generate the Marabou code, solve every rule.
Page 2 (GET /results/x)  one row per normalized rule: VERIFIED / VIOLATED / UNKNOWN,
                         with the counter-example expandable under a violated rule.

All verification logic lives in core.py. Uploads go to a temp folder; nothing is
written into the repository and nothing is cached between runs.

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

from webapp.core import run_verification

app = FastAPI(title="SLEEC2Ver")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# Single-user local tool: finished runs live in memory, keyed by a short id.
RUNS = {}


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

    try:
        rows, code, generator = run_verification(spec_text, onnx_path, workdir)
    except Exception as exc:  # bad spec, bad ONNX, solver failure: report, don't crash
        return templates.TemplateResponse(
            request, "upload.html",
            {"error": f"{type(exc).__name__}: {exc}"},
            status_code=400,
        )

    run_id = uuid.uuid4().hex[:8]
    RUNS[run_id] = {
        "spec_name": spec.filename,
        "network_name": network.filename,
        "config": generator.config,
        "rules": rows,
        "code": code,
    }
    return RedirectResponse(f"/results/{run_id}", status_code=303)


@app.get("/results/{run_id}", response_class=HTMLResponse)
def results_page(request: Request, run_id: str):
    run = RUNS.get(run_id)
    if run is None:
        return RedirectResponse("/", status_code=303)
    counts = {s: sum(1 for r in run["rules"] if r["status"].startswith(s))
              for s in ("VERIFIED", "VIOLATED", "UNKNOWN")}
    return templates.TemplateResponse(
        request, "results.html", {"run": run, "run_id": run_id, "counts": counts}
    )
