"""Build the bundled data directory for the answers viewer.

For each environment, in the order the study shows them, reads the setup's stored
verification answer (webapp/examples/<setup>/prerun.json), builds its report with the
webapp's own code, and writes into ``webeval_viewer/public/data/``:

  <env>/rules/<rule>.html   one page fragment per VIOLATED rule, from templates/rule.html:
                            the rule's panel and its diagnosis dialogs
  <env>/background.txt      the environment's background, and note.txt where it has one
                            (each rule's plain-words text goes into the manifest)
  <env>/scene.svg, images/  its pictures: for Highway, an ordinary observation drawn by
                            the webapp's renderer; for SDD-OIA, training images
  manifest.json             the environments in order, each with its rules and pictures

The stored answer is used as it is. The viewer shows each answer beside this data, so it
has to be built from the same prerun answers and content as the study site: run this
whenever webeval/scripts/build_data.py is run.

Run from the repo root, in the environment the webapp runs in (on Windows, inside WSL,
where onnxruntime and the webapp's packages are installed)::

    python webeval_viewer/scripts/build_data.py
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnxruntime
from jinja2 import Environment, FileSystemLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from webapp import core, scenes  # noqa: E402
from webapp.app import examples  # noqa: E402

WEBEVAL = REPO_ROOT / "webeval"
CONTENT = WEBEVAL / "content"
DST_ROOT = REPO_ROOT / "webeval_viewer" / "public" / "data"
SDDOIA_DATA = REPO_ROOT / "autonomous_driving" / "CBM" / "rss" / "data" / "mini_boia_out"

# The environments in the order the study shows them: (setup folder, title).
ENVIRONMENTS = [("highway-fast-v0", "Highway"), ("sddoia", "SDD-OIA")]

# SDD-OIA labels name three green lights; they are always equal, and the model reads them
# as the one measure greenLight, so the other two have no measure of their own.
SDDOIA_SAME_AS_GREEN = {"leftGreenLight", "rightGreenLight"}


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def write_json(path: Path, obj) -> None:
    write_text(path, json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def read_prerun(path: Path) -> tuple[dict, str]:
    """The stored answer and when it was solved.

    The answer is put back in the form check_property() returns it, as
    core.load_prerun() does, but without its check that the answer is still current.
    """
    record = json.loads(path.read_text(encoding="utf-8"))
    outcome = record["outcome"]
    outcome["results"] = {
        name: (verified, {int(var): value for var, value in cex.items()}, exit_code)
        for name, (verified, cex, exit_code) in outcome["results"].items()}
    return outcome, record["solved"]


def in_order(names: dict) -> list[str]:
    """A generator's name -> index map, as names in declaration order."""
    return [name for name, _ in sorted(names.items(), key=lambda kv: kv[1])]


def highway_scene(item: dict, setup: dict, generator, env_dst: Path) -> dict:
    """Draw the recorded observation with the webapp's renderer.

    The renderer reads the network's outputs as well, so the network is run on the
    observation; nothing else of it is drawn. Every measure is marked as named by a rule,
    so every car is drawn in yellow, as a diagnosis draws the cars its rule names - which
    is how cars mostly appear there. The ego keeps its own colour.
    """
    measures = json.loads((CONTENT / setup["name"] / item["scene"])
                          .read_text(encoding="utf-8"))["measures"]
    inputs = in_order(generator.measures)
    session = onnxruntime.InferenceSession(str(setup["model"]))
    x = np.array([[measures[name] for name in inputs]], dtype=np.float32)
    logits = session.run(None, {session.get_inputs()[0].name: x})[0][0]
    rows = ([{"name": n, "raw": float(measures[n]), "related": True, "output": False}
             for n in inputs]
            + [{"name": n, "raw": float(v), "related": False, "output": True}
               for n, v in zip(in_order(generator.outputs), logits)])
    renderer = scenes.pick_renderer(generator.measures, generator.outputs)
    svg = scenes.render(renderer, rows, generator.constants)
    if svg is None:
        raise SystemExit(f"{setup['name']}: the renderer drew nothing for {item['scene']}")
    write_text(env_dst / "scene.svg", svg)
    return {"src": f"/data/{setup['name']}/scene.svg", "caption": item["caption"]}


def sddoia_image(item: dict, setup: dict, generator, env_dst: Path) -> dict:
    """Copy one training image, captioned with the concepts its label marks present.

    The labels name concepts in snake_case (red_light); the specification's measures are
    the same names in camelCase (redLight), listed here in declaration order.
    """
    scene = json.loads((SDDOIA_DATA / "scenes" / f"{item['image']}.json")
                       .read_text(encoding="utf-8"))
    filename = scene["image_filename"]
    (env_dst / "images").mkdir(parents=True, exist_ok=True)
    shutil.copy2(SDDOIA_DATA / scene["split"] / filename, env_dst / "images" / filename)

    present = {re.sub(r"_(\w)", lambda m: m.group(1).upper(), name)
               for name, on in scene["concepts"].items() if on}
    unknown = present - set(generator.measures) - SDDOIA_SAME_AS_GREEN
    if unknown:
        raise SystemExit(f"{item['image']}: no measure for {', '.join(sorted(unknown))}")
    concepts = [name for name in in_order(generator.measures) if name in present]
    return {"src": f"/data/{setup['name']}/images/{filename}",
            "caption": f"{item['title']}. Concepts present: {', '.join(concepts)}."}


def build_env(setup: dict, title: str, template) -> dict:
    env_dst = DST_ROOT / setup["name"]
    generator = core.build_generator(setup["spec"].read_text(encoding="utf-8"),
                                     output_semantics=setup["semantics"],
                                     decision_threshold=setup["threshold"])
    outcome, solved = read_prerun(setup["spec"].parent / core.PRERUN)
    groups = core.build_report(outcome, generator)

    rules = []
    for group in groups:
        if group["verdict"] != "VIOLATED":
            continue
        write_text(env_dst / "rules" / f"{group['name']}.html", template.render(group=group))
        rules.append(group["name"])

    content = CONTENT / setup["name"]
    write_text(env_dst / "background.txt",
               (content / "background.txt").read_text(encoding="utf-8"))
    note = content / "note.txt"
    if note.is_file():
        write_text(env_dst / "note.txt", note.read_text(encoding="utf-8"))

    pictures = json.loads((content / "pictures.json").read_text(encoding="utf-8"))
    items = [highway_scene(item, setup, generator, env_dst) if "scene" in item
             else sddoia_image(item, setup, generator, env_dst)
             for item in pictures["items"]]

    # Each shown rule in plain words, shown above its panel. A rule without one is shown
    # without it, and said here.
    words = json.loads((content / "rules.json").read_text(encoding="utf-8"))
    descriptions = {rule: words[rule] for rule in rules if rule in words}

    print(f"[ok] {title}: prerun solved {solved}, {len(rules)} of {len(groups)} rules "
          f"violated -> {', '.join(rules)}")
    for rule in rules:
        if rule not in descriptions:
            print(f"[warn] {title}: no plain-words text for {rule} in {content / 'rules.json'}")
    return {"id": setup["name"], "title": title, "solved": solved, "rules": rules,
            "descriptions": descriptions, "note": note.is_file(),
            "pictures": {"intro": pictures["intro"], "items": items}}


def main() -> None:
    setups = {setup["name"]: setup for setup in examples()}
    missing = [name for name, _ in ENVIRONMENTS if name not in setups]
    if missing:
        raise SystemExit(f"No bundled setup named {', '.join(missing)} in webapp/examples.")

    if DST_ROOT.exists():
        shutil.rmtree(DST_ROOT)
    DST_ROOT.mkdir(parents=True)

    # As the webapp renders its templates: Starlette's Jinja2Templates turns autoescape on.
    template = Environment(loader=FileSystemLoader(str(WEBEVAL / "templates")),
                           autoescape=True).get_template("rule.html")
    manifest = {
        "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environments": [build_env(setups[name], title, template)
                         for name, title in ENVIRONMENTS],
    }
    write_json(DST_ROOT / "manifest.json", manifest)
    print(f"\nBundled {len(manifest['environments'])} environments -> {DST_ROOT}")


if __name__ == "__main__":
    sys.exit(main())
