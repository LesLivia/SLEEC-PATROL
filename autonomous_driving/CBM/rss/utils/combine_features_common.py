# Common feature combination functions
"""Dataset-independent parts of the feature combination, used by combine_features_bdd_oia.py and
combine_features_sdd_oia.py.

For every image, the vectors of several feature folders are concatenated in the given order, each folder
multiplied by one factor that gives it the same mean absolute value as the first folder on the training split
(the first folder keeps factor 1). One constant per folder keeps all relative information; it only puts the
folders on the same scale, so a single learning rate suits all of them. The output folder must not exist
yet; the sources, factors and total "dim" are written to <out>/meta.json, last, so an interrupted run
never looks complete (delete its folder and combine again).
"""

import argparse
import datetime
import json
import os

import torch

DATA = "data"


def parse_args(description):
    """Parse the command line arguments shared by every dataset

    Returns:
        args: parsed arguments
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--inputs", nargs="+", required=True, help="feature folders under data/, e.g. dinov3_l_576 pe_spatial_l_574")
    parser.add_argument("--out", required=True, help="output folder under data/ (must not exist yet)")
    return parser.parse_args()


def prepare(args):
    """Check that the output folder does not exist yet and that the inputs are all complete extraction folders

    A combined folder is not accepted as an input: its source folders are listed instead, so every source
    is scaled against the first one.

    Returns:
        inputs (folder paths), out (folder path), metas (the inputs' meta.json)
    """
    if args.out in args.inputs:
        raise SystemExit("--out must differ from the --inputs folders")
    inputs = [os.path.join(DATA, f) for f in args.inputs]
    out = os.path.join(DATA, args.out)
    if os.path.exists(out):
        raise SystemExit(f"{out} already exists; delete it to combine again.")
    metas = []
    for folder in inputs:
        with open(os.path.join(folder, "meta.json")) as f:
            metas.append(json.load(f))
        if "sources" in metas[-1]:
            raise SystemExit(f"{folder} is a combined folder; list its source folders in --inputs instead: "
                             + " ".join(s["folder"] for s in metas[-1]["sources"]))
        if not metas[-1]["complete"]:
            raise SystemExit(f"{folder} is not complete (meta.json)")
    return inputs, out, metas


def mean_abs(paths):
    """Mean absolute feature value over the given feature files"""
    total, count = 0.0, 0
    for path in paths:
        x = torch.load(path)
        total += x.abs().sum().item()
        count += x.numel()
    return total / count


def scale_factors(names, train_paths):
    """One factor per input: mean |x| of the first input / mean |x| of this input, on its training files

    Args:
        names: input folder names (for the log)
        train_paths: for every input, the paths of its training feature files

    Returns:
        scale (mean |x| per input), factors
    """
    scale = [mean_abs(paths) for paths in train_paths]
    factors = [scale[0] / s for s in scale]
    for name, s, k in zip(names, scale, factors):
        print(f"{name}: mean |x| on train {s:.4f} -> factor {k:.6g}", flush=True)
    return scale, factors


def combine_file(sources, factors, dst):
    """Concatenate the scaled vectors of one image along the feature axis and write them to dst"""
    x = torch.cat([torch.load(path) * k for path, k in zip(sources, factors)], dim=-1)
    # write then rename, so an interrupted run never leaves a partial file behind
    torch.save(x, dst + ".tmp")
    os.replace(dst + ".tmp", dst)


def write_meta(out, names, metas, scale, factors, counts):
    """Write meta.json of the combined folder (the last step)"""
    dims = [m["dim"] for m in metas]
    meta = {
        "sources": [{"folder": f, "model": m["model"], "dim": d, "mean_abs_train": s, "factor": k}
                    for f, m, d, s, k in zip(names, metas, dims, scale, factors)],
        "dim": sum(dims),
        "layout": " | ".join(f"{f} x {k:.6g} ({d})" for f, d, k in zip(names, dims, factors)),
        "scaling": "one factor per source: mean |x| of the first source / mean |x| of this source, on the train split",
        "counts": counts,
        "complete": True,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"complete: {out} (dim {meta['dim']}, meta.json written)")
