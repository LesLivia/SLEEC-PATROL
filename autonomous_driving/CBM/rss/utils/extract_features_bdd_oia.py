# Feature extraction module (BDD-OIA)
"""Extract frozen vision-backbone features for BDD-OIA, in the same layout as data/bdd2048.

Run from the rss/ folder, for example:
    python -m utils.extract_features_bdd_oia --model vit_large_patch16_dinov3.lvd1689m --out data/dinov3_l_576 --height 576 --width 1024

For every frame listed in the rsbench split files, the full 1280x720 image (no crop) is resized to
--height x --width and passed through the timm model. The output tokens are pooled into
    [global token | mean of each grid cell | max of each grid cell]
and saved as <out>/<split>/inputs/<frame>.pt, a (1, D) float32 tensor like the original
data/bdd2048 inputs. labels/, concepts/ and the split .pkl files are copied from --src, and the
settings are written to <out>/meta.json, whose "dim" sets the concept extractor's input size
(datasets/boia.py). Existing outputs are skipped, so an interrupted run can be resumed with the same
settings; a folder that already has meta.json (a finished extraction) is refused.
The dataset-independent parts are in utils/extract_features_common.py.
"""

import argparse
import os
import pickle

from utils import extract_features_common as common

SPLITS = ["train", "val", "test"]


def parse_args():
    """Parse the command line arguments

    Returns:
        args: parsed arguments
    """
    parser = argparse.ArgumentParser(description="Extract BDD-OIA image features with a timm backbone.")
    parser.add_argument("--out", required=True, help="output folder, e.g. data/dinov3_l_576")
    parser.add_argument("--src", default="data/bdd2048", help="folder with the split .pkl files, labels/ and concepts/")
    parser.add_argument("--images", default="data/BDD-OIA-lastframe/lastframe/data", help="folder with the original .jpg frames")
    parser.add_argument("--height", type=int, default=576, help="input height, a multiple of the patch size")
    parser.add_argument("--width", type=int, default=1024, help="input width, a multiple of the patch size")
    common.add_common_args(parser)
    return parser.parse_args()


def frame_path(root, split, sub, name):
    """Path of a per-frame .pt file, named like the rsbench files (<frame>.pt)"""
    return os.path.join(root, split, sub, name[:-4] + ".pt")


def mirror_split(args, split):
    """Write the split .pkl and copy labels/ and concepts/ from --src

    Returns:
        names: frame file names of the split (first --limit if given)
    """
    entries = pickle.load(open(os.path.join(args.src, f"{split}_BDD_OIA.pkl"), "rb"))
    if args.limit:
        entries = entries[: args.limit]
    names = [e["img_path"] for e in entries]
    for sub in ["inputs", "labels", "concepts"]:
        os.makedirs(os.path.join(args.out, split, sub), exist_ok=True)
    with open(os.path.join(args.out, f"{split}_BDD_OIA.pkl"), "wb") as f:
        pickle.dump(entries, f)
    for name in names:
        for sub in ["labels", "concepts"]:
            dst = frame_path(args.out, split, sub, name)
            if not os.path.exists(dst):
                common.copy_file(frame_path(args.src, split, sub, name), dst)
    return names


def verify(args, split_names, model, info):
    """Check counts, shapes, finiteness and reproducibility of the written features

    Returns:
        problems (list of str), max abs difference of recomputed features
    """
    problems = []
    for split, names in split_names.items():
        for sub in ["inputs", "labels", "concepts"]:
            missing = sum(not os.path.exists(frame_path(args.out, split, sub, n)) for n in names)
            if missing:
                problems.append(f"{split}/{sub}: {missing} of {len(names)} files missing")
    all_inputs = [frame_path(args.out, s, "inputs", n) for s, ns in split_names.items() for n in ns]
    problems += common.check_tensors(all_inputs, (1, info["dim"]))
    names = split_names["train"][:4]
    max_diff = common.recompute_max_diff(model, info, args, names, args.images,
                                         [frame_path(args.out, "train", "inputs", n) for n in names])
    return problems, max_diff


def main():
    args = parse_args()
    rows, cols = (int(v) for v in args.grid.lower().split("x"))
    common.check_settings(args.out, {"model": args.model, "height": args.height, "width": args.width,
                                     "grid": [rows, cols], "src": args.src, "images": args.images})

    print(f"Loading {args.model} (the first run downloads the weights) ...", flush=True)
    model, data_cfg, timm_version = common.load_model(args.model)
    info = common.backbone_info(model, data_cfg, args.height, args.width, (rows, cols))

    split_names = {}
    for split in SPLITS:
        names = mirror_split(args, split)
        split_names[split] = names
        common.extract_to_files(model, info, args, split, names, args.images,
                                lambda n, split=split: frame_path(args.out, split, "inputs", n), keep_batch_dim=True)

    problems, max_diff = verify(args, split_names, model, info)
    complete = not problems and args.limit is None
    meta = common.base_meta(args, info, timm_version)
    meta.update({
        "source_images": args.images,
        "source_labels": args.src,
        "limit": args.limit,
        "counts": {s: len(n) for s, n in split_names.items()},
        "recompute_max_abs_diff": max_diff,
        "complete": complete,
    })
    common.write_meta(args.out, meta)

    print(f"recomputed features max abs difference: {max_diff:.2e}")
    for p in problems:
        print("PROBLEM:", p)
    print(f"{'complete' if complete else 'NOT complete'}: {args.out} (meta.json written)")


if __name__ == "__main__":
    main()
