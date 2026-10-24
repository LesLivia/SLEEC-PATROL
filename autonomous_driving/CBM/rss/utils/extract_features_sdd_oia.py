# Feature extraction module (SDD-OIA)
"""Extract frozen vision-backbone features for SDD-OIA, in the layout of the rsbench SDD-OIA embeddings.

Run from the rss/ folder, for example:
    python -m utils.extract_features_sdd_oia --model vit_large_patch16_dinov3.lvd1689m --out data/sddoia_dinov3_l_384 --height 384 --width 464

For every image in the split folders of --src (train, val, test, ood, and ood_ambulance if present), the full
image (no crop) is resized to --height x --width and passed through the timm model. The output tokens are
pooled into
    [global token | mean of each grid cell | max of each grid cell]
and saved as <out>/<split>/<image>.pt, a (D,) float32 tensor like the rsbench SDD-OIA embeddings. The scene
file of every image (scenes/, or scenes_ambulance/ for ood_ambulance) is copied, so concepts and actions are
read from it as in SDD-OIA itself: concepts by name, label = [stop, forward, left, right]. The settings are
written to <out>/meta.json. Existing outputs are skipped, so an interrupted run can be resumed with the same
settings; a folder that already has meta.json (a finished extraction) is refused.
The dataset-independent parts are in utils/extract_features_common.py.
"""

import argparse
import os

from utils import extract_features_common as common

SPLITS = ["train", "val", "test", "ood", "ood_ambulance"]


def parse_args():
    """Parse the command line arguments

    Returns:
        args: parsed arguments
    """
    parser = argparse.ArgumentParser(description="Extract SDD-OIA image features with a timm backbone.")
    parser.add_argument("--out", required=True, help="output folder, e.g. data/sddoia_dinov3_l_384")
    parser.add_argument("--src", default="data/mini_boia_out", help="SDD-OIA folder with the split folders and scenes/")
    parser.add_argument("--height", type=int, default=384, help="input height, a multiple of the patch size")
    parser.add_argument("--width", type=int, default=464, help="input width, a multiple of the patch size")
    common.add_common_args(parser)
    return parser.parse_args()


def scenes_dir(split):
    """Folder of the scene files of a split"""
    return "scenes_ambulance" if split == "ood_ambulance" else "scenes"


def feature_path(root, split, name):
    """Path of an image's feature file, named like the image (<image>.pt)"""
    return os.path.join(root, split, os.path.splitext(name)[0] + ".pt")


def scene_path(root, split, name):
    """Path of an image's scene file"""
    return os.path.join(root, scenes_dir(split), os.path.splitext(name)[0] + ".json")


def list_split(args, split):
    """Create the split's output folders and copy the scene files of its images

    Returns:
        names: image file names of the split, by image number (first --limit if given)
    """
    names = sorted((f for f in os.listdir(os.path.join(args.src, split)) if f.endswith(".png")),
                   key=lambda f: int(os.path.splitext(f)[0].split("_")[-1]))
    if args.limit:
        names = names[: args.limit]
    os.makedirs(os.path.join(args.out, split), exist_ok=True)
    os.makedirs(os.path.join(args.out, scenes_dir(split)), exist_ok=True)
    for name in names:
        dst = scene_path(args.out, split, name)
        if not os.path.exists(dst):
            common.copy_file(scene_path(args.src, split, name), dst)
    return names


def verify(args, split_names, model, info):
    """Check counts, shapes, finiteness and reproducibility of the written features

    Returns:
        problems (list of str), max abs difference of recomputed features
    """
    problems = []
    for split, names in split_names.items():
        for kind, path_of in [("features", feature_path), ("scenes", scene_path)]:
            missing = sum(not os.path.exists(path_of(args.out, split, n)) for n in names)
            if missing:
                problems.append(f"{split}: {missing} of {len(names)} {kind} files missing")
    all_features = [feature_path(args.out, s, n) for s, ns in split_names.items() for n in ns]
    problems += common.check_tensors(all_features, (info["dim"],))
    names = split_names["train"][:4]
    max_diff = common.recompute_max_diff(model, info, args, names, os.path.join(args.src, "train"),
                                         [feature_path(args.out, "train", n) for n in names])
    return problems, max_diff


def main():
    args = parse_args()
    rows, cols = (int(v) for v in args.grid.lower().split("x"))
    common.check_settings(args.out, {"model": args.model, "height": args.height, "width": args.width,
                                     "grid": [rows, cols], "src": args.src})

    print(f"Loading {args.model} (the first run downloads the weights) ...", flush=True)
    model, data_cfg, timm_version = common.load_model(args.model)
    info = common.backbone_info(model, data_cfg, args.height, args.width, (rows, cols))

    split_names = {}
    for split in [s for s in SPLITS if os.path.isdir(os.path.join(args.src, s))]:
        names = list_split(args, split)
        split_names[split] = names
        common.extract_to_files(model, info, args, split, names, os.path.join(args.src, split),
                                lambda n, split=split: feature_path(args.out, split, n), keep_batch_dim=False)

    problems, max_diff = verify(args, split_names, model, info)
    complete = not problems and args.limit is None
    meta = common.base_meta(args, info, timm_version)
    meta.update({
        "source": args.src,
        "labels": "scenes/<image>.json (scenes_ambulance/ for ood_ambulance): concepts by name, "
                  "label = [stop, forward, left, right]",
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
