# Feature combination module (BDD-OIA)
"""Combine several extracted BDD-OIA feature folders into one, in the same layout as data/bdd2048.

Run from the rss/ folder, for example:
    python -m utils.combine_features_bdd_oia --inputs dinov3_l_576 pe_spatial_l_574 --out dinov3_l_576_pe_spatial_l_574

--inputs and --out are folder names under data/, as for --features. The inputs must list the same frames
in the same order. For every frame, the input vectors are concatenated in the given order, each folder
scaled by one factor (see utils/combine_features_common.py). labels/, concepts/ and the split .pkl files are
copied from the first folder, and <out>/meta.json gives the total "dim" read by datasets/boia.py.
"""

import os
import pickle
import shutil

from utils import combine_features_common as common
from utils.extract_features_bdd_oia import SPLITS, frame_path


def load_split(folder, split):
    """Entries of a split .pkl file"""
    with open(os.path.join(folder, f"{split}_BDD_OIA.pkl"), "rb") as f:
        return pickle.load(f)


def main():
    args = common.parse_args("Combine extracted BDD-OIA feature folders into one.")
    inputs, out, metas = common.prepare(args)
    for split in SPLITS:
        first = load_split(inputs[0], split)
        for folder in inputs[1:]:
            if load_split(folder, split) != first:
                raise SystemExit(f"{folder}: the {split} split differs from {inputs[0]}")

    train_names = [e["img_path"] for e in load_split(inputs[0], "train")]
    scale, factors = common.scale_factors(
        args.inputs, [[frame_path(f, "train", "inputs", n) for n in train_names] for f in inputs])

    counts = {}
    for split in SPLITS:
        names = [e["img_path"] for e in load_split(inputs[0], split)]
        for sub in ["inputs", "labels", "concepts"]:
            os.makedirs(os.path.join(out, split, sub), exist_ok=True)
        shutil.copyfile(os.path.join(inputs[0], f"{split}_BDD_OIA.pkl"), os.path.join(out, f"{split}_BDD_OIA.pkl"))
        for i, name in enumerate(names):
            common.combine_file([frame_path(f, split, "inputs", name) for f in inputs], factors,
                                frame_path(out, split, "inputs", name))
            for sub in ["labels", "concepts"]:
                shutil.copyfile(frame_path(inputs[0], split, sub, name), frame_path(out, split, sub, name))
            if (i + 1) % 2000 == 0 or i + 1 == len(names):
                print(f"  {split}: {i + 1}/{len(names)}", flush=True)
        counts[split] = len(names)
    common.write_meta(out, args.inputs, metas, scale, factors, counts)


if __name__ == "__main__":
    main()
