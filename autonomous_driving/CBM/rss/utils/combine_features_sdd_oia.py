# Feature combination module (SDD-OIA)
"""Combine several extracted SDD-OIA feature folders into one, in the layout of the rsbench SDD-OIA embeddings.

Run from the rss/ folder, for example:
    python -m utils.combine_features_sdd_oia --inputs sddoia_dinov3_l_384 sddoia_pe_spatial_l_378 --out sddoia_dinov3_l_pe_spatial

--inputs and --out are folder names under data/. The inputs must have the same split folders with the same
feature files (written by utils/extract_features_sdd_oia.py). For every image, the input vectors are
concatenated in the given order, each folder scaled by one factor (see utils/combine_features_common.py).
The scene folders are copied from the first input, so concepts and actions are read as in SDD-OIA itself.
"""

import os
import shutil

from utils import combine_features_common as common
from utils.extract_features_sdd_oia import SPLITS, scenes_dir


def present_splits(folder):
    """Split folders of a feature folder, in SPLITS order"""
    return [s for s in SPLITS if os.path.isdir(os.path.join(folder, s))]


def split_files(folder, split):
    """Feature file names of a split"""
    return sorted(f for f in os.listdir(os.path.join(folder, split)) if f.endswith(".pt"))


def main():
    args = common.parse_args("Combine extracted SDD-OIA feature folders into one.")
    inputs, out, metas = common.prepare(args)
    splits = present_splits(inputs[0])
    for folder in inputs[1:]:
        if present_splits(folder) != splits:
            raise SystemExit(f"{folder}: split folders {present_splits(folder)} differ from {splits} in {inputs[0]}")
        for split in splits:
            if split_files(folder, split) != split_files(inputs[0], split):
                raise SystemExit(f"{folder}: the {split} feature files differ from {inputs[0]}")

    train_files = split_files(inputs[0], "train")
    scale, factors = common.scale_factors(
        args.inputs, [[os.path.join(f, "train", n) for n in train_files] for f in inputs])

    counts = {}
    for split in splits:
        names = split_files(inputs[0], split)
        os.makedirs(os.path.join(out, split), exist_ok=True)
        for i, name in enumerate(names):
            common.combine_file([os.path.join(f, split, name) for f in inputs], factors, os.path.join(out, split, name))
            if (i + 1) % 2000 == 0 or i + 1 == len(names):
                print(f"  {split}: {i + 1}/{len(names)}", flush=True)
        counts[split] = len(names)
    for scenes in sorted({scenes_dir(s) for s in splits}):
        shutil.copytree(os.path.join(inputs[0], scenes), os.path.join(out, scenes), dirs_exist_ok=True)
    common.write_meta(out, args.inputs, metas, scale, factors, counts)


if __name__ == "__main__":
    main()
