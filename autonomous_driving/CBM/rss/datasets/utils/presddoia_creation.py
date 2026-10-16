import glob
import json
import os

import torch
import torch.utils.data
from torch.utils.data import Dataset
from datasets.utils.mnist_creation import generate_r_seq
from datasets.utils.sddoia_creation import CONCEPTS_ORDER

# rsbench's SDD-OIA generator (rssgen/rssgen/sddoia/sddoia.py, deal_with_tl and deal_with_left_right_tl) sets
# left_green_light and right_green_light from the same traffic light as green_light: they are copies of it
GREEN_LIGHT_COPIES = ["left_green_light", "right_green_light"]

# the 19 distinct SDD-OIA concepts: name -> position, in the CONCEPTS_ORDER order, renumbered 0-18
PRESDDOIA_CONCEPTS_ORDER = {
    c: i
    for i, c in enumerate(
        c for c in sorted(CONCEPTS_ORDER, key=CONCEPTS_ORDER.get) if c not in GREEN_LIGHT_COPIES
    )
}


class PreSDDOIADataset(Dataset):
    """
    Returns a compatible Torch Dataset object for SDD-OIA on precomputed image features,
    loaded like BOIADataset (datasets/utils/boia_creation.py) loads BDD-OIA, with the 19 distinct
    concepts of PRESDDOIA_CONCEPTS_ORDER
    """

    def __init__(self, feature_dir, split, c_sup=1, which_c=[-1]):
        """
        Arguments:
        feature_dir: feature folder with <split>/<image>.pt and scenes/<image>.json, written by
            utils/extract_features_sdd_oia.py (or combine_features_sdd_oia.py), or the rsbench SDD-OIA embeddings
        split: train, val, test or ood
        c_sup: fraction of training images with concept supervision, as in BOIADataset
        which_c: concepts that keep their supervision, as in BOIADataset
        """
        files = glob.glob(os.path.join(feature_dir, split, "*.pt"))
        # by image number: MINI_BOIA_<number>.pt
        self.files = sorted(files, key=lambda f: int(os.path.basename(f)[:-3].split("_")[-1]))
        self.names = [os.path.basename(f)[:-3] for f in self.files]

        # concepts and actions in the SDD-OIA order, read from each image's scene file:
        # concepts by name, positions from PRESDDOIA_CONCEPTS_ORDER; label = [stop, forward, left, right]
        sorted_concepts = sorted(PRESDDOIA_CONCEPTS_ORDER, key=PRESDDOIA_CONCEPTS_ORDER.get)
        labels, concepts = [], []
        for name in self.names:
            with open(os.path.join(feature_dir, "scenes", name + ".json")) as f:
                scene = json.load(f)
            values = scene["concepts"]
            if any(values[c] != values["green_light"] for c in GREEN_LIGHT_COPIES):
                raise ValueError(
                    f"{name}: left/right green light differ from green_light, so they are not copies "
                    "and the 19 concepts of PRESDDOIA_CONCEPTS_ORDER would lose information"
                )
            labels.append(scene["label"])
            concepts.append([int(values[k]) for k in sorted_concepts])
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.concepts = torch.tensor(concepts, dtype=torch.float32)

        self.r_seq = generate_r_seq(len(self.files))
        self.c_sup = c_sup
        self.which_c = which_c

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = torch.load(self.files[idx])
        class_label = self.labels[idx]
        attr_label = self.concepts[idx].clone()

        # filter for the concept supervision given, as in BOIADataset
        if self.c_sup != 1:
            if self.r_seq[idx] > self.c_sup:
                attr_label[:] = -1
            elif not (self.which_c[0] == -1):
                for c in range(attr_label.shape[0]):
                    if c not in self.which_c:
                        attr_label[c] = -1
            else:
                # same list as BOIADataset; the two green-light copies are not in PRESDDOIA_CONCEPTS_ORDER
                for k, order in PRESDDOIA_CONCEPTS_ORDER.items():
                    if k not in [
                        "red_light",
                        "green_light",
                        "car",
                        "person",
                        "rider",
                        "other_obstacle",
                        "stop_sign",
                        "right_green_light",
                        "left_green_light",
                    ]:
                        attr_label[order] = -1

        return img, class_label, attr_label
