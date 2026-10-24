import glob
import json
import os
import time
from argparse import Namespace

import torch
from datasets.utils.base_dataset import BaseDataset, BOIA_get_loader
from datasets.utils.presddoia_creation import PreSDDOIADataset, PRESDDOIA_CONCEPTS_ORDER
from backbones.conceptizer import Conceptizer
from utils.losses import compute_concept_pos_weights


class PRESDDOIA(BaseDataset):
    """SDD-OIA on precomputed image features (--features), trained like BDD-OIA (datasets/boia.py)"""

    NAME = "presddoia"

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        if args.task != "sddoia":
            raise SystemExit("--dataset presddoia needs --task sddoia (SDD-OIA actions: [stop, forward, left, right])")

    def _feature_dir(self):
        feature_dir = f"data/{self.args.features}/"
        if not os.path.isdir(feature_dir + "scenes"):
            raise SystemExit(
                f"--features {self.args.features}: {feature_dir} is not an SDD-OIA feature folder (no scenes/), "
                "e.g. --features sddoia_dinov3_l_384"
            )
        return feature_dir

    def get_data_loaders(self):
        start = time.time()

        feature_dir = self._feature_dir()
        self.dataset_train = PreSDDOIADataset(
            feature_dir, "train", c_sup=self.args.c_sup, which_c=self.args.which_c
        )
        self.dataset_val = PreSDDOIADataset(feature_dir, "val")
        self.dataset_test = PreSDDOIADataset(feature_dir, "test")
        # out-of-distribution split: concept combinations never seen in training
        self.dataset_ood = PreSDDOIADataset(feature_dir, "ood")

        # positive-label weights of the concept loss, measured on the training split
        c_weight = self.args.c_weight
        if c_weight != "none":
            self.args.concept_pos_weight = compute_concept_pos_weights(
                self.dataset_train.concepts, c_weight, self.args.c_weight_cap
            )
            print(f"Concept weights ({c_weight}):", [round(w, 2) for w in self.args.concept_pos_weight.tolist()])
        else:
            self.args.concept_pos_weight = None

        print(f"Loaded datasets in {time.time()-start} s.")

        print(
            "Len loaders: \n train:",
            len(self.dataset_train),
            "\n val:",
            len(self.dataset_val),
        )
        print(" len test:", len(self.dataset_test))
        print(" len ood:", len(self.dataset_ood))

        self.train_loader = BOIA_get_loader(
            self.dataset_train, self.args.batch_size, val_test=False
        )
        self.val_loader = BOIA_get_loader(
            self.dataset_val, self.args.batch_size, val_test=True
        )
        self.test_loader = BOIA_get_loader(
            self.dataset_test, self.args.batch_size, val_test=True
        )
        self.ood_loader = BOIA_get_loader(
            self.dataset_ood, self.args.batch_size, val_test=True
        )

        return self.train_loader, self.val_loader, self.test_loader

    def get_backbone(self):
        if self.args.backbone == "neural":
            raise SystemExit("--backbone neural is not available for presddoia")

        # input size of the concept extractor: the "dim" written by utils/extract_features_sdd_oia.py
        # (or combine_features_sdd_oia.py), otherwise the size of a feature file (rsbench SDD-OIA embeddings)
        feature_dir = self._feature_dir()
        if os.path.exists(feature_dir + "meta.json"):
            with open(feature_dir + "meta.json") as f:
                din = json.load(f)["dim"]
        else:
            din = torch.load(sorted(glob.glob(feature_dir + "train/*.pt"))[0]).numel()
        return Conceptizer(din=din, nconcept=len(PRESDDOIA_CONCEPTS_ORDER), hidden=self.args.concept_hidden), None

    def get_split(self):
        return 1, ()

    def get_concept_labels(self):
        sorted_concepts = sorted(PRESDDOIA_CONCEPTS_ORDER, key=PRESDDOIA_CONCEPTS_ORDER.get)
        return sorted_concepts

    def print_stats(self):
        print("## Statistics ##")
        print("Train samples", len(self.dataset_train))
        print("Validation samples", len(self.dataset_val))
        print("Test samples", len(self.dataset_test))
        print("Test OOD samples", len(self.dataset_ood))
