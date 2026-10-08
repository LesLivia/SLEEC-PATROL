from argparse import Namespace
from datasets.utils.base_dataset import BaseDataset, BOIA_get_loader
from datasets.utils.boia_creation import BOIADataset
from datasets.utils.sddoia_creation import CONCEPTS_ORDER
from backbones.boia_linear import BOIAConceptizer
from backbones.boia_mlp import BOIAMLP
from utils.losses import compute_concept_pos_weights
import json
import time


class BOIA(BaseDataset):
    NAME = "boia"

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)

    def _feature_dir(self):
        return f"data/{self.args.features}/"

    def get_data_loaders(self):
        start = time.time()

        image_dir = self._feature_dir()
        train_data_path = image_dir + "train_BDD_OIA.pkl"
        val_data_path = image_dir + "val_BDD_OIA.pkl"
        test_data_path = image_dir + "test_BDD_OIA.pkl"

        self.dataset_train = BOIADataset(
            pkl_file_path=train_data_path,
            use_attr=True,
            no_img=False,
            uncertain_label=False,
            image_dir=image_dir + "train",
            n_class_attr=2,
            transform=None,
            c_sup=self.args.c_sup,
            which_c=self.args.which_c,
        )
        self.dataset_val = BOIADataset(
            pkl_file_path=val_data_path,
            use_attr=True,
            no_img=False,
            uncertain_label=False,
            image_dir=image_dir + "val",
            n_class_attr=2,
            transform=None,
        )
        self.dataset_test = BOIADataset(
            pkl_file_path=test_data_path,
            use_attr=True,
            no_img=False,
            uncertain_label=False,
            image_dir=image_dir + "test",
            n_class_attr=2,
            transform=None,
        )

        # positive-label weights of the concept loss, measured on the training split
        c_weight = self.args.c_weight
        if c_weight != "none":
            train_concepts = [d["attribute_label"] for d in self.dataset_train.data]
            self.args.concept_pos_weight = compute_concept_pos_weights(
                train_concepts, c_weight, self.args.c_weight_cap
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

        self.train_loader = BOIA_get_loader(
            self.dataset_train, self.args.batch_size, val_test=False
        )
        self.val_loader = BOIA_get_loader(
            self.dataset_val, self.args.batch_size, val_test=True
        )
        self.test_loader = BOIA_get_loader(
            self.dataset_test, self.args.batch_size, val_test=True
        )

        return self.train_loader, self.val_loader, self.test_loader

    def get_backbone(self):
        if self.args.backbone == "neural":
            return BOIAMLP(), None

        # input size of the concept extractor: 2048 for the original rsbench features,
        # otherwise the "dim" written by utils/extract_features.py
        din = 2048
        if self.args.features != "bdd2048":
            with open(self._feature_dir() + "meta.json") as f:
                din = json.load(f)["dim"]
        return BOIAConceptizer(din=din, nconcept=21, hidden=self.args.concept_hidden), None

    def get_split(self):
        return 1, ()

    def get_concept_labels(self):
        sorted_concepts = sorted(CONCEPTS_ORDER, key=CONCEPTS_ORDER.get)
        return sorted_concepts

    def print_stats(self):
        print("## Statistics ##")
        print("Train samples", len(self.dataset_train))
        print("Validation samples", len(self.dataset_val))
        print("Test samples", len(self.dataset_test))
