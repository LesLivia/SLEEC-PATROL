# Head export module (SDD-OIA)
"""Export the action layer of a trained SDD-OIA CBM (presddoiacbm) to ONNX for Marabou verification.

Run from the rss/ folder, for example:
    python -m utils.export_head_sdd_oia --checkpoint data/ckpts/presddoia-presddoiacbm-dis-42-sddoia_dinov3_l_pe_spatial_h4096_cw_sqrt-end.pt

The model is a concept bottleneck: image features -> 19 concepts -> Linear(19, 4) -> Sigmoid
(models/utils/oia_cbm.py). Only the action layer is exported:
  Input:  (1, 19) the concepts, in the order of PRESDDOIA_CONCEPTS_ORDER (datasets/utils/presddoia_creation.py)
  Output: (1, 4)  one logit per action, in the SDD-OIA label order [stop, forward, left, right]
cbm_inference returns [1 - sigma, sigma] per action and training reads index 1 as label 1 (SDDOIA_CE), so sigma
is the probability that the action is allowed. The layer is exported without its sigmoid:
output i > 0 <=> P(action i allowed) > 0.5, and the outputs are ordered like these probabilities.
Before the file is written, the exported output, with the sigmoid put back, is checked to be the model's own
output on every 0/1 concept vector (which includes the concepts of every SDD-OIA scene) and on random concept
vectors in [0, 1].
"""

import argparse
import os

import torch
import torch.nn as nn

from datasets.utils.presddoia_creation import PRESDDOIA_CONCEPTS_ORDER

# action order of the SDD-OIA scene labels (datasets/utils/presddoia_creation.py)
ACTIONS = ["stop", "forward", "left", "right"]

OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "verifier", "resources", "models", "sddoia_cbm_head.onnx"
)


def parse_args():
    """Parse the command line arguments

    Returns:
        args: parsed arguments
    """
    parser = argparse.ArgumentParser(description="Export the action layer of a trained SDD-OIA CBM to ONNX.")
    parser.add_argument("--checkpoint", required=True,
                        help="weights of a trained presddoiacbm model, e.g. data/ckpts/<run>-end.pt")
    parser.add_argument("--out", default=OUTPUT_PATH,
                        help="ONNX file to write (default: verifier/resources/models/sddoia_cbm_head.onnx)")
    return parser.parse_args()


class SDDOIAHeadNet(nn.Module):
    """The action layer of a presddoiacbm model without its sigmoid: output i > 0 <=> action i allowed"""

    def __init__(self, weight, bias):
        super().__init__()
        # classifier.0 of OIACBM: Linear(19, 4), whose sigmoid is P(action allowed)
        self.action_net = nn.Linear(weight.shape[1], weight.shape[0])
        with torch.no_grad():
            self.action_net.weight.copy_(weight)
            self.action_net.bias.copy_(bias)

    def forward(self, concepts: torch.Tensor) -> torch.Tensor:
        return self.action_net(concepts)


def check(weight, bias, head):
    """The exported output, with the sigmoid put back, must be the model's own output

    original: sigmoid(c @ W.T + b), the action layer as trained (P(action allowed));
    exported: the same layer without its sigmoid, so sigmoid(output) must give the original back

    Returns:
        problems (list of str)
    """
    n = weight.shape[1]
    vectors = {
        "0/1 concept vectors": ((torch.arange(2 ** n)[:, None] >> torch.arange(n)) & 1).float(),
        "random concept vectors in [0, 1]": torch.rand(100000, n, generator=torch.Generator().manual_seed(0)),
    }
    problems = []
    with torch.no_grad():
        for kind, c in vectors.items():
            original = torch.sigmoid(c @ weight.T + bias)
            restored = torch.sigmoid(head(c))
            diff = (restored - original).abs().max().item()
            if diff > 1e-6:
                problems.append(f"{kind}: the exported output differs from the original by up to {diff:.2e}")
            print(f"Checked {len(c)} {kind}: max difference {diff:.2e}")
    return problems


def main():
    args = parse_args()
    concepts = sorted(PRESDDOIA_CONCEPTS_ORDER, key=PRESDDOIA_CONCEPTS_ORDER.get)

    state = torch.load(args.checkpoint, map_location="cpu")
    weight, bias = state["classifier.0.weight"], state["classifier.0.bias"]
    if tuple(weight.shape) != (len(ACTIONS), len(concepts)):
        raise SystemExit(
            f"{args.checkpoint}: action layer is {tuple(weight.shape[::-1])} -> not a presddoiacbm model "
            f"({len(concepts)} concepts -> {len(ACTIONS)} actions)"
        )
    head = SDDOIAHeadNet(weight, bias)
    head.eval()
    print(f"Exporting the action layer of {args.checkpoint}")

    dummy_input = torch.zeros(1, len(concepts))

    # Verify forward pass shape and output before export
    with torch.no_grad():
        out = head(dummy_input)
    assert out.shape == (1, len(ACTIONS)), f"Unexpected output shape: {out.shape}"
    problems = check(weight, bias, head)
    for p in problems:
        print("PROBLEM:", p)
    if problems:
        raise SystemExit("Not exported: the exported layer does not reproduce the model's output")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.onnx.export(
        head,
        dummy_input,
        args.out,
        input_names=["concepts"],
        output_names=["action_logits"],
        opset_version=11,
        verbose=False,
    )

    print(f"Exported action layer to: {os.path.normpath(args.out)}")
    print(f"Network: [{len(concepts)} -> {len(ACTIONS)}]  (linear, output i > 0 <=> P(action i allowed) > 0.5)")
    print("Input indices:  " + "  ".join(f"{i}={c}" for i, c in enumerate(concepts)))
    print("Output indices: " + "  ".join(f"{i}={a}" for i, a in enumerate(ACTIONS)))


if __name__ == "__main__":
    main()
