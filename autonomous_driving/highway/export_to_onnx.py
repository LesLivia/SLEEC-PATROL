"""
Export the PPO actor network to ONNX format for Marabou verification.

The PPO policy is an actor-critic architecture:
  - Actor (policy_net + action_net): observation → action logits   [used at inference]
  - Critic (value_net):              observation → state value      [training only, discarded]

This script extracts only the actor and exports it as a standalone ONNX model:
  Input:  (1, 25)  — 5 vehicles × 5 features, flattened
  Output: (1, 5)   — action logits (argmax = selected action)
  Actions: 0=LANE_LEFT, 1=IDLE, 2=LANE_RIGHT, 3=FASTER, 4=SLOWER
"""

import os
import torch
import torch.nn as nn
from stable_baselines3 import PPO

# ============================================================================
# Choose what to export. These are independent of train_highway_ppo.py, so the
# training script's ENV_ID does not have to match.
#
#   highway-v0        ENV_ID = "highway-v0"       MODEL_SUBDIR = "highway-v0"
#   highway-fast-v0   ENV_ID = "highway-fast-v0"  MODEL_SUBDIR = "highway-fast-v0"
#
# MODEL_NAME picks the file inside saved_models/<MODEL_SUBDIR>/:
#   "best_model"                 the better of last/best on the 50-episode comparison (default)
#   "highway_ppo_model"          the policy at the last training step
#   "best_model_evalcallback"    EvalCallback's pick, kept only when it was superseded
#   "highway_ppo_<N>_steps"      a specific checkpoint, e.g. "highway_ppo_49992_steps"
# ============================================================================
# ENV_ID = "highway-fast-v0"
ENV_ID = "highway-v0"
MODEL_SUBDIR = ENV_ID  # use f"{ENV_ID}-scaled" only once the input scaling is folded in
MODEL_NAME = "best_model"

MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models", MODEL_SUBDIR)
EXPORT_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "verifier", "resources", "models", f"ppo_{ENV_ID}.onnx"
)

OBS_DIM = 25  # 5 vehicles × 5 features (presence, x, y, vx, vy), row-major flattened
N_ACTIONS = 5


class PPOActorNet(nn.Module):
    """Wraps only the actor half of an SB3 ActorCriticPolicy."""

    def __init__(self, policy):
        super().__init__()
        # mlp_extractor.policy_net: Linear(25,256) → ReLU → Linear(256,256) → ReLU
        self.policy_net = policy.mlp_extractor.policy_net
        # action_net: Linear(256, 5)  — outputs raw logits, no softmax
        self.action_net = policy.action_net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.action_net(self.policy_net(x))


if __name__ == "__main__":
    if MODEL_SUBDIR.endswith("-scaled"):
        raise ValueError(
            "This model expects scaled inputs (OBS_SCALE = 1, 100, 10, 30, 10 per feature). "
            "Scale the inputs in the Marabou verifier first: divide every bound set on input i "
            "by OBS_SCALE[i % 5] and multiply counter-example values back for reporting; the "
            ".sleec file itself stays unchanged. Remove this check once the parser does that."
        )

    if not os.path.exists(EXPORT_PATH + ".zip"):
        available = sorted(f[:-4] for f in os.listdir(MODEL_DIR)) if os.path.isdir(MODEL_DIR) else []
        raise FileNotFoundError(
            f"No model at {EXPORT_PATH}.zip\n"
            f"Available in {MODEL_DIR}: {available or 'nothing - run train_highway_ppo.py first'}"
        )

    model = PPO.load(EXPORT_PATH, device="cpu")
    activations = {type(m).__name__ for m in model.policy.mlp_extractor.policy_net
                   if not isinstance(m, nn.Linear)}
    assert activations == {"ReLU"}, (
        f"Actor uses {activations}; Marabou needs a ReLU network. Retrain with activation_fn=nn.ReLU."
    )
    actor = PPOActorNet(model.policy)
    actor.eval()
    print(f"Exporting model trained on {ENV_ID}: {EXPORT_PATH}.zip")

    dummy_input = torch.zeros(1, OBS_DIM)

    # Verify forward pass shape before export
    with torch.no_grad():
        out = actor(dummy_input)
    assert out.shape == (1, N_ACTIONS), f"Unexpected output shape: {out.shape}"

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    torch.onnx.export(
        actor,
        dummy_input,
        OUTPUT_PATH,
        input_names=["observation"],
        output_names=["action_logits"],
        opset_version=11,
        verbose=False,
    )

    print(f"Exported actor to: {OUTPUT_PATH}")
    print(f"Network: [{OBS_DIM} → 256 → 256 → {N_ACTIONS}]  (ReLU, linear output)")
    print("Action indices: 0=LANE_LEFT  1=IDLE  2=LANE_RIGHT  3=FASTER  4=SLOWER")
