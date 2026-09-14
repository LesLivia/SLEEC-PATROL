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

MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved_models", "highway_ppo_model")
OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "verifier", "resources", "models", "highway_ppo.onnx"
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
    if not os.path.exists(MODEL_PATH + ".zip"):
        raise FileNotFoundError(
            f"No trained model found at {MODEL_PATH}.zip\n"
            "Run train_highway_ppo.py first."
        )

    model = PPO.load(MODEL_PATH, device="cpu")
    actor = PPOActorNet(model.policy)
    actor.eval()

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
