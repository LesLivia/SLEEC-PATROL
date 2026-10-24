"""Record one ordinary highway-fast-v0 observation, for the Highway intro picture.

The picture on the Highway intro page is drawn by the webapp's own renderer
(webapp/scenes/highway.py) from a real observation rather than from a counter-example, so
it shows the road as the network normally sees it. This records that observation once:
the environment is reset with a fixed seed and driven with IDLE for a few decisions, and
the 25 values are written in the specification's measure order.

Run once from the repo root, with a Python that has highway-env (on this machine, the
Windows anaconda one; the WSL env does not have it):

    python webeval/scripts/capture_highway_scene.py

The seed and the number of steps were picked by hand for an ordinary moment: the ego in the
middle lane and the four nearest cars spread over the three lanes, none close in the ego's
own lane, so no rule applies; and no two cars in neighbouring lanes nearly level, where the
renderer's labels would overlap.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import gymnasium as gym
import highway_env  # noqa: F401  registers the environments

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from webapp.scenes.highway import MEASURES  # noqa: E402

DST = REPO_ROOT / "webeval" / "content" / "highway-fast-v0" / "scene.json"

ENV_ID = "highway-fast-v0"
# As in autonomous_driving/highway/train_highway_ppo.py: real units, ego row absolute.
OBS_CONFIG = {"observation": {"type": "Kinematics", "normalize": False}}
SEED = 102
IDLE_STEPS = 3
IDLE = 1  # DiscreteMetaAction: 0 LANE_LEFT, 1 IDLE, 2 LANE_RIGHT, 3 FASTER, 4 SLOWER


def main() -> None:
    env = gym.make(ENV_ID, config=OBS_CONFIG)
    obs, _ = env.reset(seed=SEED)
    for _ in range(IDLE_STEPS):
        obs, _, terminated, truncated, _ = env.step(IDLE)
        if terminated or truncated:
            raise SystemExit("The episode ended before the chosen step; pick another seed.")
    env.close()

    values = [float(v) for v in obs.flatten()]
    if len(values) != len(MEASURES):
        raise SystemExit(f"Observation has {len(values)} values, the specification "
                         f"declares {len(MEASURES)} measures.")
    record = {
        "env": ENV_ID,
        "highway_env": highway_env.__version__,
        "config": OBS_CONFIG,
        "seed": SEED,
        "idle_steps": IDLE_STEPS,
        "measures": dict(zip(MEASURES, values)),
    }
    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] {ENV_ID}, seed {SEED}, {IDLE_STEPS} IDLE steps -> {DST}")


if __name__ == "__main__":
    sys.exit(main())
