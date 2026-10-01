"""PPO training with normalized network inputs (variant of train_highway_ppo.py).

Identical network, hyperparameters and callbacks; the only difference is that observations are
divided by fixed per-feature constants before they reach the policy.

Why: raw Kinematics values span three orders of magnitude (ego x reaches ~1100 m, speeds ~25 m/s,
presence flags 0/1). Measured on highway-v0 at 20k steps, seeds 1 and 2:

    raw inputs     entropy -0.0005 / -0.0000, approx_kl 0.0000  -> policy frozen on one action
                   (seed 2: LANE_LEFT 100%, 95% crashes)
    scaled inputs  entropy -0.94 / -1.06,     approx_kl ~0.008  -> healthy training, no freeze
                   (seed 1: SLOWER 75% / LANE_RIGHT 25%, 0% crashes)

Scaling fixes the optimisation, not the driving: every non-crashing highway-v0 policy still
settles at the 20 m/s minimum, because the default reward makes cautious driving optimal there.

NOTE FOR EXPORT: a model trained here expects scaled inputs, while the SLEEC constants and
verifier/sleec2marabou_parser.py work in real units (metres, m/s). export_to_onnx.py does NOT
handle that yet, so it must not be pointed at these models as-is. The scaling is linear, so when
the time comes it can be absorbed without changing the verification: divide the first layer's
weight columns by OBS_SCALE (exact, no extra layers, same 512 ReLUs), or prepend a Mul node to
the ONNX graph (weights untouched, no extra ReLUs).
"""

import os

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

TRAIN = True

# ============================================================================
# Select the environment here; everything else follows from ENV_CONFIGS.
# ============================================================================
ENV_ID = "highway-v0"  # or "highway-fast-v0"

# Same structure as train_highway_ppo.py. Shared by both: 256x256 ReLU actor and critic,
# n_epochs=10, learning_rate=5e-4, ent_coef=0.0, default highway-env reward, and the input
# scaling below. n_steps is the rollout length PER COPY (128 trains, 32 froze the policy).
ENV_CONFIGS = {
    # 3 lanes, 20 vehicles, 5 Hz physics, 30-step episodes; ~190 steps/s with 6 copies.
    # Mirrors the unscaled settings; input scaling has not been measured on this environment,
    # which already trains to a varied policy without it.
    "highway-fast-v0": dict(
        n_cpu=6,
        total_timesteps=50_000,
        seed=1,
        n_steps=128,      # 6 copies x 128 = 768 steps per update
        batch_size=64,
        gamma=0.8,        # the highway-env example value
    ),
    # 4 lanes, 50 vehicles, 15 Hz physics, 40-step episodes; ~29 steps/s with 12 copies.
    # Measured at 20k with scaling, seeds 1 and 2: entropy -0.94 / -1.06 and approx_kl ~0.008
    # (healthy) versus entropy 0 and approx_kl 0 (frozen) without it. Seed 1 gave
    # SLOWER 75% / LANE_RIGHT 25%, 0% crashes. Budgets beyond 20k are untested here.
    "highway-v0": dict(
        n_cpu=12,
        total_timesteps=20_000,
        seed=1,
        n_steps=128,      # 12 copies x 128 = 1536 steps per update
        batch_size=256,
        gamma=0.95,
    ),
}

CONFIG = ENV_CONFIGS[ENV_ID]
N_CPU = CONFIG["n_cpu"]
TOTAL_TIMESTEPS = CONFIG["total_timesteps"]
SEED = CONFIG["seed"]
N_EVAL_EPISODES = 50

# Separate folders so scaled and unscaled runs never overwrite each other.
MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models", f"{ENV_ID}-scaled")
MODEL_PATH = os.path.join(MODEL_DIR, "highway_ppo_model")  # policy at the last training step
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model")  # best periodic evaluation (EvalCallback)
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs", f"{ENV_ID}-scaled")

# The environment still reports real-world units; the scaling below is applied by this script.
# (highway-env's own "normalize" option would also work with "clip": False, but its divisors are
# derived at runtime and the y range depends on the lane count, which makes the export ambiguous.)
OBS_CONFIG = {"observation": {"type": "Kinematics", "normalize": False}}

# Per-feature divisors: presence, x, y, vx, vy. Fixed constants rather than running statistics,
# so runs stay reproducible and the transform is exactly invertible at export time.
OBS_SCALE = np.array([1.0, 100.0, 10.0, 30.0, 10.0], dtype=np.float32)

ACTION_NAMES = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}


class ScaleObservation(gym.ObservationWrapper):
    """Divides each Kinematics feature by OBS_SCALE; all five vehicle rows share the divisors."""

    def observation(self, obs):
        return (obs / OBS_SCALE).astype(np.float32)


def make_env(n_envs, seed, vec_env_cls=None):
    return make_vec_env(ENV_ID, n_envs=n_envs, seed=seed, vec_env_cls=vec_env_cls,
                        env_kwargs={"config": OBS_CONFIG}, wrapper_class=ScaleObservation)


def evaluate(model_path, n_episodes=N_EVAL_EPISODES):
    """Deterministic evaluation on fixed seeds, using the same input scaling as training.

    Reports mean speed and the action mix as well: a policy that just brakes also avoids
    crashes, but sits at the minimum speed (20 m/s) and outputs SLOWER almost always.
    """
    model = PPO.load(model_path, device="cpu")
    env = ScaleObservation(gym.make(ENV_ID, config=OBS_CONFIG))
    lengths, rewards, speeds, crashes = [], [], [], 0
    action_counts = {a: 0 for a in ACTION_NAMES}
    for episode in range(n_episodes):
        obs, info = env.reset(seed=1000 + episode)
        done = truncated = False
        ep_reward, ep_len = 0.0, 0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            action_counts[int(action)] += 1
            obs, reward, done, truncated, info = env.step(action)
            speeds.append(float(info["speed"]))
            ep_reward += reward
            ep_len += 1
        crashes += int(info["crashed"])
        lengths.append(ep_len)
        rewards.append(ep_reward)
    env.close()
    total_actions = sum(action_counts.values())
    action_mix = " ".join(
        f"{ACTION_NAMES[a]}:{action_counts[a] / total_actions:.0%}"
        for a in sorted(action_counts) if action_counts[a]
    )
    return (crashes / n_episodes, sum(lengths) / n_episodes, sum(rewards) / n_episodes,
            sum(speeds) / len(speeds), action_mix)


if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Training config for {ENV_ID} (scaled inputs): {CONFIG}")

    if TRAIN:
        env = make_env(N_CPU, SEED, vec_env_cls=SubprocVecEnv)

        model = PPO(
            "MlpPolicy",
            env,
            # ReLU (instead of SB3's default Tanh) keeps the network piecewise-linear,
            # which is what Marabou verifies exactly.
            policy_kwargs=dict(
                activation_fn=nn.ReLU,
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
            ),
            n_steps=CONFIG["n_steps"],
            batch_size=CONFIG["batch_size"],
            n_epochs=10,
            learning_rate=5e-4,
            ent_coef=0.0,
            gamma=CONFIG["gamma"],
            verbose=2,
            seed=SEED,
            # The simulation runs on CPU; a GPU only adds transfer overhead for this small MLP.
            device="cpu",
            tensorboard_log=LOG_DIR,
        )

        # Four checkpoints and five evaluations per run, whatever the budget.
        checkpoint = CheckpointCallback(
            save_freq=max(TOTAL_TIMESTEPS // 4 // N_CPU, 1),
            save_path=MODEL_DIR,
            name_prefix="highway_ppo",
        )
        eval_callback = EvalCallback(
            make_env(1, SEED + 123),
            best_model_save_path=MODEL_DIR,
            log_path=LOG_DIR,
            eval_freq=max(TOTAL_TIMESTEPS // 5 // N_CPU, 1),
            n_eval_episodes=20,
            deterministic=True,
        )
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[checkpoint, eval_callback])
        model.save(MODEL_PATH)
        env.close()
        print(f"Model saved to {MODEL_PATH}.zip")

    if not os.path.exists(MODEL_PATH + ".zip"):
        raise FileNotFoundError(
            f"No trained model at {MODEL_PATH}.zip — set TRAIN = True first."
        )

    for label, path in [("last", MODEL_PATH), ("best", BEST_MODEL_PATH)]:
        if not os.path.exists(path + ".zip"):
            continue
        crash_rate, mean_len, mean_reward, mean_speed, action_mix = evaluate(path)
        print(
            f"Evaluation over {N_EVAL_EPISODES} episodes [{label}: {os.path.basename(path)}.zip]: "
            f"crash rate = {crash_rate:.0%} | "
            f"mean length = {mean_len:.1f} | "
            f"mean reward = {mean_reward:.2f} | "
            f"mean speed = {mean_speed:.1f} m/s"
        )
        print(f"    actions: {action_mix}")
