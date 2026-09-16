import os
import gymnasium as gym
import highway_env  # noqa: F401
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

TRAIN = True
# --- highway-fast-v0: 3 lanes, 20 vehicles, 5 Hz physics, 30-step episodes (~190 steps/s) ---
# ENV_ID = "highway-fast-v0"
# N_CPU = 6

# --- highway-v0: 4 lanes, 50 vehicles, 15 Hz physics, 40-step episodes ---
# ~39 steps/s with 24 copies on this machine (24 physical cores), so 50k takes ~21 min.
ENV_ID = "highway-v0"
N_CPU = 24
# Learning plateaus by ~30k steps; a 200k run scored no better (26% crashes vs 18% at 50k).
TOTAL_TIMESTEPS = 50_000
# Measured at 50k over seeds 0/1/2: 18% / 7% / 17% crashes (mean 14% +/- 5%). Seed 1 gives the
# safest policy that still varies its actions (IDLE 72%, SLOWER 19%, LANE_RIGHT 9%).
SEED = 1
N_EVAL_EPISODES = 50
BATCH_SIZE = 64

# Models, checkpoints and logs are kept per environment so runs on different ENV_IDs don't overwrite each other.
MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models", ENV_ID)
MODEL_PATH = os.path.join(MODEL_DIR, "highway_ppo_model")  # policy at the last training step
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model")  # best periodic evaluation (EvalCallback)
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs", ENV_ID)

# highway-env replaces (not merges) the "observation" dict, so "type" must be given explicitly.
# normalize=False keeps real-world units (meters, m/s), which allows integer
# constants in the SLEEC specification (the SLEEC grammar only supports INT values).
# absolute=False (default) makes other vehicles' positions and velocities relative to the ego
# vehicle; the ego row itself stays absolute.
OBS_CONFIG = {"observation": {"type": "Kinematics", "normalize": False}}


ACTION_NAMES = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}


def evaluate(model_path, n_episodes=N_EVAL_EPISODES):
    """Deterministic evaluation on fixed seeds.

    Reports mean speed and the action mix as well: a policy that just brakes also avoids
    crashes, but sits at the minimum speed (20 m/s) and outputs SLOWER almost always.
    """
    model = PPO.load(model_path, device="cpu")
    env = gym.make(ENV_ID, config=OBS_CONFIG)
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

    if TRAIN:
        env = make_vec_env(
            ENV_ID,
            n_envs=N_CPU,
            seed=SEED,
            vec_env_cls=SubprocVecEnv,
            env_kwargs={"config": OBS_CONFIG},
        )

        model = PPO(
            "MlpPolicy",
            env,
            # ReLU (instead of SB3's default Tanh) keeps the network piecewise-linear,
            # which is what Marabou verifies exactly.
            policy_kwargs=dict(
                activation_fn=nn.ReLU,
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
            ),
            # What matters is the rollout length PER COPY: 128 steps trains, 32 steps collapses
            # (measured on highway-v0: 100% crashes, entropy 0, approx_kl 0 = policy frozen).
            #
            # highway-fast-v0 with 6 copies: the formula gives 128 steps per copy, 768 per update.
            # n_steps=BATCH_SIZE * 12 // N_CPU,
            # batch_size=BATCH_SIZE,
            #
            # highway-v0 with 24 copies: keep 128 steps per copy (3072 per update, 12 minibatches).
            n_steps=128,
            batch_size=256,
            n_epochs=10,
            learning_rate=5e-4,
            # No entropy bonus: ent_coef=0.01 collapsed the policy onto "always SLOWER"
            # (0% crashes, but stuck at the 20 m/s minimum and useless for SLEEC checking).
            ent_coef=0.0,
            gamma=0.8,
            verbose=2,
            seed=SEED,
            # The simulation runs on CPU; a GPU only adds transfer overhead for this small MLP.
            device="cpu",
            tensorboard_log=LOG_DIR,
        )

        # save_freq counts calls per vectorized step, so divide by N_CPU for ~50k timesteps.
        checkpoint = CheckpointCallback(
            save_freq=max(50_000 // N_CPU, 1),
            save_path=MODEL_DIR,
            name_prefix="highway_ppo",
        )
        # Keeps the best policy seen during training (saved_models/<ENV_ID>/best_model.zip),
        # so a late dip in performance cannot cost you the good model.
        eval_callback = EvalCallback(
            make_vec_env(ENV_ID, n_envs=1, seed=SEED + 123, env_kwargs={"config": OBS_CONFIG}),
            best_model_save_path=MODEL_DIR,
            log_path=LOG_DIR,
            eval_freq=max(10_000 // N_CPU, 1),
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
