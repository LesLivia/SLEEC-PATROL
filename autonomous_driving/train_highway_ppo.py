import os
import gymnasium as gym
import highway_env  # noqa: F401
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

TRAIN = True

# ============================================================================
# Select the environment here; everything else follows from ENV_CONFIGS.
# ============================================================================
ENV_ID = "highway-fast-v0"  # or "highway-v0"

# Shared by both environments: 256x256 ReLU actor and critic, n_epochs=10,
# learning_rate=5e-4, ent_coef=0.0 and the default highway-env reward.
#
# n_steps is the rollout length PER COPY, and that matters more than the total: 128 steps per
# copy trains, while 32 froze the policy on highway-v0 (100% crashes, entropy 0, approx_kl 0).
ENV_CONFIGS = {
    # 3 lanes, 20 vehicles, 5 Hz physics, 30-step episodes; ~190 steps/s with 6 copies.
    # Learning plateaus by ~30k steps and a 200k run scored no better. Seeds 0/1/2 at 50k gave
    # 18% / 7% / 17% crashes; seed 1 keeps a varied mix (IDLE 72%, SLOWER 19%, LANE_RIGHT 9%).
    "highway-fast-v0": dict(
        n_cpu=6,
        total_timesteps=50_000,
        seed=1,
        n_steps=128,      # 6 copies x 128 = 768 steps per update (the highway-env example)
        batch_size=64,
        gamma=0.8,        # the highway-env example value
    ),
    # 4 lanes, 50 vehicles, 15 Hz physics, 40-step episodes; ~39 steps/s with 24 copies.
    # 50k converges to a single constant action (SLOWER 99%), so stop at 20k. Seeds measured
    # with these settings over 30 episodes:
    #   seed 0: IDLE 17% / LANE_RIGHT 55% / SLOWER 28%, 0% crashes, entropy -1.14  <- balanced
    #   seed 1: IDLE 81% / SLOWER 19%,                  0% crashes, entropy -0.33
    #   seed 2: IDLE 100%,                             95% crashes                 <- collapsed
    #   seed 3: LANE_RIGHT 100%,                       97% crashes                 <- collapsed
    # gamma 0.95 was the least degenerate of 0.8 / 0.95 / 0.99.
    "highway-v0": dict(
        n_cpu=24,
        total_timesteps=20_000,
        seed=0,
        n_steps=128,      # 24 copies x 128 = 3072 steps per update, 12 minibatches
        batch_size=256,
        gamma=0.95,
    ),
}

CONFIG = ENV_CONFIGS[ENV_ID]
N_CPU = CONFIG["n_cpu"]
TOTAL_TIMESTEPS = CONFIG["total_timesteps"]
SEED = CONFIG["seed"]
N_EVAL_EPISODES = 50

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
    print(f"Training config for {ENV_ID}: {CONFIG}")

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
            n_steps=CONFIG["n_steps"],
            batch_size=CONFIG["batch_size"],
            n_epochs=10,
            learning_rate=5e-4,
            # No entropy bonus: ent_coef=0.01 collapsed the policy onto "always SLOWER"
            # (0% crashes, but stuck at the 20 m/s minimum and useless for SLEEC checking).
            ent_coef=0.0,
            gamma=CONFIG["gamma"],
            verbose=2,
            seed=SEED,
            # The simulation runs on CPU; a GPU only adds transfer overhead for this small MLP.
            device="cpu",
            tensorboard_log=LOG_DIR,
        )

        # save_freq counts calls per vectorized step, so divide by N_CPU. A quarter of the budget
        # gives four snapshots, so a late drift into a constant-action policy can be spotted.
        checkpoint = CheckpointCallback(
            save_freq=max(TOTAL_TIMESTEPS // 4 // N_CPU, 1),
            save_path=MODEL_DIR,
            name_prefix="highway_ppo",
        )
        # Keeps the best policy seen during training (saved_models/<ENV_ID>/best_model.zip),
        # so a late dip in performance cannot cost you the good model.
        eval_callback = EvalCallback(
            make_vec_env(ENV_ID, n_envs=1, seed=SEED + 123, env_kwargs={"config": OBS_CONFIG}),
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
