import os
import gymnasium as gym
import highway_env  # noqa: F401
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

TRAIN = True
N_CPU = 6
TOTAL_TIMESTEPS = 20_000

MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
MODEL_PATH = os.path.join(MODEL_DIR, "highway_ppo_model")

# normalize=False keeps real-world units (meters, m/s), which allows integer
# constants in the SLEEC specification (the SLEEC grammar only supports INT values).
# absolute=False (default) makes positions and velocities relative to the ego vehicle.
OBS_CONFIG = {"observation": {"normalize": False}}


if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)

    if TRAIN:
        batch_size = 64
        env = make_vec_env(
            "highway-fast-v0",
            n_envs=N_CPU,
            vec_env_cls=SubprocVecEnv,
            env_kwargs={"config": OBS_CONFIG},
        )

        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
            n_steps=batch_size * 12 // N_CPU,
            batch_size=batch_size,
            n_epochs=10,
            learning_rate=5e-4,
            gamma=0.8,
            verbose=2,
            tensorboard_log=os.path.join(os.path.dirname(__file__), "logs/"),
        )

        model.learn(total_timesteps=TOTAL_TIMESTEPS)
        model.save(MODEL_PATH)
        env.close()
        print(f"Model saved to {MODEL_PATH}.zip")

    if not os.path.exists(MODEL_PATH + ".zip"):
        raise FileNotFoundError(
            f"No trained model at {MODEL_PATH}.zip — set TRAIN = True first."
        )

    model = PPO.load(MODEL_PATH, device="cpu")
    eval_env = gym.make("highway-fast-v0", config=OBS_CONFIG)
    total_reward = 0.0
    for episode in range(3):
        obs, info = eval_env.reset()
        done = truncated = False
        ep_reward = 0.0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = eval_env.step(action)
            ep_reward += reward
        total_reward += ep_reward
        print(f"Episode {episode + 1}: reward = {ep_reward:.2f}")
    eval_env.close()
    print(f"Mean reward over 3 episodes: {total_reward / 3:.2f}")
