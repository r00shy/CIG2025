from stable_baselines3 import PPO
import os
from game_env import PokemonBattleEnv
import time

instance = "v2"  # same as before, or generate dynamically

models_dir = f"models/{instance}/"
logdir = f"logs/{instance}/"

os.makedirs(models_dir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)

env = PokemonBattleEnv()
env.reset()

# === Check if a saved model exists to resume from ===
latest_model = None
model_files = sorted(
    [f for f in os.listdir(models_dir) if f.endswith(".zip")],
    key=lambda x: int(x.replace(".zip", ""))
)

if model_files:
    latest_model_path = os.path.join(models_dir, model_files[-1])
    print(f"Loading model from {latest_model_path}")
    model = PPO(
    "MlpPolicy",
    env,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.1,
    learning_rate=1e-4,
    policy_kwargs=dict(net_arch=[256, 256]),
    clip_range=0.2,
    verbose=1,
    tensorboard_log=logdir
    )
    iters = int(model_files[-1].replace(".zip", "")) // 10000
else:
    print("No saved model found, starting fresh.")
    model = PPO(
    "MlpPolicy",
    env,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.1,
    learning_rate=1e-4,
    policy_kwargs=dict(net_arch=[256, 256]),
    clip_range=0.2,
    verbose=1,
    tensorboard_log=logdir
    )
    iters = 0

# === Training loop ===
TIMESTEPS = 10000
while True:
    iters += 1
    model.learn(
        total_timesteps=TIMESTEPS,
        reset_num_timesteps=False,
        tb_log_name="PPO"  # Keep this constant to preserve a clean log curve
    )
    model.save(f"{models_dir}/{TIMESTEPS * iters}")
