from stable_baselines3.common.env_checker import check_env
from game_env import PokemonBattleEnv

env = PokemonBattleEnv()

check_env(env)

