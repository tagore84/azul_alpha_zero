import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from azul.env import AzulEnv
from players.random_player import RandomPlayer
import numpy as np

env = AzulEnv()
obs = env.reset()
player = RandomPlayer()

try:
    action = player.predict(obs)
    print("RandomPlayer predicted action:", action)
except RuntimeError as e:
    print("RandomPlayer CRASHED:", e)
except Exception as e:
    print("RandomPlayer CRASHED with other error:", e)
