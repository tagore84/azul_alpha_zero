import sys
import os
# Add project src folder to PYTHONPATH for module resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import random
from azul.rules import validate_origin, place_on_pattern_line, Color
from azul.env import AzulEnv

def random_game():
    env = AzulEnv(num_players=2, factories_count=5)
    obs = env.reset()
    done = False
    total_rewards = [0, 0]

    while not done:
        p_idx = env.current_player
        pattern_lines = [line.tolist() for line in env.players[p_idx]['pattern_lines']]
        valid_actions = []
        # sources 0..N-1 are factories, N is center
        for source_idx in range(env.N + 1):
            origin = ("factory", source_idx) if source_idx < env.N else ("center", None)
            # determine count of tiles at origin
            for color in range(env.C):
                if source_idx < env.N:
                    count = int(env.factories[source_idx, color])
                else:
                    count = int(env.center[color])
                if count == 0:
                    continue
                # only consider if tiles available at source
                if not validate_origin(env.factories.tolist(), env.center.tolist(), origin, Color(color)):
                    continue
                # Only include dests where at least one tile can be placed on a pattern line
                for dest in range(5):
                    new_line, overflow = place_on_pattern_line(pattern_lines[dest], Color(color), count)
                    placed = count - overflow
                    if placed > 0:
                        valid_actions.append((source_idx, color, dest))
        # If no pattern-line action is valid, fallback to any valid floor action
        if not valid_actions:
            for source_idx in range(env.N + 1):
                origin = ("factory", source_idx) if source_idx < env.N else ("center", None)
                for color in range(env.C):
                    if validate_origin(env.factories.tolist(), env.center.tolist(), origin, Color(color)):
                        valid_actions.append((source_idx, color, 5))
        # select a random valid action
        action = random.choice(valid_actions)
        obs, reward, done, info = env.step(action)
        total_rewards[obs['current_player'] - 1] += reward  # recompensa del jugador
        env.render()
        print(f"Reward this move: {reward}\n{'-'*40}\n")

    print("Juego terminado. Recompensas totales:", total_rewards)

if __name__ == "__main__":
    random_game()