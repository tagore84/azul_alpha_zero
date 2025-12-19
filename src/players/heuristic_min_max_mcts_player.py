# src/players/heuristic_player.py
import random
import torch
import numpy as np
import copy
from azul.rules import (
    validate_origin,
    place_on_pattern_line,
    transfer_to_wall,
    calculate_floor_penalization,
    calculate_final_bonus,
    Color
)
from azul.env import AzulEnv
import math

class HeuristicMinMaxMCTSPlayer:
    def __init__(self, strategy='minmax', simulations=50, depth=2):
        self.device = torch.device("cpu")
        self.strategy = strategy
        self.simulations = simulations
        self.depth = depth

    def _reconstruct_env(self, obs):
        """
        Reconstructs a full AzulEnv from the observation dictionary.
        """
        env = AzulEnv(num_players=len(obs["players"]))
        
        # Restore scalars
        env.current_player = int(obs["current_player"])
        env.round_count = int(obs["round_count"])
        env.first_player_token = bool(obs["first_player_token"])
        
        # Restore arrays
        env.bag = np.array(obs["bag"], dtype=int)
        env.discard = np.array(obs["discard"], dtype=int)
        env.factories = np.array(obs["factories"], dtype=int)
        env.center = np.array(obs["center"], dtype=int)
        
        # Restore players
        for i, p_data in enumerate(obs["players"]):
            env.players[i]['score'] = int(p_data['score'])
            env.players[i]['wall'] = np.array(p_data['wall'], dtype=int)
            env.players[i]['floor_line'] = np.array(p_data['floor_line'], dtype=int)
            
            # Pattern lines need careful handling (list of arrays)
            # obs['players'][i]['pattern_lines'] is usually a list of arrays or list of lists
            # The environment expects list of np.arrays of different sizes
            pl_data = p_data['pattern_lines']
            env.players[i]['pattern_lines'] = [np.array(line, dtype=int) for line in pl_data]
            
        return env

    def predict(self, obs):
        # 1. Reconstruct Env
        env = self._reconstruct_env(obs)
        
        # 2. Search
        if self.strategy == 'minmax':
            best_action = self._minmax_search(env)
        elif self.strategy == 'mcts':
            best_action = self._mcts_search(env)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
            
        return best_action

    # ==========================
    # MinMax Implementation
    # ==========================
    def _minmax_search(self, env):
        valid_actions = env.get_valid_actions()
        if not valid_actions:
             return None
             
        best_val = float("-inf")
        best_action = None
        
        # Root expansion
        random.shuffle(valid_actions)
        
        alpha = float("-inf")
        beta = float("inf")
        
        # Identify Root Player to ensure we maximize THEIR score relative to opponent
        root_player = env.current_player
        
        for action in valid_actions:
            next_env = env.clone()
            next_env.step(action)
            
            # Recursive call
            # minimizing_player=False passed to next level means "Next level is Opponent" (Minimizing)
            # We pass root_player to evaluate correctly at leaf
            val = self._minmax(next_env, self.depth - 1, alpha, beta, False, root_player)
            
            if val > best_val:
                best_val = val
                best_action = action
            
            alpha = max(alpha, best_val)
            
        if best_action is None:
             return valid_actions[0]
             
        return env.action_to_index(best_action)

    def _minmax(self, env, depth, alpha, beta, maximizing_player, root_player):
        if depth == 0 or env.done:
            return self._evaluate_state(env, root_player)
        
        valid_actions = env.get_valid_actions()
        if not valid_actions:
             return self._evaluate_state(env, root_player)

        if maximizing_player:
            max_eval = float("-inf")
            for action in valid_actions:
                next_env = env.clone()
                next_env.step(action)
                eval_val = self._minmax(next_env, depth - 1, alpha, beta, False, root_player)
                max_eval = max(max_eval, eval_val)
                alpha = max(alpha, eval_val)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float("inf")
            for action in valid_actions:
                next_env = env.clone()
                next_env.step(action)
                eval_val = self._minmax(next_env, depth - 1, alpha, beta, True, root_player)
                min_eval = min(min_eval, eval_val)
                beta = min(beta, eval_val)
                if beta <= alpha:
                    break
            return min_eval

    # ==========================
    # MCTS Implementation
    # ==========================
    def _mcts_search(self, env):
        # Determine player index for the root
        root_player = env.current_player
        
        # Root Node
        root = MCTSNode(env.clone())
        
        for i in range(self.simulations):
            node = root
            
            # 1. Selection
            # Traverse until leaf or terminal
            while not node.is_leaf() and not node.env.done:
                 node = node.select_child()
            
            # 2. Expansion
            if not node.env.done and node.visits > 0: # Expand if visited at least once
                 node.expand()
                 if not node.is_leaf(): # If expanded successfully
                     node = node.select_child() # Move to first new child
            
            # 3. Simulation (Rollout)
            # Value for the player at this node (Negamax convention)
            val = self._mcts_rollout(node.env.clone(), node.env.current_player)
            
            # 4. Backpropagation
            node.backpropagate(val)
        
        # Select best move: Max Visits
        best_child = None
        max_visits = -1
        
        valid_actions = env.get_valid_actions() # For fallback check
        
        for action, child in root.children.items():
            # print(f"MCTS Child {action}: Visits={child.visits}, Value={child.total_value/child.visits:.2f}")
            if child.visits > max_visits:
                max_visits = child.visits
                best_child = action
            elif child.visits == max_visits:
                 # Break ties by value
                 if best_child and root.children[best_child].value() < child.value():
                      best_child = action

        if best_child is None:
             if valid_actions: return env.action_to_index(valid_actions[0])
             return None
             
        return env.action_to_index(best_child)

    def _mcts_rollout(self, env, perspective_player):
        """
        Run a simulation until depth limit or game end.
        Returns value from perspective of root_player.
        """
        # We use a short rollout depth or just heuristic eval?
        # Full rollout until end is expensive. 
        # Let's do a limited depth rollout + heuristic eval at end.
        rollout_depth = 5
        
        for _ in range(rollout_depth):
            if env.done:
                break
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
            # Random policy for rollout? Or Heuristic?
            # Heuristic is better but slower. Random is fast.
            # Let's use Random for diversity + Heuristic Eval at end.
            action = random.choice(valid_actions)
            env.step(action)
            
        return self._evaluate_state(env, perspective_player)

    def _evaluate_state(self, env, perspective_player=None):
        """
        Static evaluation of the state.
        Returns score from the perspective of 'perspective_player'.
        If perspective_player is None, defaults to Player 0.
        """
        # Determine index
        p_idx = 0
        if perspective_player is not None:
             if isinstance(perspective_player, int):
                 p_idx = perspective_player
             else:
                 # Attempt to infer if it's a player object, though we pass ints usually
                 pass
        
        scores = env.get_final_scores()
        
        # Heuristic bonuses
        bonuses = [0, 0]
        for i in range(2):
            wall = env.players[i]['wall']
            # Columns (7pts) - Adjusted to 7 in official rules but code said 5/2?
            # Creating consistent bonus calc
            for col in range(5):
                 col_tiles = (wall[:, col] != -1).sum()
                 if col_tiles == 4: bonuses[i] += 5 # Almost complete
                 elif col_tiles == 3: bonuses[i] += 2
                 elif col_tiles == 5: bonuses[i] += 7 # Complete
                 
            # Colors (10pts)
            flat_wall = wall.flatten()
            flat_wall = flat_wall[flat_wall != -1]
            counts = np.bincount(flat_wall, minlength=5)
            for c in counts:
                 if c == 4: bonuses[i] += 6
                 elif c == 3: bonuses[i] += 2
                 elif c == 5: bonuses[i] += 10
                 
            # Rows (2pts) - Usually handled in score but good for heuristic to encourage completion
            for r in range(5):
                 row = wall[r]
                 filled = (row != -1).sum()
                 if filled == 4: bonuses[i] += 3
                 if filled == 5: bonuses[i] += 2 # Algorithm already scores this, but extra weight helps
        
        eval_p0 = scores[0] + bonuses[0]
        eval_p1 = scores[1] + bonuses[1]
        
        if p_idx == 0:
            return eval_p0 - eval_p1
        else:
            return eval_p1 - eval_p0

# ==========================
# MCTS Node Helper
# ==========================
class MCTSNode:
    def __init__(self, env, parent=None, action=None):
        self.env = env
        self.parent = parent
        self.action = action # Action taken to get here
        self.children = {}
        self.visits = 0
        self.total_value = 0.0
        
    def is_leaf(self):
        return len(self.children) == 0
        
    def expand(self):
        valid_actions = self.env.get_valid_actions()
        for action in valid_actions:
            next_env = self.env.clone()
            next_env.step(action)
            self.children[action] = MCTSNode(next_env, parent=self, action=action)
            
    def select_child(self):
        # UCT Selection
        # C = 1.41
        C = 1.414
        best_score = float("-inf")
        best_child = None
        
        for action, child in self.children.items():
            if child.visits == 0:
                 # Infinite priority for unvisited?
                 score = float("inf")
            else:
                 # Value is presumably from perspective of the player at THIS node.
                 # But my values are stored as "Root Perspective".
                 # MCTS typically alternates: child value is for next player.
                 # Actually, usually we store Q as "Mean Value for Player who just moved (Action)"?
                 # Or "Mean Value for ROOT player".
                 
                 # Let's standardize: `total_value` is always "Score relative to Root Player".
                 # So we simply Maximize UCB.
                 # Wait, if it's Opponent turn at this node, they should Minimize value for Root (or Maximize for themselves).
                 # If `total_value` is `Score(Root) - Score(Opp)`, then:
                 # - Root Player wants to Maximize Q.
                 # - Opponent wants to Minimize Q.
                 
                 q_val = child.total_value / child.visits
                 
                 # Perspective correction
                 # If current player is Root: we want high Q.
                 # If current player is Opponent: we want low Q.
                 
                 # Correction: We want to pick the node that the CURRENT player would pick.
                 # If Root moves: Pick Max Q.
                 # If Opp moves: Pick Min Q.
                 
                 # Easier: Always flip value in backprop.
                 # Store `total_value` as "Value for player who MOVED to get here".
                 # Then UCT is always Maximize Q + U.
                 pass
                 
            # Let's stick to "Value for Root Player".
            # Current Node Player:
            current_p = self.env.current_player
            # Root Player logic is tricky without passing root index down.
            # Let's assume simple Max-UCT with Perspective Values.
            
            # SIMPLIFICATION:
            # `total_value` stores value for the player who takes the action `action`?
            # Standard AlphaZero stores value for current_player.
            pass
            
        # RE-IMPLEMENT UCT SIMPLE:
        # Just use average value (normalized?)
        # Let's use the simplest:
        # Value is always P0 - P1.
        # If self.env.current_player == 0: Maximize (Q + U)
        # If self.env.current_player == 1: Minimize (Q - U)? No, Minimize Q, Maximize U (exploration).
        # Actually standard UCT is Max(Q + U).
        
        # Let's fix `total_value` to be "Value for the player at self.env.current_player".
        # So backprop must flip signs.
        
        # But `_evaluate_state` returns (P0 - P1).
        # If we use `flip` backprop, we need `evaluate` to return relative score.
        pass
        
        # Let's implement robust selection:
        log_n = math.log(self.visits + 1)
        
        best_uct = float("-inf")
        best_node = None
        
        for action, child in self.children.items():
            if child.visits == 0:
                return child
                
            # Value for the player who is deciding (self.env.current_player)
            # If child stores "Value for Root", we need to know who Root is.
            # Assuming Root is stored/known? No.
            
            # Let's just store "Value for Current Player" in the child?
            # No, Child is Next State.
            # Value stored in Child is usually V(Child State).
            # Which is Value for Next Player.
            # So Current Player wants to Minimize Child Value?
            # Or Child stores Q(s,a) = Value for Current Player? YES.
            
            # q = Value for PARENT (current selection node)
            # child.value() is Value for CHILD's player (next player)
            # So Q = -child.value()
            q = -child.value()
            u = C * math.sqrt(log_n / child.visits)
            
            uct = q + u
            if uct > best_uct:
                best_uct = uct
                best_node = child
                
        return best_node

    def value(self):
        if self.visits == 0: return 0.0
        return self.total_value / self.visits

    def backpropagate(self, val):
        self.visits += 1
        self.total_value += val
        if self.parent:
            # Flip value for opponent?
            # If `val` is "Benefit for Me", then for Parent (Opponent) it is -Benefit.
            self.parent.backpropagate(-val)

def decode_action(index):
    C = 5; D = 6
    factory = index // (C * D)
    remainder = index % (C * D)
    color = remainder // D
    row = remainder % D
    return factory, color, row

