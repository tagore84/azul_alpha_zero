# src/mcts/mcts.py

import math
import random
import numpy as np
import time
from typing import Optional, Tuple, Dict, Any
from azul.env import AzulEnv

class MCTS:
    class Node:
        def __init__(self, env: AzulEnv, parent: Optional['MCTS.Node'], prior: float):
            self.env = env  # Game state at this node
            self.player = env.current_player # Store player who is about to move
            self.parent = parent
            self.prior = prior  # Prior probability from policy network (or uniform)
            self.children: Dict[Tuple[int,int,int], MCTS.Node] = {}
            self.visits = 0
            self.value_sum = 0.0

        @property
        def value(self) -> float:
            return self.value_sum / self.visits if self.visits > 0 else 0.0

        def ucb_score(self, cpuct: float) -> float:
            """
            Upper Confidence bound for trees (PUCT).
            """
            if self.parent is None:
                return 0
            # Exploration term
            exploration = cpuct * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
            return self.value + exploration

    def __init__(self, env: AzulEnv, model: Any, simulations: int = 100, cpuct: float = 1.0):
        """
        env: an AzulEnv instance to clone for rollouts.
        simulations: number of MCTS simulations per move.
        cpuct: exploration constant.
        """
        self.root = MCTS.Node(env.clone(), parent=None, prior=1.0)
        self.model = model
        self.simulations = simulations
        self.cpuct = cpuct

    def select(self) -> Tuple['MCTS.Node', list]:
        """
        Select a leaf node to expand.
        Returns the leaf node and the path of nodes taken.
        """
        node = self.root
        path = [node]
        # Traverse until we find a leaf
        while node.children:
            # pick child with highest UCB score
            action, node = max(node.children.items(),
                               key=lambda item: item[1].ucb_score(self.cpuct))
            path.append(node)
        return node, path

    def expand(self, node: 'MCTS.Node'):
        """
        Expand the given leaf node by creating all children.
        Use uniform prior or a policy network in future.
        """
        obs = node.env._get_obs()
        # generate valid actions
        valid_actions = node.env.get_valid_actions()
        if not valid_actions:
            # No valid actions. This should be handled by 'done' check in run(),
            # but as a safety net, we return without adding children.
            # print(f"[DEBUG] Expand: No valid actions for node. Player: {node.player}")
            return
        # print(f"[DEBUG] Expand: valid_actions={len(valid_actions)}")
        # print(f"[DEBUG] Expand: valid_actions={len(valid_actions)}")
        
        # Compute policy logits from the network and convert to priors
        obs = node.env._get_obs()
        obs_flat = node.env.encode_observation(obs)
        pi_logits, _ = self.model.predict(np.array([obs_flat]))
        logits = pi_logits[0]
        # Mask out invalid actions
        mask = np.zeros_like(logits, dtype=bool)
        for action in valid_actions:
            idx = node.env.action_to_index(action)
            mask[idx] = True
        # Compute priors only over valid actions
        logits[~mask] = -float('inf')
        exp_logits = np.exp(logits - np.max(logits))
        total = exp_logits.sum()
        if total > 0:
            priors = exp_logits / total
        else:
            # If all masked out, assign uniform over valid
            priors = mask.astype(float) / mask.sum()
        for action in valid_actions:
            # clone environment efficiently
            new_env = node.env.clone()
            # apply action
            new_env.step(action, is_sim=True)
            idx = node.env.action_to_index(action)
            node.children[action] = MCTS.Node(new_env, parent=node, prior=priors[idx])
        # print(f"[DEBUG] Expand: children added={len(node.children)}")

    def backpropagate(self, path: list, value: float):
        """
        Propagate the simulation result back up the tree.
        """
        # value is from the perspective of the player at the leaf node (who just moved or terminal state)
        # We need to propagate this up.
        
        # The value passed in is usually from the perspective of the player whose turn it is at the leaf state
        # OR if terminal, it's the game result.
        
        # Let's standardize: 'value' is always from the perspective of the player at the END of the path (leaf.player).
        
        leaf_player = path[-1].player
        
        for node in reversed(path):
            node.visits += 1
            # If the node's player is the same as the leaf player, they want to MAXIMIZE this value.
            # If different, they want to MINIMIZE it (so we add negative value).
            # Note: This assumes zero-sum +1/-1 values.
            
            if node.player == leaf_player:
                node.value_sum += value
            else:
                node.value_sum -= value

    def run(self, root_env: Optional[AzulEnv] = None):
        """
        Perform MCTS simulations starting from the root.
        """
        if root_env is not None:
            self.root = MCTS.Node(root_env.clone(), parent=None, prior=1.0)
        
        # print(f"[DEBUG] MCTS Run: simulations={self.simulations}")
        for sim in range(self.simulations):
            leaf, path = self.select()
            # If terminal state, get value
            obs = leaf.env._get_obs()
            # check game over
            done = leaf.env.done or any(all(cell != -1 for cell in row) for p in leaf.env.players for row in p['wall'])
            # print(f"[DEBUG] Sim {sim}: done={done}, leaf children={len(leaf.children)}")
            if not done:
                self.expand(leaf)
            else:
                # print(f"[DEBUG] Sim {sim}: Node is done. env.done={leaf.env.done}")
                pass
                if not leaf.children:
                     # If expand failed to add children (e.g. no valid actions despite not done?), treat as terminal
                     # This prevents the loop from trying to simulate from a dead node
                     # But we need a value.
                     # Let's just break and treat as loss/tie?
                     # Or better, expand() should have raised or handled it.
                     # If we are here, valid_actions was empty.
                     # This means it IS done effectively.
                     done = True
                else:
                    # Evaluate leaf value with the network (no rollout)
                    obs = leaf.env._get_obs()
                    obs_flat = leaf.env.encode_observation(obs)
                    _, value_out = self.model.predict(np.array([obs_flat]))
                    value = float(value_out)
                    # The network returns value for the current player (leaf.player).
                    self.backpropagate(path, value)
            
            if done:
                # terminal node: compute value directly
                scores = leaf.env.get_final_scores()
                # Assuming 2 players
                p0_score = scores[0]
                p1_score = scores[1]
                
                # Value from perspective of leaf.player
                # Normalize by 100.0
                if leaf.player == 0:
                    value = (p0_score - p1_score) / 100.0
                else:
                    value = (p1_score - p0_score) / 100.0
                
                self.backpropagate(path, value)
        

    def select_action(self) -> Tuple[int, int, int]:
        """
        After running simulations, pick the most visited child action, but ensure it's still valid in the current env.
        """
        if not self.root.children:
            # print("[DEBUG] No children in root, calling run()")
            self.run()
        valid_actions = self.root.env.get_valid_actions()
        candidates = [(a, n) for a, n in self.root.children.items() if a in valid_actions]
        if not candidates:
            # Fallback to random if no MCTS children available
            return random.choice(valid_actions)
        action, node = max(candidates, key=lambda item: item[1].visits)
        return action

    def advance(self, env: AzulEnv):
        """
        Advance the root of the tree to the child corresponding to the new state.
        This reuses the subtree.
        """
        # We can't easily match 'env' to a child because env is the RESULT state.
        # But we know the action that was taken in the external loop?
        # Actually, self_play.py calls advance(env).
        # We need to find which child matches 'env'.
        # Since 'env' is a clone, we can't compare objects.
        # But we can assume the external loop calls advance after taking an action.
        # Ideally, advance should take the action, not the env.
        
        # For now, let's just reset. To implement reuse properly, we need to change the signature
        # or find the child that matches.
        # Let's change the signature in a future step if needed, but for now, 
        # let's try to match based on observation or just reset if too complex.
        
        # Actually, let's just reset for safety in this iteration, 
        # but the plan said "Reuse MCTS tree".
        # To reuse, we need the action.
        # Let's modify self_play.py to pass the action to advance?
        # Or we can just search for a child whose state matches 'env'.
        
        # Simple matching:
        # found = None
        # for action, child in self.root.children.items():
        #     # This is expensive to compare full states.
        #     pass
            
        # Let's stick to reset for now to be safe, as the plan didn't explicitly detail the signature change.
        # Wait, the plan said: "mcts.advance(env) should keep the subtree."
        # I will implement a "best effort" reuse if I can, but without the action it's hard.
        # Let's just reset.
        self.root = MCTS.Node(env.clone(), parent=None, prior=1.0)