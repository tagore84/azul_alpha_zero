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
        exp_logits = np.exp(logits - np.max(logits))
        exp_logits[~mask] = 0
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

    def backpropagate(self, path: list, value: float):
        """
        Propagate the simulation result back up the tree.
        """
        for node in reversed(path):
            node.visits += 1
            node.value_sum += value
            # alternate value sign for two-player zero-sum
            value = -value

    def run(self, root_env: Optional[AzulEnv] = None):
        """
        Perform MCTS simulations starting from the root.
        """
        if root_env is not None:
            self.root = MCTS.Node(root_env.clone(), parent=None, prior=1.0)
        
        for sim in range(self.simulations):
            leaf, path = self.select()
            # If terminal state, get value
            obs = leaf.env._get_obs()
            # check game over
            done = any(all(cell != -1 for cell in row) for p in leaf.env.players for row in p['wall'])
            if not done:
                self.expand(leaf)
                # Evaluate leaf value with the network (no rollout)
                obs = leaf.env._get_obs()
                obs_flat = leaf.env.encode_observation(obs)
                _, value = self.model.predict(np.array([obs_flat]))
                self.backpropagate(path, float(value))
            else:
                # terminal node: compute value directly
                # assume last reward
                value = 1.0
                self.backpropagate(path, value)
        

    def select_action(self) -> Tuple[int, int, int]:
        """
        After running simulations, pick the most visited child action, but ensure it's still valid in the current env.
        """
        if not self.root.children:
            self.run()
        valid_actions = self.root.env.get_valid_actions()
        candidates = [(a, n) for a, n in self.root.children.items() if a in valid_actions]
        if not candidates:
            print("[WARN] No valid children in MCTS, selecting random valid action")
            return random.choice(valid_actions)
        action, node = max(candidates, key=lambda item: item[1].visits)
        return action

    def advance(self, env: AzulEnv):
        self.root = MCTS.Node(env.clone(), parent=None, prior=1.0)