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

    def expand(self, node: 'MCTS.Node') -> float:
        """
        Expand the given leaf node by creating all children.
        Use policy network with action mask to get priors.
        Returns the value of the node from the network perspective.
        """
        obs = node.env._get_obs()
        # generate valid actions
        valid_actions = node.env.get_valid_actions()
        if not valid_actions:
            # No valid actions. This should be handled by 'done' check in run(),
            # but as a safety net, we return 0 value (neutral)
            return 0.0
        
        # Compute policy logits from the network with action mask
        obs_flat = node.env.encode_observation(obs)
        
        # Create action mask (1 for legal, 0 for illegal)
        action_mask = np.zeros(node.env.action_size, dtype=np.float32)
        for action in valid_actions:
            idx = node.env.action_to_index(action)
            action_mask[idx] = 1.0
        
        # Pass mask to model - Single inference for Policy AND Value
        pi_logits, values = self.model.predict(np.array([obs_flat]), np.array([action_mask]))
        logits = pi_logits[0]
        value = float(values[0])
        
        # Compute priors with softmax (logits are already masked by network)
        exp_logits = np.exp(logits - np.max(logits))
        total = exp_logits.sum()
        if total > 0:
            priors = exp_logits / total
        else:
            # Fallback to uniform over valid actions
            priors = action_mask / action_mask.sum()
        
        # Extract priors for valid actions and renormalize
        valid_priors = []
        for action in valid_actions:
            idx = node.env.action_to_index(action)
            valid_priors.append(priors[idx])
            
        valid_priors = np.array(valid_priors)
        total_valid_prior = valid_priors.sum()
        
        if total_valid_prior > 0:
            valid_priors /= total_valid_prior
        else:
            # Fallback if network predicts 0 prob for all valid moves
            valid_priors = np.ones(len(valid_actions)) / len(valid_actions)
        
        for i, action in enumerate(valid_actions):
            # clone environment efficiently
            new_env = node.env.clone()
            # apply action
            new_env.step(action, is_sim=True)
            node.children[action] = MCTS.Node(new_env, parent=node, prior=valid_priors[i])
            
        return value

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
            # Check if terminal state
            obs = leaf.env._get_obs()
            done = leaf.env.done or any(all(cell != -1 for cell in row) for p in leaf.env.players for row in p['wall'])
            
            if not done:
                # Non-terminal: expand and evaluate with network simultaneously
                value = self.expand(leaf)
                
                # Network returns value for current player (leaf.player)
                self.backpropagate(path, value)
            else:
                # Terminal: compute exact game value
                scores = leaf.env.get_final_scores()
                p0_score, p1_score = scores[0], scores[1]
                
                # Check for max_rounds termination
                termination_reason = getattr(leaf.env, 'termination_reason', 'normal_end')
                
                if termination_reason == 'max_rounds':
                    # Fix: Penalize both players significantly to avoid this state
                    # Value is from perspective of leaf.player. If it's bad for everyone, it's -1.0
                    value = -1.0
                else:
                    # Value from perspective of leaf.player
                    # Win/Loss: +1/-1/0
                    if p0_score > p1_score:
                        value = 1.0 if leaf.player == 0 else -1.0
                    elif p1_score > p0_score:
                        value = 1.0 if leaf.player == 1 else -1.0
                    else:
                        value = 0.0
                
                self.backpropagate(path, value)
        

    def add_root_noise(self, alpha: float = 0.3, epsilon: float = 0.25):
        """
        Add Dirichlet noise to the root node's priors to encourage exploration.
        P(s, a) = (1 - epsilon) * P(s, a) + epsilon * Dirichlet(alpha)
        """
        if not self.root.children:
            self.expand(self.root)
            
        children = self.root.children
        if not children:
            return
            
        actions = list(children.keys())
        noise = np.random.dirichlet([alpha] * len(actions))
        
        for i, action in enumerate(actions):
            node = children[action]
            node.prior = (1 - epsilon) * node.prior + epsilon * noise[i]

    def select_action(self, temperature: float = 1.0) -> Tuple[int, int, int]:
        """
        After running simulations, pick a child action based on visit counts and temperature.
        - If temperature == 0: Greedy selection (max visits).
        - If temperature > 0: Sample proportional to visits^(1/temp).
        """
        if not self.root.children:
            self.run()
            
        valid_actions = self.root.env.get_valid_actions()
        # Filter children that are valid actions (should be all, but safety check)
        candidates = [(a, n) for a, n in self.root.children.items() if a in valid_actions]
        
        if not candidates:
            return random.choice(valid_actions)
            
        if temperature == 0:
            # Greedy selection
            action, node = max(candidates, key=lambda item: item[1].visits)
            return action
        else:
            # Stochastic selection
            visits = np.array([n.visits for a, n in candidates], dtype=np.float64)
            
            # Raise to temperature power
            # Avoid overflow if temp is small (though here we expect temp=1.0)
            try:
                visits = visits ** (1.0 / temperature)
            except OverflowError:
                # If overflow, treat max element as 1 and others as 0 (greedy-like)
                max_idx = np.argmax(visits)
                visits = np.zeros_like(visits)
                visits[max_idx] = 1.0
            
            # Check for infinities
            if np.isinf(visits).any():
                visits = np.where(np.isinf(visits), 1.0, 0.0)
            
            visit_sum = visits.sum()
            
            if visit_sum <= 0:
                # Fallback to uniform if no visits (should not happen if sims > 0)
                probs = np.ones_like(visits) / len(visits)
            else:
                probs = visits / visit_sum
            
            # Clip and re-normalize to ensure strict sum to 1.0 for np.random.choice
            probs = np.clip(probs, 1e-10, 1.0)
            probs /= probs.sum()
            
            # Sample index
            try:
                idx = np.random.choice(len(candidates), p=probs)
            except ValueError as e:
                # Last resort fallback
                print(f"[MCTS] Warning: Probability error in select_action: {e}. Probs: {probs}", flush=True)
                idx = np.argmax(probs)
                
            return candidates[idx][0]

    def advance(self, action: Tuple[int, int, int], env: AzulEnv):
        """
        Advance the root to the child corresponding to the action taken.
        Reuses the subtree if the action was explored.
        
        Args:
            action: The action that was taken (source_idx, color, dest)
            env: The new environment state after the action
        """
        if action in self.root.children:
            # Reuse subtree: promote the child node to new root
            new_root = self.root.children[action]
            
            # Detach from parent (critical for garbage collection of old tree)
            new_root.parent = None
            
            # CRITICAL: Update the root's env to match the actual game state
            # The child's env was created during expand() with is_sim=True
            # We need to replace it with the real env to keep them synchronized
            new_root.env = env.clone()
            
            # Set as new root
            self.root = new_root
            
            # The subtree is preserved with all visit counts, values, and children!
        else:
            # Fallback: action not in tree (shouldn't happen normally)
            # This could occur if select_action() returned a fallback random action
            # Create fresh root from the new state
            self.root = MCTS.Node(env.clone(), parent=None, prior=1.0)