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

    def __init__(self, env: AzulEnv, model: Any, simulations: int = 100, cpuct: float = 1.0, single_player_mode: bool = False):
        """
        env: an AzulEnv instance to clone for rollouts.
        simulations: number of MCTS simulations per move.
        cpuct: exploration constant.
        single_player_mode: if True, backpropagates values without flipping signs (assumes general sum / cooperative environment logic or own-score concern only).
        """
        self.root = MCTS.Node(env.clone(), parent=None, prior=1.0)
        self.model = model
        self.simulations = simulations
        self.cpuct = cpuct
        self.single_player_mode = single_player_mode

    def select(self) -> Tuple['MCTS.Node', list]:
        """
        Select a leaf node to expand.
        Returns the leaf node and the path of nodes taken.
        """
        node = self.root
        path = [node]
        # Traverse until we find a leaf
        # Traverse until we find a leaf
        while node.children:
            # Check if Single Player Mode AND Opponent Turn
            if self.single_player_mode and node.player != self.root.player:
                # Opponent Node -> Treated as Random Environment Transition
                # Do NOT use UCB. Pick randomly based on priors (or uniform if no priors/visits)
                # We simply follow the visit distribution if available, or uniform if not?
                # Actually, environment nodes should just be sampled.
                # Let's pick a random child.
                # Since we don't store "Environment Children" differently, we just pick from keys.
                action = random.choice(list(node.children.keys()))
                node = node.children[action]
            else:
                # Standard UCB (Agent Turn or Standard MCTS)
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
        # --- SINGLE PLAYER MODE: OPPONENT SKIP LOGIC ---
        if self.single_player_mode and node.player != self.root.player:
             # We are expanding an Opponent Node (Environment Node).
             # We do NOT generally want to evaluate this node with the network 
             # because the network predicts 'Opponent Score' (Value relative to current player).
             # We want 'Agent Score'.
             # Strategy:
             # 1. Generate all children (Environment transitions).
             # 2. Pick ONE random child to simulate "Reaction".
             # 3. Recursively expand THAT child (Agent Node).
             # 4. Return the value from that Agent Node.
             # Note: This means we only add ONE path deep from an opponent leaf. 
             # Future visits might expand siblings? No, `expand` is only called once per leaf.
             # So we must create all children but only evaluate one?
             # Yes.
             
             # 1. Generate children (Uniform priors for environment)
            valid_actions = node.env.get_valid_actions()
            if not valid_actions:
                return 0.0 # Terminal? Should be caught in run()

            uniform_prior = 1.0 / len(valid_actions)
            for action in valid_actions:
                 new_env = node.env.clone()
                 new_env.step(action, is_sim=True)
                 node.children[action] = MCTS.Node(new_env, parent=node, prior=uniform_prior)
            
            # 2. Pick Child based on Opponent Model (Smart Opponent)
            # We want to simulate a realistic opponent, not a random one.
            # So we query the model for the opponent's policy.
            
            # Predict policy for opponent
            obs = node.env._get_obs()
            obs_flat = node.env.encode_observation(obs)
            action_mask = np.zeros(node.env.action_size, dtype=np.float32)
            for action in valid_actions:
                idx = node.env.action_to_index(action)
                action_mask[idx] = 1.0
                
            pi_logits, _ = self.model.predict(np.array([obs_flat]), np.array([action_mask]))
            logits = pi_logits[0]
            
            # Compute probabilities
            exp_logits = np.exp(logits - np.max(logits))
            total = exp_logits.sum()
            if total > 0:
                priors = exp_logits / total
            else:
                priors = action_mask / action_mask.sum()
                
            # Filter for valid actions
            valid_probs = []
            for action in valid_actions:
                idx = node.env.action_to_index(action)
                valid_probs.append(priors[idx])
                
            valid_probs = np.array(valid_probs)
            total_valid = valid_probs.sum()
            if total_valid > 0:
                valid_probs /= total_valid
            else:
                valid_probs = np.ones(len(valid_actions)) / len(valid_actions)
                
            # Sample action
            try:
                # Assuming valid_probs sums to 1 (renormalized)
                chosen_idx = np.random.choice(len(valid_actions), p=valid_probs)
                chosen_action = valid_actions[chosen_idx]
            except ValueError:
                chosen_action = random.choice(valid_actions)

            random_child = node.children[chosen_action]
            
            # 3. Recurse (Expand the Agent Node)
            # Check if terminal
            obs = random_child.env._get_obs()
            done = random_child.env.done # env.step updates done
            # Check implicit done (wall complete)
            if not done: 
                 done = any(all(cell != -1 for cell in row) for p in random_child.env.players for row in p['wall'])
            
            if done:
                # Terminal state reached after opponent move
                scores = random_child.env.get_final_scores()
                # Value = Normalized Agent Score
                agent_idx = self.root.player
                agent_score = scores[agent_idx]
                return np.clip(agent_score / 100.0, -1.0, 1.0)
            else:
                return self.expand(random_child) 

        # --- STANDARD AGENT EXPANSION ---
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
            if self.single_player_mode:
                # Single Player / Optimization Mode:
                # We assume 'value' is the absolute score/utility for the Agent (root player).
                # All nodes maximize this value (or we are optimistic/cooperative).
                # We do NOT flip signs.
                node.value_sum += value
            else:
                # Standard Zero-Sum Mode:
                # If the node's player is the same as the leaf player, they want to MAXIMIZE this value.
                # If different, they want to MINIMIZE it (so we add negative value).
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
                
                if self.single_player_mode:
                     # Single Player / Optimization Mode:
                     # Value is the normalized score of the Agent (root.player)
                     # regardless of who the 'leaf.player' is (backpropagate will handle perspective)
                     # Wait, backpropagate assumes 'value' is for 'leaf.player'?
                     # Let's check backpropagate: 
                     # "if node.player == leaf_player: node.value_sum += value"
                     # So if we pass AgentScore, and leaf_player is Agent, it adds.
                     # If leaf_player is Opponent, it subtracts? NO.
                     # In single_player_mode backpropagate: "node.value_sum += value" (ALWAYS ADDS)
                     # So we just need to pass the Agent's Score as 'value'.
                     agent_idx = self.root.player
                     agent_score = scores[agent_idx]
                     value = np.clip(agent_score / 100.0, -1.0, 1.0)
                else:
                    # Standard Zero-Sum Logic:
                    if p0_score > p1_score:
                        value = (1.0 if leaf.player == 0 else -1.0)
                    elif p1_score > p0_score:
                        value = (1.0 if leaf.player == 1 else -1.0)
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
            
            # NEW: Validate that the cached children are still legal in the new environment.
            # This is necessary because stochastic events (like factory refill) might have happened differently
            # in the real game vs the simulation, rendering previously valid actions illegal.
            real_valid_actions = set(env.get_valid_actions())
            cached_actions = set(new_root.children.keys())
            
            if not cached_actions.issubset(real_valid_actions):
                # Stale tree detected (diverged environment). Reset this node to trigger re-expansion.
                new_root.children = {}
                new_root.visits = 0
                new_root.value_sum = 0
                new_root.prior = 1.0 # Reset prior? Or keep? Resetting is safer.
            
            # Set as new root
            self.root = new_root
            
            # The subtree is preserved with all visit counts, values, and children!
        else:
            # Fallback: action not in tree (shouldn't happen normally)
            # This could occur if select_action() returned a fallback random action
            # Create fresh root from the new state
            self.root = MCTS.Node(env.clone(), parent=None, prior=1.0)