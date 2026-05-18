import math
import numpy as np
import torch
from utils import is_feasible

class Node:
    """
    MCTS Node for 2D Nonogram.
    """
    __slots__ = [
        'state', 'parent', 'action_taken', 'children',
        'visit_count', 'value_sum', 'prior_p',
        'is_expanded', 'is_terminal', 'reward'
    ]

    def __init__(self, state, parent=None, action_taken=None, prior_p=0.0):
        self.state = state  # 2D board (N, N)
        self.parent = parent
        self.action_taken = action_taken  # Tuple (row, col, value) where value is 1 or -1
        
        self.children = {}  # action -> Node
        
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior_p = prior_p
        
        self.is_expanded = False
        self.is_terminal = False
        self.reward = 0.0

    @property
    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def puct_score(self, c_puct=1.5):
        """AlphaZero PUCT formula for action selection."""
        parent_visits = self.parent.visit_count if self.parent is not None else 1
        exploration = c_puct * self.prior_p * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.q_value + exploration


class AlphaZeroMCTS:
    """
    MCTS integrated with the Dual-Attention Neural Network and Perfect Environment Simulator.
    """
    def __init__(self, network, row_hints, col_hints, c_puct=2.5, num_simulations=200,
                 dirichlet_alpha=0.3, dirichlet_epsilon=0.25, action_masking=True, device='cpu'):
        self.network = network
        self.row_hints = row_hints  # List of lists (e.g., [[2], [1, 1], ...])
        self.col_hints = col_hints
        self.N = len(row_hints)
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.action_masking = action_masking
        self.device = device
        
        # Precompute hint tensors for the neural network
        self.k = (self.N + 1) // 2
        self.row_hints_padded = self._pad_hints(row_hints)
        self.col_hints_padded = self._pad_hints(col_hints)

    def _pad_hints(self, hints_list):
        padded = []
        for h in hints_list:
            actual = [int(x) for x in h if int(x) > 0]
            pad_len = self.k - len(actual)
            padded.append([0] * pad_len + actual)
        return torch.tensor(padded, dtype=torch.long, device=self.device).unsqueeze(0)

    def search(self, initial_board, temperature=1.0):
        """
        Run MCTS starting from the current board state.
        Returns a dict of action probabilities pi(a|s).
        
        Args:
            initial_board: Current board state (N, N)
            temperature: Controls action selection randomness.
                         >0: proportional to visit_count^(1/T)
                         0: argmax (greedy)
        """
        root = Node(state=initial_board.copy())
        
        # Expand root node first
        self._evaluate_and_expand(root)
        
        if not root.children:
            return {}
        
        # Add Dirichlet noise to root priors for exploration
        if self.dirichlet_epsilon > 0 and len(root.children) > 0:
            actions = list(root.children.keys())
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
            for i, action in enumerate(actions):
                child = root.children[action]
                child.prior_p = (
                    (1 - self.dirichlet_epsilon) * child.prior_p
                    + self.dirichlet_epsilon * noise[i]
                )
        
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            
            # 1. Select
            while node.is_expanded and not node.is_terminal:
                if not node.children:
                    break
                # Select the child with the highest PUCT score
                best_action = max(node.children.keys(), key=lambda a: node.children[a].puct_score(self.c_puct))
                node = node.children[best_action]
                search_path.append(node)
                
            # 2. Evaluate & Expand
            if not node.is_terminal:
                value = self._evaluate_and_expand(node)
            else:
                value = node.reward
                
            # 3. Backpropagate
            self._backpropagate(search_path, value)
            
        # 4. Compute target policy pi based on visit counts with temperature
        action_probs = {}
        total_visits = sum(child.visit_count for child in root.children.values())
        if total_visits == 0:
            return {}
        
        if temperature <= 0:
            # Greedy: pick the most visited action
            best_action = max(root.children.keys(), key=lambda a: root.children[a].visit_count)
            for action in root.children:
                action_probs[action] = 1.0 if action == best_action else 0.0
        else:
            # Temperature-scaled visit counts
            visit_counts = np.array([root.children[a].visit_count for a in root.children], dtype=np.float64)
            
            if temperature == 1.0:
                probs = visit_counts / visit_counts.sum()
            else:
                # visit_count^(1/T) with numerical stability
                log_counts = np.log(visit_counts + 1e-10)
                log_counts = log_counts / temperature
                log_counts -= log_counts.max()
                probs = np.exp(log_counts)
                probs = probs / probs.sum()
            
            for i, action in enumerate(root.children.keys()):
                action_probs[action] = float(probs[i])
            
        return action_probs

    def _evaluate_and_expand(self, node):
        board = node.state
        
        # --- [PERFECT SIMULATOR LOGIC] ---
        # Did the last action cause a mathematical contradiction?
        if node.action_taken is not None:
            r, c, v = node.action_taken
            
            # Prune branch instantly if the row or col became infeasible
            if not is_feasible(board[r, :], self.row_hints[r], self.N):
                node.is_terminal = True
                node.reward = -1.0
                return -1.0
            if not is_feasible(board[:, c], self.col_hints[c], self.N):
                node.is_terminal = True
                node.reward = -1.0
                return -1.0
                
        # Are all cells filled? (And since it passed the feasible check, it is a perfect solution)
        if np.all(board != 0):
            node.is_terminal = True
            node.reward = 1.0 
            return 1.0

        # --- [NEURAL NETWORK EVALUATION] ---
        board_t = torch.tensor(board, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            policy_logits, value_t = self.network(board_t, self.row_hints_padded, self.col_hints_padded)
        
        value = value_t.item()
        
        # Find valid actions (empty cells only)
        empty_cells = np.argwhere(board == 0)
        
        if len(empty_cells) == 0:
            node.is_expanded = True
            return value
        
        # Build mask for valid actions only, then apply masked softmax
        logits_flat = policy_logits.reshape(2, self.N, self.N)  # (2, N, N)
        
        # Extract logits only for valid actions
        valid_logits = []
        valid_actions = []
        for r, c in empty_cells:
            for v_idx, v in enumerate([1, -1]):
                if self.action_masking:
                    # Check feasibility BEFORE expanding (in-place for speed)
                    board[r, c] = v
                    is_r_feasible = is_feasible(board[r, :], self.row_hints[r], self.N)
                    is_c_feasible = is_feasible(board[:, c], self.col_hints[c], self.N)
                    board[r, c] = 0  # Revert
                    
                    if is_r_feasible and is_c_feasible:
                        valid_logits.append(logits_flat[v_idx, r, c].item())
                        valid_actions.append((int(r), int(c), v))
                else:
                    # No masking: add all actions
                    valid_logits.append(logits_flat[v_idx, r, c].item())
                    valid_actions.append((int(r), int(c), v))
        
        if len(valid_actions) == 0:
            node.is_terminal = True
            node.reward = -1.0
            return -1.0
        
        # Softmax over valid actions only
        valid_logits = np.array(valid_logits, dtype=np.float64)
        valid_logits -= valid_logits.max()  # numerical stability
        exp_logits = np.exp(valid_logits)
        probs = exp_logits / exp_logits.sum()
        
        node.is_expanded = True
        
        # Expand node with all valid moves
        for i, (r, c, v) in enumerate(valid_actions):
            prob = float(probs[i])
            next_board = board.copy()
            next_board[r, c] = v
            action = (r, c, v)
            node.children[action] = Node(state=next_board, parent=node, action_taken=action, prior_p=prob)
                
        return value

    def _backpropagate(self, search_path, value):
        for node in reversed(search_path):
            node.visit_count += 1
            node.value_sum += value
            # Note: Unlike Go or Chess (zero-sum 2-player games) where value is inverted (-value) 
            # for the opponent, Nonogram is a single-player puzzle.
            # The win probability 'value' remains absolute for all nodes in the path!
