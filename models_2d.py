import torch
import torch.nn as nn
import torch.nn.functional as F

class AxialAttention(nn.Module):
    """
    Computes attention strictly along the row and column for each cell.
    This explicitly models the cross-logic constraints of Nonograms.
    """
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.row_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.col_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.norm_row = nn.LayerNorm(dim)
        self.norm_col = nn.LayerNorm(dim)

    def forward(self, x):
        # x shape: (B, N, N, dim)
        B, N, _, dim = x.shape
        
        # 1. Row Attention (Attend only to cells in the same row)
        # Reshape to treat each row as an independent sequence
        x_row = x.view(B * N, N, dim)
        out_row, _ = self.row_attn(x_row, x_row, x_row)
        out_row = out_row.view(B, N, N, dim)
        x = self.norm_row(x + out_row)  # Pre-norm residual

        # 2. Column Attention (Attend only to cells in the same column)
        # Transpose and reshape to treat each column as an independent sequence
        x_col = x.transpose(1, 2).reshape(B * N, N, dim)
        out_col, _ = self.col_attn(x_col, x_col, x_col)
        out_col = out_col.view(B, N, N, dim).transpose(1, 2)
        x = self.norm_col(x + out_col)  # Pre-norm residual

        return x


class GlobalAttention(nn.Module):
    """
    Computes standard global attention across all N*N cells to capture
    board-wide macro patterns.
    """
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, N, _, dim = x.shape
        x_flat = x.view(B, N * N, dim)
        out, _ = self.attn(x_flat, x_flat, x_flat)
        out = out.view(B, N, N, dim)
        return self.norm(x + out)


class NonogramTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        # User's brilliant idea: split into Axial and Global!
        self.axial = AxialAttention(dim, num_heads, dropout=dropout)
        self.global_attn = GlobalAttention(dim, num_heads, dropout=dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.axial(x)
        x = self.global_attn(x)
        x = self.norm(x + self.ffn(x))
        return x


class AlphaZeroNonogramNet(nn.Module):
    """
    Dual-head AlphaZero model for 2D Nonogram processing.
    """
    def __init__(self, N, hint_max_length, dim=128, num_heads=4, num_layers=4, dropout=0.1):
        super().__init__()
        self.N = N
        self.dim = dim
        
        # --- 1. State Embeddings ---
        # Cell state: 0 (unknown), 1 (filled), 2 (X, mapped from -1)
        self.cell_emb = nn.Embedding(3, dim)
        
        # Hint embeddings: max hint value is N, 0 is used for padding.
        self.hint_emb = nn.Embedding(N + 1, dim, padding_idx=0)
        
        # RNN to compress variable-length hint sequences into a single fixed vector
        self.hint_rnn = nn.GRU(dim, dim, batch_first=True)
        
        # Feature projector
        self.input_proj = nn.Linear(3 * dim, dim)
        self.input_norm = nn.LayerNorm(dim)

        # --- 2. Transformer Backbone ---
        self.blocks = nn.ModuleList([
            NonogramTransformerBlock(dim, num_heads, dropout=dropout) 
            for _ in range(num_layers)
        ])

        # --- 3. Output Heads (AlphaZero Style) ---
        # Policy Head: Predicts action probabilities for each cell (whether to put 1 or X)
        self.policy_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 2) # Output logits for (1, -1) actions per cell
        )

        # Value Head: Predicts win probability in range [-1, 1]
        self.value_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Tanh()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Xavier/Kaiming initialization for better training start."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)

    def forward(self, board, row_hints, col_hints):
        """
        board: (B, N, N) values in {-1, 0, 1}
        row_hints: (B, N, hint_max_length)
        col_hints: (B, N, hint_max_length)
        """
        B = board.size(0)
        
        # Map -1 to 2 for embedding lookup
        board_mapped = board.clone()
        board_mapped[board == -1] = 2
        cell_features = self.cell_emb(board_mapped.long()) # (B, N, N, dim)
        
        # Process Row Hints
        r_hints_flat = row_hints.view(B * self.N, -1)
        r_emb = self.hint_emb(r_hints_flat)
        _, r_hidden = self.hint_rnn(r_emb)
        row_features = r_hidden.squeeze(0).view(B, self.N, 1, self.dim)
        row_features = row_features.expand(-1, -1, self.N, -1) # Broadcast across columns
        
        # Process Column Hints
        c_hints_flat = col_hints.view(B * self.N, -1)
        c_emb = self.hint_emb(c_hints_flat)
        _, c_hidden = self.hint_rnn(c_emb)
        col_features = c_hidden.squeeze(0).view(B, 1, self.N, self.dim)
        col_features = col_features.expand(-1, self.N, -1, -1) # Broadcast across rows

        # Combine all features for each cell (x, y)
        combined = torch.cat([cell_features, row_features, col_features], dim=-1)
        x = self.input_norm(self.input_proj(combined)) # (B, N, N, dim)

        # Pass through Dual-Attention Transformer
        for block in self.blocks:
            x = block(x)

        # Output Policy (B, 2, N, N)
        policy_logits = self.policy_head(x).permute(0, 3, 1, 2)
        
        # Output Value (B, 1) via Global Average Pooling
        global_repr = x.view(B, self.N * self.N, self.dim).mean(dim=1)
        value = self.value_head(global_repr)

        return policy_logits, value
