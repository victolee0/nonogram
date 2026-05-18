"""보드 시각화 유틸리티."""

import numpy as np

FILLED = '■'
EMPTY = '□'


def render_board(board: np.ndarray, row_hints: list, col_hints: list) -> str:
    """보드를 row/col hint와 함께 문자열로 렌더링.

    ■ = filled (+1), □ = empty/X (-1) or unknown (0)
    """
    N = board.shape[0]

    # Column hint header
    max_col_h = max((len(c) for c in col_hints), default=1)
    col_header_rows = []
    for layer in range(max_col_h):
        parts = []
        for c in col_hints:
            pad = max_col_h - len(c)
            if layer < pad:
                parts.append('  ')
            else:
                parts.append(f'{c[layer - pad]:1d} ')
        col_header_rows.append(''.join(parts))

    # Row hint width
    row_hint_strs = [' '.join(str(x) for x in h) for h in row_hints]
    row_hint_width = max((len(s) for s in row_hint_strs), default=0)

    out_lines = []
    indent = ' ' * (row_hint_width + 2)
    for header in col_header_rows:
        out_lines.append(indent + header)

    for r in range(N):
        cells = []
        for c in range(N):
            v = board[r, c]
            cells.append(FILLED if v == 1 else EMPTY)
        out_lines.append(f'{row_hint_strs[r]:>{row_hint_width}}  ' + ' '.join(cells))

    return '\n'.join(out_lines)


def render_board_debug(board: np.ndarray) -> str:
    """디버깅용 보드 렌더링. # = filled, . = X, ? = unknown."""
    chars = {1: '#', -1: '.', 0: '?'}
    return '\n'.join(
        ' '.join(chars[int(v)] for v in board[r, :])
        for r in range(board.shape[0])
    )
