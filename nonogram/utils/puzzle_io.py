"""
퍼즐 입출력 유틸리티.

YAML 파일에서 퍼즐(hints)을 읽고, 결과를 저장.
CLI 문자열에서 hints를 파싱하는 기능도 제공.
"""

from pathlib import Path
from typing import Optional

import yaml


def load_puzzle(path: str | Path) -> dict:
    """YAML 파일에서 퍼즐 로드.

    Expected format:
        row_hints:
          - [1, 3]
          - [2]
          ...
        col_hints:
          - [4]
          - [1, 1]
          ...

    Returns:
        dict with 'row_hints', 'col_hints', and optionally 'N'
    """
    path = Path(path)
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    row_hints = data["row_hints"]
    col_hints = data["col_hints"]

    if len(row_hints) != len(col_hints):
        raise ValueError(
            f"row_hints has {len(row_hints)} rows but col_hints has "
            f"{len(col_hints)} columns — the board must be square."
        )

    return {
        "row_hints": row_hints,
        "col_hints": col_hints,
        "N": len(row_hints),
    }


def save_puzzle(path: str | Path, row_hints: list, col_hints: list) -> None:
    """퍼즐을 YAML 파일로 저장."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "row_hints": row_hints,
        "col_hints": col_hints,
    }
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def parse_hints_str(hints_str: str) -> list[list[int]]:
    """CLI 문자열에서 hints 파싱.

    Format: "1,3;2;1,1;..." (세미콜론으로 줄 구분, 쉼표로 값 구분)

    예:
        "1,3;2;1,1" → [[1,3], [2], [1,1]]
        "0;1;2"     → [[0], [1], [2]]
    """
    result = []
    for line in hints_str.strip().split(";"):
        values = [int(x.strip()) for x in line.split(",")]
        result.append(values)
    return result
