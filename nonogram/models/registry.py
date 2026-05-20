"""
모델 레지스트리 — config의 model.type 문자열로 모델 클래스를 매핑.
"""

import torch.nn as nn

from nonogram.env.state import hint_length

# Registry dict: type_name → class
MODEL_REGISTRY: dict[str, type[nn.Module]] = {}


def register_model(name: str):
    """모델 클래스를 레지스트리에 등록하는 데코레이터."""
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def create_model(config: dict) -> nn.Module:
    """Config에서 모델 인스턴스를 생성.

    Args:
        config: 전체 config dict (model, env 섹션 사용)

    Returns:
        초기화된 nn.Module 인스턴스
    """
    model_type = config["model"]["type"]
    if model_type not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model type '{model_type}'. Available: {available}")

    cls = MODEL_REGISTRY[model_type]
    N = config["env"]["N"]
    model_cfg = config["model"]

    return cls(N=N, **{k: v for k, v in model_cfg.items() if k != "type"})


# Import all model modules to trigger registration
from nonogram.models import mlp, dueling, cnn, board_cnn, board_gnn, board_transformer, deep_crl, symmetry_gnn, board_cross_attn  # noqa: E402, F401

