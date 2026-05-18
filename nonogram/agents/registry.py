"""
에이전트 레지스트리 — config의 agent.type으로 에이전트 클래스를 매핑.
"""

from nonogram.agents.base import BaseAgent

# Registry dict: type_name → class
AGENT_REGISTRY: dict[str, type[BaseAgent]] = {}


def register_agent(name: str):
    """에이전트 클래스를 레지스트리에 등록하는 데코레이터."""
    def decorator(cls):
        AGENT_REGISTRY[name] = cls
        return cls
    return decorator


def create_agent(config: dict, **kwargs) -> BaseAgent:
    """Config에서 에이전트 인스턴스를 생성.

    kwargs는 model, device 등 추가 인자를 전달.
    """
    agent_type = config["agent"]["type"]
    if agent_type not in AGENT_REGISTRY:
        available = list(AGENT_REGISTRY.keys())
        raise ValueError(f"Unknown agent type '{agent_type}'. Available: {available}")

    cls = AGENT_REGISTRY[agent_type]
    return cls(config=config, **kwargs)


# Import all agent modules to trigger registration
from nonogram.agents import dqn, double_dqn, ppo  # noqa: E402, F401
