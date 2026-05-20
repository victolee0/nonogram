"""풀이기 레지스트리."""

SOLVER_REGISTRY: dict[str, type] = {}


def register_solver(name: str):
    def decorator(cls):
        SOLVER_REGISTRY[name] = cls
        return cls
    return decorator


def create_solver(config: dict, **kwargs):
    solver_type = config["solver"]["type"]
    if solver_type not in SOLVER_REGISTRY:
        available = list(SOLVER_REGISTRY.keys())
        raise ValueError(f"Unknown solver type '{solver_type}'. Available: {available}")
    cls = SOLVER_REGISTRY[solver_type]
    return cls(config=config, **kwargs)


# Import all solver modules to trigger registration
from nonogram.solvers import line_solver, board_solver, hybrid_solver  # noqa: E402, F401
