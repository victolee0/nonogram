import argparse
import os
import yaml
from pathlib import Path

def load_config(config_path: str, overrides: list = None) -> dict:
    """YAML 설정 파일을 로드하고 configs/default.yaml과 병합한 뒤 CLI 인자로 오버라이드."""
    # 1. default.yaml 로드
    default_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    if default_path.exists():
        with open(default_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}
        
    # 2. 지정된 config_path 로드 및 병합
    with open(config_path, 'r', encoding='utf-8') as f:
        user_config = yaml.safe_load(f) or {}
        
    def deep_merge(target: dict, source: dict):
        for k, v in source.items():
            if isinstance(v, dict) and k in target and isinstance(target[k], dict):
                deep_merge(target[k], v)
            else:
                target[k] = v
                
    deep_merge(config, user_config)
    
    # 3. CLI 오버라이드
    if overrides:
        for i in range(0, len(overrides), 2):
            key = overrides[i].lstrip('-').replace('.', ':')
            val = overrides[i+1]
            if val.lower() == 'true': val = True
            elif val.lower() == 'false': val = False
            else:
                try:
                    if '.' in val: val = float(val)
                    else: val = int(val)
                except: pass
            
            parts = key.split(':')
            curr = config
            for p in parts[:-1]:
                curr = curr.setdefault(p, {})
            curr[parts[-1]] = val
    return config

def save_config(config: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False)

def make_base_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    return parser

def resolve_device(device_str: str):
    import torch
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)

def get_experiment_dir(config: dict) -> Path:
    training_cfg = config.get("training", {})
    exp_name = training_cfg.get("exp_name", "default")
    agent_type = config.get("agent", {}).get("type", "unknown")
    model_type = config.get("model", {}).get("type", "unknown")
    N = config.get("env", {}).get("N", 0)
    path = Path("runs") / f"{exp_name}_{agent_type}_{model_type}_N{N}"
    path.mkdir(parents=True, exist_ok=True)
    (path / "checkpoints").mkdir(exist_ok=True)
    return path
