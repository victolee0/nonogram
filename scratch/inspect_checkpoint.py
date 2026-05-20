import torch
import sys

def inspect(path):
    try:
        checkpoint = torch.load(path, map_location='cpu')
        print(f"Loaded checkpoint from: {path}")
        print("Keys:", checkpoint.keys())
        print("N:", checkpoint.get("N"))
        print("hidden_dim:", checkpoint.get("hidden_dim"))
        print("model_type:", checkpoint.get("model_type"))
        print("step_count:", checkpoint.get("step_count"))
        
        state_dict = checkpoint.get("model_state_dict", {})
        print("\nModel weights statistics:")
        for name, param in state_dict.items():
            if 'weight' in name or 'bias' in name:
                nan_count = torch.isnan(param).sum().item()
                max_val = param.max().item()
                min_val = param.min().item()
                mean_val = param.mean().item()
                print(f"  {name:30s} | shape: {str(list(param.shape)):15s} | nan: {nan_count:3d} | min: {min_val:+.4f} | max: {max_val:+.4f} | mean: {mean_val:+.4f}")
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    path = "runs/double_dqn_10x10_super_1d_double_dqn_dueling_N10/checkpoints/best.pt"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    inspect(path)
