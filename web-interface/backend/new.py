import torch

# Try loading with weights_only=False
try:
    state = torch.load("tech_model_final.pth", map_location="cpu", weights_only=False)
    print("✓ Loaded successfully with weights_only=False")
    print(f"Keys in state dict: {list(state.keys())[:5]}...")  # Show first 5 keys
except Exception as e:
    print(f"✗ Failed to load: {e}")

# Try loading with weights_only=True
try:
    state = torch.load("tech_model_final.pth", map_location="cpu", weights_only=True)
    print("✓ Loaded successfully with weights_only=True")
except Exception as e:
    print(f"✗ Failed with weights_only=True: {e}")