import os
import torch

a = torch.rand((5,5,769))
b = torch.rand((5,6,674))
ckpt_name = "Roberta_only.ckpt"
pth_name = ckpt_name.replace(".ckpt", ".pth")
checkpoint_path = os.path.join(os.path.dirname(__file__), "data", "checkpoints", ckpt_name)
pth_path = os.path.join(os.path.dirname(__file__), "data", "checkpoints", pth_name)

print(checkpoint_path)
checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
state_dict = checkpoint['state_dict']
print('weird o')
print(state_dict.keys())
# Clean up keys if needed
clean_state_dict = {k.replace("net.", "", 1): v for k, v in state_dict.items()}

print(clean_state_dict.keys())

torch.save(clean_state_dict, pth_path)

print("Saved to", pth_path)
ld_state_dict = torch.load(pth_path, weights_only=True)
print(ld_state_dict.keys())