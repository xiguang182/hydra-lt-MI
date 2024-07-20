import torch
import inspect
from typing import Tuple
# determine the number of inputs to a model
# define an arbitrary size tuple as a parameter to the model
class MyModel(torch.nn.Module):
    def __init__(self, test_param: Tuple[int, ...]) -> None:
        super(MyModel, self).__init__()
        self.fc1 = torch.nn.Linear(10, 5)
        self.fc2 = torch.nn.Linear(5, 1)
        print(f'Initialized with test_param size: {len(test_param)}')

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

model = MyModel((1,2,3,4,5))

# Inspect the forward method
forward_signature = inspect.signature(model.forward)
print(forward_signature)

# Get the number of inputs
num_inputs = len(forward_signature.parameters)
print(f"The model takes {num_inputs} inputs.")
