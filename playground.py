import torch
import inspect
from typing import Tuple
# determine the number of inputs to a model
# define an arbitrary size tuple as a parameter to the model

from torchvision.datasets import DatasetFolder
from torch.utils.data import DataLoader

import csv
import numpy as np

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



# datasetFolder example
def csv_loader(path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        line = next(reader)
        return np.array(line, dtype=np.float32)
# Define any transformations (if needed)
# # In this case, we assume no additional transformations are needed
# transform = None
class NormalizeTransform:
    def __call__(self, sample):
        return (sample - sample.mean()) / sample.std()

transform = NormalizeTransform()
# Create the dataset
path = './data/sample/lang'
dataset = DatasetFolder(root=path, loader=csv_loader, extensions=('csv'),transform=transform)

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)
for data in dataloader:
    # print(data.shape)
    # will automatically be add a target tensor, but just zeros
    print(data[0].shape)
    break