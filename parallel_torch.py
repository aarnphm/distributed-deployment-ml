import torch
import torch.nn as nn
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel as DDP

class SimpleNet(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(SimpleNet, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        pred = self.linear2(h_relu)
        return self.relu(pred)