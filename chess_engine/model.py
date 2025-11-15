# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


class ChessNet(nn.Module):
    def __init__(self, planes=18*41, moves=4672):
        super().__init__()

        self.conv_in = nn.Conv2d(planes, 128, 3, padding=1)
        self.bn_in = nn.BatchNorm2d(128)

        self.res_layers = nn.ModuleList([ResidualBlock(128) for _ in range(12)])

        # Policy
        self.conv_policy = nn.Conv2d(128, 32, 1)
        self.bn_policy = nn.BatchNorm2d(32)
        self.fc_policy = nn.Linear(32 * 8 * 8, moves)

        # Value
        self.conv_value = nn.Conv2d(128, 8, 1)
        self.bn_value = nn.BatchNorm2d(8)
        self.fc_value_1 = nn.Linear(8 * 8 * 8, 128)
        self.fc_value_2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.bn_in(self.conv_in(x)))
        for r in self.res_layers:
            x = r(x)

        # Policy head
        p = F.relu(self.bn_policy(self.conv_policy(x)))
        p = p.view(p.size(0), -1)
        p = self.fc_policy(p)

        # Value head
        v = F.relu(self.bn_value(self.conv_value(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.fc_value_1(v))
        v = torch.tanh(self.fc_value_2(v))

        return p, v
