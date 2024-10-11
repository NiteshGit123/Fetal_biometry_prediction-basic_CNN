import torch
import torch.nn as nn
import torch.nn.functional as F

class BiometryDetection(nn.Module):
    def __init__(self, H=270, W=400, num_classes=8):
        super(BiometryDetection, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.drop = nn.Dropout2d(p=0.2)

        self.fc_input_size = 24 * (H // 4) * (W // 4)
        self.fc = nn.Linear(self.fc_input_size, num_classes)

    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = F.dropout(self.drop(x), training=self.training)
        x = x.view(-1, self.fc_input_size)
        x = self.fc(x)
        return x
