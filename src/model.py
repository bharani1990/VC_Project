import torch.nn as nn

class LappedTransform(nn.Module):
    def __init__(self, kernel_size=16):
        super(LappedTransform, self).__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size // 2), bias=False)
    
    def forward(self, x):
        x = self.conv(x)
        transformed_x = x[:, :, :-1, :-1]
        return transformed_x
