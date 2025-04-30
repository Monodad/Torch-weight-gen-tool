import torch
import torch.nn as nn
import torch.nn.functional as F
import random

if __name__ == "__main__":
    # (batch_size, channels, height, width)
    Input = torch.randint(low=0, high=255, size=(1, 3, 32, 32))

    Input_zero_point = random.randint(1, 255)
    # (out_channels, in_channels, kernel_height, kernel_width)
    weight = torch.randint(low=0, high=255, size=(3, 3, 3, 3))
    base_line_output = F.conv2d(Input, weight, stride=1, padding=1)
