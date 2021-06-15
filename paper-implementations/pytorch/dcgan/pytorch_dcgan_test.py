import torch
import  torch.nn as nn


z_dim = 100
g_channels = 64
img_channels = 3


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.G = nn.Sequential(
            nn.ConvTranspose2d(z_dim, g_channels * 16, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0)), # (N, g_channels * 16, 4, 4)
            nn.ConvTranspose2d(g_channels * 16, g_channels * 8, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)), # (N, g_channels * 8, 8, 8)
            nn.ConvTranspose2d(g_channels * 8, g_channels * 4, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)), # (N, g_channels * 4, 16, 16)
            nn.ConvTranspose2d(g_channels * 4, g_channels * 2, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)), # (N, g_channels * 2, 32, 32)
            nn.ConvTranspose2d(g_channels * 2, img_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)), # (N, img_channels, 64, 64)
        )

    def forward(self, z):
        return self.G(z)


g = Generator()
z = torch.randn((1, 100, 1, 1))
output = g(z)
assert output.shape == (1, 3, 64, 64), "Shape is not correct"
print(output.shape)