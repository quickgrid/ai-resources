"""ProGAN Model definition.

Attributes:
    factors (List): Scaling factor for filter count. Following paper mentioned filters current values scale the image
        till 256x256 resolution rgb image instead of paper mentioned 1024x1024. Add `1/16` and `1/32` to use that.

References:
- Code based on, https://www.youtube.com/watch?v=nkQHASviYac
- ProGAN paper, https://arxiv.org/abs/1710.10196
"""

import torch
import torch.nn as nn
import torch.functional.F as F
from torch import log2


factors = [1, 1, 1, 1, 1/2, 1/4, 1/8]


class WSConv2D(nn.Module):
    def __init__(self, in_channel, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0), gain=2):
        """Weight scaled convolution.

        Look into ProGAN paper, section 4.1 for weight initialization explanation.
        Weights are initialized from a normal distribution with mean 0, variance 1 and scaled at runtime with equation,
        `w_hat_i = w_i / c`.

        Kaiming he normal equation, https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_normal_.
        `std = gain / sqrt(fan_mode)`, using, `fan_mode = fan_in`, number of input units should be the number
        of channels multiplied by height, width. In this case (height, width) is (kernel_size, kernel_size).
        So, `fan_in = kernel_size * kernel_size * in_channels`.

        Further details on bias, weight initialization application found in section A.1.

        Args:
            in_channel:
            out_channels:
            kernel_size:
            stride:
            padding:
            gain: Scaling factor for weight initialization.

        Returns:
            Scaled convolution.
        """

        super(WSConv2D, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channels, kernel_size, stride, padding)
        self.scale = (gain / (in_channel * kernel_size ** 2)) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        # Initialize conv layer parameter
        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)


class PixelNorm(nn.Module):
    def __init__(self):
        """Pixel wise feature vector normalization for generator.

        Look into ProGAN paper, section 4.2 for equation explanation and epsilon value.
        Normalization is performed depth wise for each feature map.

        Returns:
            pixel wise normalized feature vector
        """

        super(PixelNorm, self).__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_pixel_norm=True):
        super(ConvBlock, self).__init__()
        self.conv1 = WSConv2D(in_channels, out_channels)
        self.conv2 = WSConv2D(in_channels, out_channels)
        self.use_pixel_norm = use_pixel_norm
        self.leaky = nn.LeakyReLU(0.2)
        self.pixel_norm = PixelNorm()

    def forward(self, x):
        x = self.leaky(self.conv1(x))
        x = self.pixel_norm(x) if self.use_pixel_norm else x
        x = self.leaky(self.conv2(x))
        x = self.pixel_norm(x) if self.use_pixel_norm else x
        return x


class Generator(nn.Module):
    def __init__(self, z_dim, num_channels, img_channels=3):
        """Generator.

        Pixel norm application details, LeakyRelu with value is mentioned in section A.1 in ProGAN paper.
        Pixel norm is applied after each 3x3 convolution as mentioned in the paper.

        The network architecture is available in Table 2. Here, initial is the first block in generator
        diagram as it is different than architecture of other blocks.

        Figure 2, detail part `toRGB`, `fromRGB` details. Both use 1x1 convolutions.
        `toRGB` projects feature map to RGB 3 channels and `fromRGB` does the opposite.
        """

        super(Generator, self).__init__()
        self.initial = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(z_dim, num_channels, kernel_size=(4, 4), stride=(1, 1), padding=(0, 0)),
            nn.LeakyReLU(0.2),
            WSConv2D(num_channels, num_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.LeakyReLU(0.2),
            PixelNorm()
        )

        self.initial_rgb = WSConv2D(num_channels, img_channels, kernel_size=(1, 1), stride=(1, 1))

        self.progressive_blocks, self.rgb_layers = nn.ModuleList(), nn.ModuleList()
        self.rgb_layers.append(self.initial_rgb)

        for i in range(len(factors) - 1):
            conv_in_c = int(num_channels * factors[i])
            conv_out_c = int(num_channels * factors[i+1])
            self.progressive_blocks.append(ConvBlock(conv_in_c, conv_out_c))
            self.rgb_layers.append(WSConv2D(conv_out_c, img_channels, kernel_size=(1, 1), stride=(1, 1)))

    def fade_in(self, alpha, upscaled, generated):
        """Section A.1 mentions training and generated images are represented in [-1,1].

        Tanh activation meets the requirement close enough with range of (-1, 1).
        Figure 2, diagram (b) provides equation residual block like sum.

        Args:
            alpha: Goes from 0 to 1.
            upscaled:
            generated:

        Returns:
            Faded data between generated and nearest neighbour upsampled.
        """

        return torch.tanh(alpha * generated + (1 - alpha) * upscaled)

    def forward(self, x, alpha, steps):
        """Upsample 2x and pass to next progressive layer.

        Look into Table 2 and figure 2 diagram for building generator model.

        Args:
            x:
            alpha:
            steps:

        Returns:
            Generated image.
        """
        x = self.initial(x)

        if steps == 0:
            return self.initial_rgb(x)

        upscaled = None
        for step in range(steps):
            upscaled = F.interpolate(x, scale_factor=2, mode='nearest')
            x = self.progressive_blocks[step](upscaled)

        final_upscaled = self.rgb_layers[steps - 1](upscaled)
        final_output = self.rgb_layers[steps](x)
        return self.fade_in(alpha, final_upscaled, final_output)


class Discriminator(nn.Module):
    def __init__(self, z_dim, num_channels, img_channels):
        """Discriminator more correctly critic in this case.

        Args:
            z_dim:
            num_channels:
            img_channels:
        """
        super(Discriminator, self).__init__()
        self.progressive_blocks, self.rgb_layers = nn.ModuleList(), nn.ModuleList()
        self.leaky = nn.LeakyReLU(0.2)

        for i in range(len(factors) - 1, 0, -1):
            conv_in_c = int(num_channels * factors[i])
            conv_out_c = int(num_channels * factors[i - 1])
            self.progressive_blocks.append(ConvBlock(conv_in_c, conv_out_c, use_pixel_norm=False))
            self.rgb_layers.append(WSConv2D(img_channels, conv_in_c, kernel_size=(1, 1), stride=(1, 1)))

    def fade_in(self):
        pass

    def minibatch_std(self, x):
        pass

    def forward(self, x):
        pass
