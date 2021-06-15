import os

import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SineLayer(nn.Module):
    def __init__(self, input_units, output_units, bias=True, is_first=False, omega_0=30, weight_init_type='uniform'):
        """Implementation of sine layer.

        Calculates omega_0 * Wx + b as proposed in paper section 3.2 last paragraph.
        If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the nonlinearity.
        Different signals may require different omega_0 in the first layer - this is a hyperparameter.

        See below for more details,
        https://github.com/vsitzmann/siren
        https://arxiv.org/abs/2006.09661

        Args:
            input_units:
            output_units:
            bias:
            is_first:
            omega_0:
            weight_init_type:

        Returns:
            omega_0 * Wx + b followed by sine activation.

        Raises:
            Exception: Unsupported weight initialization type.
        """

        super(SineLayer, self).__init__()

        self.omega_0 = omega_0
        self.is_first = is_first
        self.input_units = input_units
        self.weight_init_type = weight_init_type

        self.linear = nn.Linear(in_features=input_units, out_features=output_units, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.weight_init_type == 'uniform':
                if self.is_first:
                    torch.nn.init.uniform_(self.linear.weight, -1 / self.input_units, 1 / self.input_units)
                else:
                    torch.nn.init.uniform_(self.linear.weight, -np.sqrt(6 / self.input_units) / self.omega_0,
                                           np.sqrt(6 / self.input_units) / self.omega_0)
            elif self.weight_init_type == 'normal':
                if self.is_first:
                    torch.nn.init.normal_(self.linear.weight, -1 / self.input_units, 1 / self.input_units)
                else:
                    torch.nn.init.normal_(self.linear.weight, -np.sqrt(6 / self.input_units) / self.omega_0,
                                          np.sqrt(6 / self.input_units) / self.omega_0)
            else:
                raise Exception(F'{self.weight_init_type} is not valid, only `uniform` or `normal` allowed.')

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class SIREN(nn.Module):
    def __init__(self, input_units, hidden_units, hidden_layer_count, output_units, first_omega_0=30, hidden_omega_0=30,
                 outermost_linear=True, weight_init_type='uniform'):
        """SIREN implementation width only linear layers.

        Args:
            input_units:
            hidden_units:
            hidden_layer_count:
            output_units:
            first_omega_0:
            hidden_omega_0:
            outermost_linear:
            weight_init_type:

        Returns:
            Predicted pixel color output and pixel coordinates.

        Raises:
            Exception: Unsupported weight initialization type.
        """
        super(SIREN, self).__init__()

        self.hidden_layer_count = hidden_layer_count
        self.input_layer = SineLayer(input_units=input_units, output_units=hidden_units, is_first=True,
                                     omega_0=first_omega_0, weight_init_type=weight_init_type)

        # self.hidden_layer = SineLayer(input_units=hidden_units, output_units=hidden_units, is_first=False, omega_0=hidden_omega_0)
        self.hidden_layers = nn.ModuleList(
            [SineLayer(input_units=hidden_units, output_units=hidden_units, is_first=False, omega_0=hidden_omega_0) for
             i in range(hidden_layer_count)])

        if outermost_linear:
            self.output_layer = nn.Linear(in_features=hidden_units, out_features=output_units)

            with torch.no_grad():
                if weight_init_type == 'uniform':
                    torch.nn.init.uniform_(self.output_layer.weight, -np.sqrt(6 / hidden_units) / hidden_omega_0,
                                           np.sqrt(6 / hidden_units) / hidden_omega_0)
                elif weight_init_type == 'normal':
                    torch.nn.init.uniform_(self.output_layer.weight, -np.sqrt(6 / hidden_units) / hidden_omega_0,
                                           np.sqrt(6 / hidden_units) / hidden_omega_0)
                else:
                    raise Exception(F'{weight_init_type} is not supported. Try `uniform` or `normal`.')
        else:
            self.output_layer = SineLayer(input_units=hidden_units, output_units=output_units, is_first=False,
                                          omega_0=hidden_omega_0)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)

        x = self.input_layer(coords)

        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)

        output = self.output_layer(x)

        return output, coords


def get_mesh_grid(image_height, image_width, image_channels=3):
    """Generates 2D and 3D mesh grid of image indices.

    Works with non square image shapes based on image height and width.
    For image channels set to 1 it generates values from -1 to 1 based on image (height, width).
    For image channels set to 3 it generates values from -1 to 1 based on image (height, width, channels).

    Args:
        image_height: Input image height
        image_width: Input image width
        image_channels: Input image channels. Only RGB or Greyscale allowed.

    Returns:
        2D or 3D mesh grid based on number of image_channels.

    Raises:
        Exception unsupported image channels.
    """

    if image_channels == 3:
        t1 = tuple([torch.linspace(-1, 1, steps=image_height)])
        t2 = tuple([torch.linspace(-1, 1, steps=image_width)])
        t3 = tuple([torch.linspace(-1, 1, steps=image_channels)])
        mesh_grid = torch.stack(torch.meshgrid(*t1, *t2, *t3), dim=-1)
        mesh_grid = mesh_grid.reshape(-1, image_channels)
        return mesh_grid
    elif image_channels == 1:
        t1 = tuple([torch.linspace(-1, 1, steps=image_height)])
        t2 = tuple([torch.linspace(-1, 1, steps=image_width)])
        mesh_grid = torch.stack(torch.meshgrid(*t1, *t2), dim=-1)
        mesh_grid = mesh_grid.reshape(-1, 2)
        return mesh_grid
    else:
      raise Exception(F'{image_channels} not allowed try 1 or 3.')


loaded_model = SIREN(input_units=2, output_units=3, hidden_units=128, hidden_layer_count=2, outermost_linear=True).to(device)
loaded_model.load_state_dict(torch.load('sample/siren_model.pt'))

loaded_model.eval()
with torch.no_grad():

  c_h, c_w, c_c = (346, 512, 1)

  model_input1 = get_mesh_grid(c_h, c_w, c_c)
  model_input1 = model_input1.to(device)

  model_output1, coords1 = loaded_model(model_input1)

  output_image_numpy_array = model_output1.cpu().view(c_h, c_w, 3).detach().numpy()

  scaled_output_image_numpy_array = np.uint8((output_image_numpy_array - output_image_numpy_array.min()) / (output_image_numpy_array.max() - output_image_numpy_array.min()) * 255)

  #im2 = Image.fromarray(scaled_output_image_numpy_array)
  #im2.save('/content/img2.jpg')

  fig, axes = plt.subplots(1,1, figsize=(18,6))
  axes.imshow(output_image_numpy_array)
  plt.show()