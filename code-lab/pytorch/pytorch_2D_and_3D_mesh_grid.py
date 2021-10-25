"""Mesh grid implementation to map pixels to locations.

For a single channel 2D image it can be identified with (x,y) for each location.
For a three channel 2D image it can be identified with (x,y,z) for each location.
"""


import torch


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


if __name__ == '__main__':
    # For 2d mesh grid there are two values (x,y) to represent a single location.
    # For 3d mesh grid there are three values (x,y,z) to represent a single or three locations.

    im_h, im_w = (200, 400)

    im_c = 3
    m_grid = get_mesh_grid(im_h, im_w, im_c)
    print(m_grid)
    print(m_grid.shape)

    assert m_grid.shape == (im_h * im_w * im_c, im_c), 'Invalid shape.'

    im_c = 1
    m_grid = get_mesh_grid(im_h, im_w, im_c)
    print(m_grid)
    print(m_grid.shape)

    assert m_grid.shape == (im_h * im_w * im_c, 2), 'Invalid shape.'
