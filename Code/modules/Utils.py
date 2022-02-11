import glob
import os

import numpy as np
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter


def create_patch_indexes(image_shape, patch_sizes):
    """Creates image patch coordinates for 3D dimension. Image dimensions must be divisible by patch size without
    remainder.

    Parameters
    ----------
    image_shape: tuple
    patch_sizes: tuple

    Returns
    -------
    patches: list
    """

    x, y, z = image_shape
    ps_x, ps_y, ps_z = patch_sizes
    assert x % ps_x == 0, "First dimension of the image must be divisible by patch size without remainder."
    assert y % ps_y == 0, "Second dimension of the image must be divisible by patch size without remainder."
    assert z % ps_z == 0, "Third dimension of the image must be divisible by patch size without remainder."
    px, py, pz = int(x / ps_x), int(y / ps_y), int(z / ps_z)

    sx, sy, sz = 0, 0, 0  # starting points

    patches = []

    for i in range(px):
        for j in range(py):
            for u in range(pz):
                patches.append([[sx, sy, sz], [sx + ps_x, sy + ps_y, sz + ps_z]])
                sz += ps_z
            sz = 0
            sy += ps_y
        sy = 0
        sx += ps_x

    return patches


def get_file_names(path_data):
    """List the files in sub directories.

    Parameters
    ----------
    path_data: str
        Path of the data folder.

    Returns
    -------
    paths: dict
        {'sub_name': [image_name, mask_name]}
    """

    path = os.path.join(path_data, "**", "*.nii.gz")
    files = glob.glob(path, recursive=True)
    files = sorted(files)

    def pairwise(files):
        iterator = iter(files)

        return zip(iterator, iterator)

    paths = {}
    for image, mask in pairwise(files):
        paths[image.split(os.sep)[1]] = [image, mask]

    return paths


def create_onehot_mask(pred_shape, mask, device):
    """Creates onehot mask for multidimensional mask.

    Parameters
    ----------
    pred_shape: tuple
        (bs,number_of_classes,x,y)
                    or
        (bs,number_of_classes,x,y,z)
    mask: torch.tensor
        (bs,1,x,y)
            or
        (bs,1,x,y,z)
    device: str
        'cuda' or 'cpu'...
    Returns
    -------
    mask_onehot: torch.tensor
        [bs,number_of_classes,x,y,z]
    """

    mask_onehot = torch.zeros(pred_shape, requires_grad=False).to(device)
    mask = mask.long()
    mask_onehot.scatter_(1, mask, 1)

    return mask_onehot


def init_weights_kaiming(m):
    """Initialize model weights with uniform distribution.
    Find here why kaiming is better for ReLU activations: https://towardsdatascience.com/954fb9b47c79

    Parameters
    ----------
    m: torch.model

    Returns
    -------
    None
    """

    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0)


def calculate_dice_score(pred, mask, smooth=1e-5):
    """This method calculates the dice score for each given class.

    Parameters
    ----------
    pred: torch.Tensor
        [bs, number_of_classes, x, y]
    mask: torch.Tensor
        [bs, number_of_classes, x, y]
    Returns
    -------
    dice_scores: torch.Tensor
        [n_classes, dice_scores] Dice scores of the given classes.
    """

    (bs, n_classes, x, y) = pred.shape

    with torch.no_grad():
        pred = pred.permute(1, 0, 2, 3).contiguous()
        pred = pred.view(n_classes, bs * x * y)

        mask = mask.permute(1, 0, 2, 3).contiguous()
        mask = mask.view(n_classes, bs * x * y)

    intersection = (pred * mask).sum(-1)
    denominator = (pred + mask).sum(-1)

    dice_scores = (2 * intersection.clamp(min=smooth)) / denominator.clamp(min=smooth)

    return dice_scores


class TensorboardModules:
    """This class consists of methods that allow adding data to Tensorboard.
    """

    def __init__(self, log_dir="events"):
        """Initialize summary writer.

        Parameters
        ----------
        log_dir: str
            Storage path of the event files.
        """

        self.writer = SummaryWriter(log_dir)

    def close(self):
        """Close summary writer."""

        self.writer.close()

    def add_images(self, tag, image, slices):
        """Adds images to tensorboard.

        Parameters
        ----------
        tag: str
            Data identifier.
        image: torch.Tensor
            [x, y, z] shape.
        slices: dict of ints
            Start and end point of the image slice, and step size:

        Returns
        -------
        None
        """

        start, end, step = slices

        if type(image) == np.ndarray:
            image = torch.Tensor(image)

        grid_data = image[:, :, start:end:step].permute(2, 0, 1)

        img_grid = torchvision.utils.make_grid(grid_data.view(10, 1, 256, 256))
        self.writer.add_image(tag, img_grid)

    def add_graph(self, model, input_size, device):
        """Creates an input to model and adds model graph to Tensorboard.

        Parameters
        ----------
        model: torch.nn.Module or inherited it.
        input_size: tuple of ints
        device: str
            "cpu" or "cuda"

        Returns
        -------
        None
        """

        w, h = input_size
        inp = torch.rand((1, 1, w, h)).to(device)

        self.writer.add_graph(model, inp)

    def add_loss(self, loss, step):
        self.writer.add_scalar("Training loss", loss, step)

    def add_dice_score(self, scores, step):
        """Add dice scores of each class to Tensorboard.

        Parameters
        ----------
        scores: torch.Tensor
        step: int

        Returns
        -------
        None
        """

        for class_id, dice_score in enumerate(scores):
            self.writer.add_scalar(f"Class {class_id+1} training dice score", dice_score.item(), step)
