import glob
import os
import re

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch


def calculate_dice_score(pred, mask, smooth=1e-5):
    """This method calculates the dice score for each given class.

    Parameters
    ----------
    pred: torch.Tensor
        [bs, number_of_classes, x, y] or
        [bs, number_of_classes, x, y, z]
    mask: torch.Tensor
        [bs, number_of_classes, x, y] or
        [bs, number_of_classes, x, y, z]
    smooth: float

    Returns
    -------
    dice_scores: torch.Tensor
        [n_classes, dice_scores] Dice scores of the given classes.
    """

    with torch.no_grad():
        if len(pred.shape) == 4:
            (bs, n_classes, x, y) = pred.shape
            pred = pred.permute(1, 0, 2, 3).contiguous()
            pred = pred.view(n_classes, bs * x * y)

            mask = mask.permute(1, 0, 2, 3).contiguous()
            mask = mask.view(n_classes, bs * x * y)
        elif len(pred.shape) == 5:
            (bs, n_classes, x, y, z) = pred.shape
            pred = pred.permute(1, 0, 2, 3, 4).contiguous()
            pred = pred.view(n_classes, bs * x * y * z)

            mask = mask.permute(1, 0, 2, 3, 4).contiguous()
            mask = mask.view(n_classes, bs * x * y * z)

    intersection = (pred * mask).sum(-1)
    denominator = (pred + mask).sum(-1)

    dice_scores = ((2 * intersection).clamp(min=smooth)) / denominator.clamp(min=smooth)

    return dice_scores


def create_onehot_mask(pred_shape, mask):
    """Creates onehot mask for multidimensional mask.

    Parameters
    ----------
    pred_shape: tuple
        (bs,number_of_classes,x,y)
                    or
        (bs,number_of_classes,x,y,z)
    mask: torch.Tensor
        (bs,1,x,y)
            or
        (bs,1,x,y,z)

    Returns
    -------
    mask_onehot: torch.Tensor
        [bs,number_of_classes,x,y,z]
    """

    mask_onehot = torch.zeros(pred_shape, requires_grad=False).to(mask.device)
    mask = mask.long()
    mask_onehot.scatter_(1, mask, 1)

    return mask_onehot


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
    """List the files in subdirectories.

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
    for data1, data2 in pairwise(files):
        if re.findall("T2W*", data1):
            image = data1
            mask = data2
        else:
            image = data2
            mask = data1

        # After different dataset moved under the 'data' folder index was set to '2'
        paths[image.split(os.sep)[2]] = [image, mask]

    return paths


def init_weights_kaiming(m):
    """Initialize model weights with uniform distribution.
    Find here why kaiming is better for ReLU activations: https://towardsdatascience.com/954fb9b47c79

    Parameters
    ----------
    m: torch.Model

    Returns
    -------
    None
    """

    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0)


def plot_sub(image, mask, pred_mask=None, fig_size=(13, 13)):
    """ Plots image, mask and prediction.

    Parameters
    ----------
    fig_size: tuple of ints
    image: torch.Tensor
    mask: torch.Tensor
    pred_mask: torch.Tensor

    Returns
    -------
    None
    """

    fig = plt.figure(figsize=fig_size)
    fig.add_subplot(1, 3, 1)
    plt.imshow(image)
    fig.add_subplot(1, 3, 2)
    plt.imshow(mask)

    try:
        if torch.any(pred_mask):
            fig.add_subplot(1, 3, 3)
            plt.imshow(pred_mask)
    except TypeError:
        if np.any(pred_mask):
            fig.add_subplot(1, 3, 3)
            plt.imshow(pred_mask)

    plt.show()


def save_nii(folder, file_name, data):
    """Saves mri or mask data as nii.gz file. First converts ndarray to Nifti1Image, then save it as nii.gz. file.

    Parameters
    ----------
    folder: str
        What folder data will be saved under.
    file_name: str
        File name without '.nii.gz' extension.
    data: ndarray
        3 dimension (x, y, z,) MRI or mask data.

    Returns
    -------
    None.
    """

    file_name += ".nii.gz"
    full_path = os.path.join(folder, file_name)

    # First convert ndarray nibabel format.
    try:
        data = nib.Nifti1Image(data, np.eye(4))
    except Exception as e:
        print(str(e))

    # Save data in format of Nifti1Image.
    try:
        nib.save(data, full_path)
    except Exception as e:
        print(str(e))


class EarlyStopping:
    # Implemented from https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/
    # and modified.

    def __init__(self, patience=5, min_delta=0.025):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            # print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True


class LearningRateFinder:
    """ Learning rate finder changes the learning rate of the model within given lr range
    and finds losses for each training with corresponding lr.
    """

    def __init__(self, trainer):
        self.trainer = trainer

        self.lrs = []
        self.losses = []

    def find(self, min_lr=10e-10, max_lr=1e+1, lr_factor=1e+1):
        """Searches best learning rates between given learning rates and factor.
        """

        curr_lr = min_lr

        while curr_lr <= max_lr:
            # One forward pass for all training data.
            avg_train_loss = self.trainer.fit()

            self.lrs.append(curr_lr)
            self.losses.append(avg_train_loss)
            curr_lr *= lr_factor
            self.trainer.optimizer.param_groups[0]['lr'] = curr_lr

        self.__plot_loss()

    def __plot_loss(self):
        plt.plot(self.lrs, self.losses)
        plt.ylim(min(self.losses), 3)  # These limits where chosen due to DS+BCE Loss range to visualize curve better.
        plt.xscale("log")
        plt.xlabel("Learning Rate (Log Scale)")
        plt.ylabel("Loss")
        plt.title("Learning Rates")
        plt.show()
