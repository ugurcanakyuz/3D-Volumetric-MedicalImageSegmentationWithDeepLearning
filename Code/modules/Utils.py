import glob
import os

import numpy as np
import matplotlib.pyplot as plt
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
    for image, mask in pairwise(files):
        paths[image.split(os.sep)[1]] = [image, mask]

    return paths


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


def plot_sub(image, mask, pred_mask=None):
    """ Plots image, mask and prediction.

    Parameters
    ----------
    image: torch.Tensor
    mask: torch.Tensor
    pred_mask: torch.Tensor

    Returns
    -------
    None
    """

    fig = plt.figure(figsize=(13, 13))
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
        plt.ylim(min(self.losses), 3)  # These limits where choosen due to DS+BCE Loss range to visualize curve better.
        plt.xscale("log")
        plt.xlabel("Learning Rate (Log Scale)")
        plt.ylabel("Loss")
        plt.title("Learning Rates")
        plt.show()


class TensorboardModules:
    """This class consists of methods that allow adding data into Tensorboard.
    """

    def __init__(self, log_dir="events"):
        """Initialize summary writer.

        Parameters
        ----------
        log_dir: str
            Storage path of the event files.
        """

        self.step = 0
        self.writer = SummaryWriter(log_dir)

    def close(self):
        """Close summary writer."""

        self.writer.close()

    def add_images(self, tag, image, slices):
        """Adds images into tensorboard.

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
        n_images, x, y = grid_data.shape

        img_grid = torchvision.utils.make_grid(grid_data.view(n_images, 1, x, y))
        self.writer.add_image(tag, img_grid)

    def add_image_mask(self, mri_image, mri_mask, slices):
        """Adds 2D slices of 3D image and mask into the tensorboard.

        Parameters
        ----------
        mri_image: torch.Tensor or ndarray
        mri_mask: torch.Tensor or ndarray
        slices: tuple of int
            (start, end, step)

        Returns
        -------
        None.
        """
        class_index = max(np.unique(mri_mask))

        for i in range(*slices):
            mri_mask[:, :, i][mri_mask[:, :, i] == class_index] = 255
            class_index -= 1

            if class_index <= 0:
                class_index = max(np.unique(mri_mask))

        self.add_images("Fetal Brain Images", mri_image, slices)
        self.add_images("Fetal Brain Masks", mri_mask, slices)

    def add_graph(self, model, input_size, device):
        """Creates an input to model and adds model graph into Tensorboard.

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

        inp = torch.rand((1, 1, *input_size)).to(device)

        self.writer.add_graph(model, inp)

    def add_lr(self, lr):
        self.writer.add_scalar("Learning rate", lr, self.step)

    def add_dice_score(self, scores):
        """Add dice scores of each class into Tensorboard.

        Parameters
        ----------
        scores: list of torch.Tensor

        Returns
        -------
        None
        """

        cb_labels = ["Background and Non-Brain", "Extra-axial CSF", "Gray Matter and developing cortical plate",
                     "White matter and subplate", "Lateral ventricles",
                     "Cerebellum", "Thalamus and putamen", "Brainstem"]

        for class_id, dice_score in enumerate(scores):
            self.writer.add_scalar(f"Dice Score of class: {class_id}-{cb_labels[class_id]}",
                                   dice_score.item(), self.step)

    def add_train_loss(self, loss):
        self.writer.add_scalar("Training loss", loss, self.step)

    def add_val_loss(self, loss):
        self.writer.add_scalar("Validation loss", loss, self.step)

    def add_scalars(self, step, ds=0, lr=0, train_loss=0, val_loss=0):
        """ Add scalars into tensorboard.

        Parameters
        ----------
        step: int
        ds: list of floats
            Dice scores.
        lr: float
            Learning rate.
        train_loss: float
        val_loss: float

        Returns
        -------
        None.
        """

        self.step = step
        self.add_dice_score(ds)
        self.add_lr(lr)
        self.add_train_loss(train_loss)
        self.add_val_loss(val_loss)
