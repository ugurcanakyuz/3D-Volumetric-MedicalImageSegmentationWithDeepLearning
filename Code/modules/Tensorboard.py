import numpy as np
import torch

import torchvision
from torch.utils.tensorboard import SummaryWriter


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

