import random

import numpy as np
import torch
import torchio as tio


class Preprocessing:
    def __init__(self, path_landmark):
        self.preprocessing = tio.HistogramStandardization(landmarks=path_landmark,
                                                          masking_method=tio.ZNormalization.mean)

    @staticmethod
    def apply_normalization(mri):
        preprocessing = tio.ZNormalization(masking_method=tio.ZNormalization.mean)
        mri = mri.view(1, *mri.shape)
        mri = preprocessing(mri)
        mri = mri.view(mri.shape[1:])

        return mri

    def apply_histogram_equalization(self, mri):
        """See: https://torchio.readthedocs.io/transforms/preprocessing.html#histogramstandardization

        Parameters
        ----------
        mri: torch.Tensor
            (x, y, z)

        Returns
        -------
        mri: torch.Tensor
            (x, y, z)
        """

        subject = tio.Subject(t2=tio.ScalarImage(tensor=mri.unsqueeze(0)))
        subject = self.preprocessing(subject)
        mri = subject['t2'].data
        mri = mri.squeeze(0)

        return mri


class Mask:
    """Sets voxel outside of brain region the background value. Similar to torchio.Mask.
       See details: https://torchio.readthedocs.io/transforms/preprocessing.html#mask
       Some Data are not correctly labelled, this problem can be solved with masking operation.

    """
    def __call__(self, sample):
        mri, mask = sample

        # Get only brain region.
        brain = mask.clone()
        brain[brain > 1] = 1

        mri[brain == 0] = mri.min()

        return mri, mask


class RandomAffine:
    """Apply a random affine transformation and resample mri and mask.
    See details: https://torchio.readthedocs.io/transforms/augmentation.html#randomaffine
    FeTA and dHCP datasets have already transformed to isotropic Data with 0.5x0.5x0.5mm resolutions.
    Therefore, 'isotropic' and 'translation' parameters was set manually.
    """

    def __init__(self, scales=(0.95, 0.95), degrees=0):
        self.degrees = degrees
        self.isotropic = True  # Datasets are isotropic.
        self.scales = scales
        self.translation = 0.5  # Datasets translations range in mm.

    def __call__(self, sample):
        mri, mask = sample

        random_affine = tio.RandomAffine(scales=self.scales, degrees=self.degrees,
                                         translation=self.translation, isotropic=self.isotropic)

        mri, mask = apply_transform(mri, mask, random_affine)

        return mri, mask


class RandomElasticDeformation:
    """Apply random elastic deformation to mri and mask.
    See details: https://torchio.readthedocs.io/transforms/augmentation.html#randomelasticdeformation
    This transformation has high computational cost and can extend the training time.
    """

    def __init__(self, num_control_points=7, locked_borders=2):
        self.num_control_points = num_control_points
        self.locked_borders = locked_borders

    def __call__(self, sample):
        mri, mask = sample
        random_ed = tio.RandomElasticDeformation(num_control_points=self.num_control_points,
                                                 locked_borders=self.locked_borders)

        mri, mask = apply_transform(mri, mask, random_ed)

        return mri, mask


class RandomMotion:
    """Simulate random motion artifacts for Data augmentation to mri and mask.
    See details: https://torchio.readthedocs.io/transforms/augmentation.html#randommotion
    """

    def __init__(self, degrees=0, num_transforms=2, translations=0.5):
        self.degrees = degrees
        self.translation = translations
        self.num_transforms = num_transforms

    def __call__(self, sample):
        mri, mask = sample
        random_motion = tio.RandomMotion(num_transforms=self.num_transforms)
        mri = random_motion(mri)

        return mri, mask


def apply_transform(mri, mask, transform):
    """Apply same structural transform to mri and mask.

    Parameters
    ----------
    mri: torch.Tensor
    mask: torch.Tensor
    transform: torchio.Transform

    Returns
    -------
    mri: torch.Tensor
    mask: torch.Tensor
    """

    seed = np.random.randint(2147483647)

    random.seed(seed)
    torch.manual_seed(seed)
    mri = transform(mri)

    random.seed(seed)
    torch.manual_seed(seed)
    mask = transform(mask)

    return mri, mask
