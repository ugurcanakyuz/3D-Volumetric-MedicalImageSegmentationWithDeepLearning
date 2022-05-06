import random

import numpy as np
import torch
import torchio as tio


def apply_transform(mri, mask, transform):

    seed = np.random.randint(2147483647)

    random.seed(seed)
    torch.manual_seed(seed)
    mri = transform(mri)

    random.seed(seed)
    torch.manual_seed(seed)
    mask = transform(mask)

    return mri, mask


class RandomAffine:
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
    def __init__(self, degrees=0, num_transforms=2, translations=0.5):
        assert False, "Needs to be fixed. Don't use it."

        self.degrees = degrees
        self.translation = translations
        self.num_transforms = num_transforms

    def __call__(self, sample):
        mri, mask = sample
        random_motion = tio.RandomMotion(num_transforms=self.num_transforms)
        mri = random_motion(mri)

        return mri, mask
