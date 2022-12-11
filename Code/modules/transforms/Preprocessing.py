import cv2
import numpy as np
import torch
import torchio as tio


def z_normalization(mri):
    """Apply ZNormalization to 3D tensors.

    Parameters
    ----------
    mri: torch.Tensor
       Shape: [x,y,z]
    Returns
    ---------
    mri: torch.Tensor
       Shape: [x,y,z]
    """

    z_normalize = tio.ZNormalization(masking_method=tio.ZNormalization.mean)
    mri = mri.unsqueeze(0)
    mri = z_normalize(mri)
    mri = mri.squeeze(0)

    return mri


def apply_clahe(mri):
    """Apply Contrast Limited Adaptive Histogram Equalization to 3D array.

    Parameters
    ----------
    mri: numpy.Array
       Shape: [x,y,z]

    Returns
    ---------
    mri: numpy.Array
       Shape: [x,y,z]
    """

    new_mri = []
    clahe = cv2.createCLAHE(clipLimit=4)

    for i in range(mri.shape[0]):
        coronal_slice = mri[i].astype(np.uint16)
        mri_slice = clahe.apply(coronal_slice) + 30
        new_mri.append(mri_slice)

    return np.array(new_mri).astype(dtype='<f8')



class HistogramStandardization:
    def __init__(self, path_landmark):
        self.preprocessing = tio.HistogramStandardization(landmarks=path_landmark,
                                                          masking_method=tio.ZNormalization.mean)

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
