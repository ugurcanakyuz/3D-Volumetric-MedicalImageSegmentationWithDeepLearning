import numpy as np
import torch
import torchio as tio

from .Utils import plot_sub


class Predictor:
    """To evaluate results visually this class implemented. It can be used to get prediction of the models.
    It also provides method to draw slices.

    Example:
        predictor = Evaluator3D(model, model, val_loader)
        output = predictor.predict()

    """
    def __init__(self, model, patch_size):
        self.model = model
        self.device = next(model.parameters()).device
        self.patch_size = patch_size

    def predict(self, subject):
        """This methods gets an input image and gives it to model and return predicted mask. Masks are logits.

        Parameters
        ----------
        subject: torchio.Subject
            Subject has mri and mask attributes.

        Returns
        -------
        output: torch.Tensor
            [1, x, y, z] Logits, not the actual class labels of Softmax function.
            Use argmax function to get class labels.
        """

        overlap_mode_ = 'average'

        self.model.eval()
        with torch.no_grad():
            sampler = tio.data.GridSampler(subject=subject, patch_size=self.patch_size)
            aggregator = tio.data.GridAggregator(sampler, overlap_mode=overlap_mode_)

            for j, patch in enumerate(sampler(subject)):
                patch_image = patch.mri.data.unsqueeze(1).float().to(self.device)  # [bs,1,x,y,z]
                output = self.model(patch_image)
                aggregator.add_batch(output, patch["location"].unsqueeze(0))

            output = aggregator.get_output_tensor().unsqueeze(0)

        return output

    @staticmethod
    def __rotate_slice(mri_slice):
        """
        Returns
        -------
        mri_slice: torch.Tensor
        """

        return np.rot90(mri_slice)

    def plot_slice(self, output, mask, orientation='axial', slice_index=None):
        """This method plot random slices of the prediction output and the mask.

        output: torch.Tensor
            (1, x, y, z)
        mask: torch.Tensor
            (1, x, y, z)
        orientation: str
            Default 'axial'. 'coronal' or 'sagittal'
        """

        output = output.argmax(dim=1)

        if orientation == 'axial':
            if not slice_index:
                slice_index = torch.randint(low=10, high=output.shape[-1] - 10, size=(1,)).item()
            plot_sub(self.__rotate_slice(output[0, :, :, slice_index]),
                     self.__rotate_slice(mask[0, :, :, slice_index]))
        elif orientation == 'coronal':
            if not slice_index:
                slice_index = torch.randint(low=10, high=output.shape[-2] - 10, size=(1,)).item()
            plot_sub(self.__rotate_slice(output[0, :, slice_index, :]),
                     self.__rotate_slice(mask[0, :, slice_index, :]))
        elif orientation == 'sagittal':
            if not slice_index:
                slice_index = torch.randint(low=10, high=output.shape[-3] - 10, size=(1,)).item()
            plot_sub(self.__rotate_slice(output[0, slice_index, :, :]),
                     self.__rotate_slice(mask[0, slice_index, :, :]))
        else:
            assert False, 'Wrong orientation!'
