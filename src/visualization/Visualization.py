import os

import cv2
import matplotlib.colorbar as colorbar
import matplotlib.colors as clr
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

from utils.Utils import get_file_names

class COLORS:
    def __init__(self):
        """RGB color codes for RGB masks.
        """

        self.colors = {0: (0, 0, 0),  # Black for background.
                       1: (128, 128, 0),  # Olive for CSF.
                       2: (211, 211, 211),  # Light gray for gray matter.
                       3: (255, 255, 255),  # White for white matter.
                       4: (0, 0, 128),  # Navy for ventricles.
                       5: (255, 69, 0),  # Orange red for cerebellum.
                       6: (105, 105, 105),  # Dim gray for deep gray matter.
                       7: (178, 34, 34)  # Firebrick for brainstem and spinal cord.
                       }


class Visualization:
    def __init__(self, folder="feta_2.1/"):
        Visualization.folder = folder

    @staticmethod
    def __rotate_image(image, angle):
        """Rotate 2D image for given angle.
        WARNING! This can cause image degradation.

        Parameters
        ----------
        image: 2D numpy array

        Returns
        -------
        2D numpy array
        """

        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

        return result

    @staticmethod
    def __convert_to_3D(mask):
        """Adds one more dimeson to 2D mask image in order to obtain RGB mask.

        Parameters
        ----------
        mask: 2D numpy array

        Returns
        -------
        mask_rgb: 2D numpy array
        """

        zeros = np.zeros((*mask.shape[:2], 2))
        mask_rgb = np.stack((zeros[:, :, 0], zeros[:, :, 1], mask), axis=2).astype(int)

        return mask_rgb

    @staticmethod
    def __change_colors(mask, colors):
        """Changes the color of label in the mask.

        Parameters
        ----------
        mask: 2D numpy array
        colors: dict
            label number: RGB code

        Returns
        -------
        2D numpy array
        """

        mask = mask.astype(int)
        labels = set(mask[:, :, 2].flatten())

        for label in labels:
            mask[mask[:, :, 2] == label] = colors[label]

        return mask

    @staticmethod
    def __create_RGB_mask(mask):
        """Creates RGB mask from 2D mask.

        Parameters
        ----------
        mask: 2D numpy array

        Returns
        -------
        mask_rgb: 2D numpy array
        """

        mask_3d = Visualization.__convert_to_3D(mask)
        colors = COLORS()
        mask_rgb = Visualization.__change_colors(mask_3d, colors.colors)

        return mask_rgb

    @staticmethod
    def __draw_colorbar(fig):
        """Draws a custom colorbar to describe the brain region labels.

        Parameters
        ----------
        fig: matplotlib.figure.Figure

        Returns
        -------
        None
        """

        ax = fig.add_axes([0.92, 0.328, 0.02, 0.349])

        cb_colors = ["#000000", "#808000", "#D3D3D3", "#FFFFFF", "#000080", "#FF4500", "#696969", "#B22222"]
        cb_labels = ["background", "CSF", "gray matter", "white matter",
                     "ventricles", "cerebellum", "deep gray matter", "brain stem&spinal cord"]
        cmap_ = clr.ListedColormap(cb_colors)

        cb = colorbar.ColorbarBase(ax, orientation='vertical',
                                   cmap=cmap_, norm=plt.Normalize(-0.5, len(cb_colors) - 0.5))

        cb.set_ticks(range(len(cb_colors)))
        cb.ax.set_yticklabels(cb_labels)

    @staticmethod
    def draw_layout(index, sub_name, orientation):
        """Draws image, mask and colorbar.

        Parameters
        ----------
        index: int
            Index of the 2D image in 3D image.
        sub_name: str
            Name (number) of the subject like sub-040.

        orientation: str
            axial|coronal|sagittal

        Returns
        -------
        None
        """

        path = get_file_names(os.path.join(Visualization.folder, sub_name))
        path_image = path[sub_name][0]
        path_mask = path[sub_name][1]

        image = nib.load(path_image).get_fdata()
        mask = nib.load(path_mask).get_fdata()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))
        Visualization.__draw_colorbar(fig)
        ax1.set_title('Image')
        ax2.set_title('Mask')

        (x, y, z) = image.shape

        if orientation == "axial":
            if index < z:
                slice_ = image[:, :, index]
                slice_ = np.rot90(slice_)

                mask = mask[:, :, index]
                mask = np.rot90(mask)
                mask = Visualization.__create_RGB_mask(mask)

                ax1.imshow(slice_, cmap='gray')
                ax2.imshow(mask)

        elif orientation == "coronal":
            if index < y:
                slice_ = image[:, index, :]
                slice_ = np.rot90(slice_)
                mask = mask[:, index, :]
                mask = np.rot90(mask)
                mask = Visualization.__create_RGB_mask(mask)

                ax1.imshow(slice_, cmap='gray')
                ax2.imshow(mask)

        else:
            if index < x:
                slice_ = image[index, :, :]
                slice_ = np.rot90(slice_)

                mask = mask[index, :, :]
                mask = np.rot90(mask)
                mask = Visualization.__create_RGB_mask(mask)

                ax1.imshow(slice_, cmap='gray')
                ax2.imshow(mask)

        plt.show()
