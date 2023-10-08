import os

import nibabel as nib
import pandas as pd
import torch
from torch.utils.data import Dataset

from modules.Utils import get_file_names


class FeTADataSet(Dataset):
    # def __init__(self, quality=[], labels=[], pathologies=[], load_3d=None):
    def __init__(self, path="feta_2.1", train=True, transform=None, pathology="all"):
        """"""

        count_train = 70  # First 70 MRI image consist of 40 Pathological and 20 Neurotypical.
        self.__path_base = path
        self.__train = train
        self.__transform = transform

        self.meta_data = pd.read_csv(os.path.join(self.__path_base, "participants.tsv"), sep="\t")
        self.__paths_file = get_file_names(self.__path_base)

        # Images below might have bad qualities
        # self.meta_data = self.meta_data.drop(index=self.meta_data[
        # self.meta_data["participant_id"]=="sub-007"
        # ].index)
        # self.meta_data = self.meta_data.drop(index=self.meta_data[
        # self.meta_data["participant_id"]=="sub-009"
        # ].index)

        if pathology == "Pathological":
            self.meta_data = self.meta_data[self.meta_data.Pathology == "Pathological"]
        elif pathology == "Neurotypical":
            self.meta_data = self.meta_data[self.meta_data.Pathology == "Neurotypical"]
        else:
            # Return data for training or test.
            if self.__train:
                self.meta_data = self.meta_data[:count_train]
            else:
                self.meta_data = self.meta_data[count_train:]
                self.meta_data = self.meta_data.reset_index().drop("index", axis=1)

    def __get_data(self, sub_id):
        data = self.__paths_file[sub_id]
        path_image, path_mask = data[0], data[1]

        mri_image = nib.load(path_image).get_fdata()
        mri_image = torch.Tensor(mri_image)

        mri_mask = nib.load(path_mask).get_fdata()
        mri_mask = torch.Tensor(mri_mask)

        return mri_image, mri_mask

    def __getitem__(self, index):
        """"""
        if isinstance(index, int):
            sub_id = self.meta_data.participant_id[index]
            mri_image, mri_mask = self.__get_data(sub_id)

            if self.__transform:
                mri_image = mri_image.view(1, 256, 256, 256)
                mri_image = self.__transform(mri_image)
                mri_image = mri_image.view(256, 256, 256)

            return mri_image, mri_mask

        elif isinstance(index, slice):
            sub_ids = self.meta_data.participant_id[index].tolist()

            mri_images = torch.Tensor()
            mri_masks = torch.Tensor()

            for sub_id in sub_ids:
                mri_image, mri_mask = self.__get_data(sub_id)
                mri_image = mri_image.view(1, *mri_image.shape)
                mri_mask = mri_mask.view(1, *mri_mask.shape)

                mri_images = torch.cat([mri_image, mri_images], 0)
                mri_masks = torch.cat([mri_mask, mri_masks], 0)

            if self.__transform:
                mri_images = self.__transform(mri_images)

            return tuple(zip(mri_images, mri_masks))

    def __len__(self):
        return self.meta_data.shape[0]
