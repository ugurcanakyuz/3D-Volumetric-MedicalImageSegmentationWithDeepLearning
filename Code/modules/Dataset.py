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


    def __getitem__(self, index):
        """"""

        data = self.__paths_file[self.meta_data.participant_id[index]]
        path_image, path_mask = data[0], data[1]

        mri_image = nib.load(path_image).get_fdata()
        mri_mask = nib.load(path_mask).get_fdata()

        if self.__transform:
            mri_image = torch.tensor(mri_image)
            mri_image = mri_image.view(1, 256, 256, 256)
            mri_image = self.__transform(mri_image)
            mri_image = mri_image.view(256, 256, 256)

        return mri_image, mri_mask

    def __len__(self):
        return self.meta_data.shape[0]
