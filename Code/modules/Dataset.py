import os

import nibabel as nib
import pandas as pd
import torch
from torch.utils.data import Dataset

from modules.Utils import get_file_names


class FeTADataSet(Dataset):
    """Load FeTA2.1 dataset and splits it into train, validation or test sets.
    """

    def __init__(self, set_="train", path="feta_2.1", transform=None):
        """Creates train, validation or test sets from FeTA2.1 dataset.

        Parameters
        ----------
        set_: str
            "train", "val" or "test"
        path: str
            Main folder path of the data.
        transform: torch or torchio transforms
        """

        self.__path_base = path
        self.__transform = transform

        self.meta_data = pd.read_csv(os.path.join(self.__path_base, "participants.tsv"), sep="\t")
        self.__paths_file = get_file_names(self.__path_base)

        # Images below might have bad qualities
        # "sub-007" and "sub-009"
        split_data = _SplitDataset(self.meta_data)

        if set_ == "train":
            train_indexes = split_data.get_train_indexes()
            self.meta_data = self.meta_data.iloc[train_indexes]
        elif set_ == "val":
            val_indexes = split_data.get_val_indexes()
            self.meta_data = self.meta_data.iloc[val_indexes]
        else:
            test_indexes = split_data.get_test_indexes()
            self.meta_data = self.meta_data.iloc[test_indexes]

        self.meta_data = self.meta_data.reset_index().drop("index", axis=1)

    def __getitem__(self, index):
        if isinstance(index, int):
            sub_id = self.meta_data.participant_id[index]
            mri_image, mri_mask = self.__get_data(sub_id)
            (x, y, z) = mri_image.shape

            if self.__transform:
                mri_image = mri_image.view(1, x, y, z)
                mri_image = self.__transform(mri_image)
                mri_image = mri_image.view(x, y, z)

            return mri_image, mri_mask

        elif isinstance(index, slice):
            assert index.stop <= self.meta_data.shape[0], "Index out of range."

            sub_ids = self.meta_data.participant_id[index].tolist()
            mri_images = []
            mri_masks = []

            for sub_id in sub_ids:
                mri_image, mri_mask = self.__get_data(sub_id)

                if self.__transform:
                    mri_image = mri_image.view(1, *mri_image.shape)
                    mri_image = self.__transform(mri_image)
                    mri_image = mri_image.view(mri_image.shape[1:])

                mri_images.append(mri_image)
                mri_masks.append(mri_mask)

            return tuple(zip(mri_images, mri_masks))

    def __len__(self):
        return self.meta_data.shape[0]

    def __get_data(self, sub_id):
        data = self.__paths_file[sub_id]
        path_image, path_mask = data[0], data[1]

        mri_image = nib.load(path_image).get_fdata()
        mri_image = torch.Tensor(mri_image)

        mri_mask = nib.load(path_mask).get_fdata()
        mri_mask = torch.Tensor(mri_mask)

        return mri_image, mri_mask


class _SplitDataset:
    """
    There are 80 MRI images of 80 subjects. Gestational ages of subjects ranges 20 weeks to 35 weeks.
    There are Pathological and Neurotypical subjects.
    First 40 MRI images (sub-001 - sub-040) constructed by mialSRTK method.
    Other 40 MRI images (sub-041 - sub-080) constructed by simpleIRTK method.

    Data distribution.
    ------------------
    mialSRTK reconstruction:
        * Gestational age <=28
            - Neurotypical: 7 MRI images.   [train:5, val:1, test:1]
            - Pathological: 20 MRI images.  [train:16, val:2, test:2]
        * Gestational age > 28
            - Neurotypical: 8 MRI images.   [train:6, val:1, test:1]
            - Pathological: 5 MRI images.   [train:3, val:1, test:1]

    simpleIRTK reconstruction:
        * Gestational age <=28
            - Neurotypical: 7 MRI images.   [train:5, val:1, test:1]
            - Pathological: 17 MRI images.  [train:13, val:2, test:2]
        * Gestational age > 28
            - Neurotypical: 11 MRI images.  [train:9, val:1, test:1]
            - Pathological: 5 MRI images.   [train:3, val:1, test:1]

    Note: 28 was determined intuitively for diversity gestational weeks and smoother age distribution
    """

    def __init__(self, meta_data):
        mial_srtk = meta_data[:40]
        simple_irtk = meta_data[40:]

        self.index1 = mial_srtk[
            (mial_srtk["Gestational age"] <= 28) & (mial_srtk["Pathology"] == "Neurotypical")].index.to_list()
        self.index2 = mial_srtk[
            (mial_srtk["Gestational age"] <= 28) & (mial_srtk["Pathology"] == "Pathological")].index.to_list()

        self.index3 = mial_srtk[
            (mial_srtk["Gestational age"] > 28) & (mial_srtk["Pathology"] == "Neurotypical")].index.to_list()
        self.index4 = mial_srtk[
            (mial_srtk["Gestational age"] > 28) & (mial_srtk["Pathology"] == "Pathological")].index.to_list()

        self.index5 = simple_irtk[
            (simple_irtk["Gestational age"] <= 28) & (simple_irtk["Pathology"] == "Neurotypical")].index.to_list()
        self.index6 = simple_irtk[
            (simple_irtk["Gestational age"] <= 28) & (simple_irtk["Pathology"] == "Pathological")].index.to_list()

        self.index7 = simple_irtk[
            (simple_irtk["Gestational age"] > 28) & (simple_irtk["Pathology"] == "Neurotypical")].index.to_list()
        self.index8 = simple_irtk[
            (simple_irtk["Gestational age"] > 28) & (simple_irtk["Pathology"] == "Pathological")].index.to_list()

    def get_train_indexes(self):
        train = [self.index1[:5], self.index2[:16], self.index3[:6], self.index4[:3], self.index5[:5], self.index6[:13],
                 self.index7[:9], self.index8[:3]]

        train = [item for sub_arr in train for item in sub_arr]

        return sorted(train)

    def get_val_indexes(self):
        validation = [self.index1[5:6], self.index2[16:18], self.index3[6:7], self.index4[3:4], self.index5[5:6],
                      self.index6[13:15], self.index7[9:10], self.index8[3:4]]

        validation = [item for sub_arr in validation for item in sub_arr]

        return sorted(validation)

    def get_test_indexes(self):
        test = [self.index1[6:], self.index2[18:], self.index3[7:], self.index4[4:], self.index5[6:], self.index6[15:],
                self.index7[10:], self.index8[4:]]

        test = [item for sub_arr in test for item in sub_arr]

        return sorted(test)
