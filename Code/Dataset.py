import os

import nibabel as nib
import pandas as pd

from torch.utils.data import Dataset


class FeTADataSet(Dataset):
    # def __init__(self, quality=[], labels=[], pathologies=[], load_3d=None):
    def __init__(self, train=True, pathology="all"):
        """"""
        self.path_base = "feta_2.1"
        self.train = train

        self.meta_data = pd.read_csv(os.path.join(self.path_base, "participants.tsv"), sep="\t")
        count_train = 70  # First 70 MRI image consist of 40 Pathological and 20 Neurotypical.

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
            if self.train:
                self.meta_data = self.meta_data[:count_train]
            else:
                self.meta_data = self.meta_data[count_train:]
                self.meta_data = self.meta_data.reset_index().drop("index", axis=1)

        self.n_samples = self.meta_data.shape[0]

    def __getitem__(self, index):
        """"""
        reconstruction = ""

        if index < 40 and self.train == True:
            reconstruction = "_rec-mial"
        else:
            reconstruction = "_rec-irtk"

        path_image = os.path.join(self.path_base,
                                  self.meta_data.participant_id[index],
                                  "anat",
                                  self.meta_data.participant_id[index] + reconstruction + "_T2w.nii.gz")

        path_mask = os.path.join(self.path_base,
                                 self.meta_data.participant_id[index],
                                 "anat",
                                 self.meta_data.participant_id[index] + reconstruction + "_dseg.nii.gz")

        mri_image = nib.load(path_image).get_fdata()
        mri_mask = nib.load(path_mask).get_fdata()

        return mri_image, mri_mask

    def __len__(self):
        return self.n_samples
