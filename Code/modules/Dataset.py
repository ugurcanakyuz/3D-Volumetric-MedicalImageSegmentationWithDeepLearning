from abc import ABC, abstractmethod
import os
from enum import Enum

import nibabel as nib
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchio as tio

from modules.Utils import get_file_names


class _BaseClass(ABC):
    """Base class to create different dataset combinations.
    """

    @abstractmethod
    def get_train_indexes(self):
        """
        Returns train data indexes.
        """
        pass

    @abstractmethod
    def get_val_indexes(self):
        """
        Returns validation data indexes.
        """
        pass

    @abstractmethod
    def get_test_indexes(self):
        """
        Returns test data indexes.
        """
        pass


class _FeTA(_BaseClass):
    def __init__(self, meta_data, cv):
        """5-Fold Cross Validation applied.
        | - 16 - | - 16 - | - 16 - | - 16 - | - 16 -|

        Parameters
        ----------
        meta_data: pandas.DataFramer
        cv: str

        Returns
        -------
        None.
        """

        self.meta_data = meta_data

        if cv == "cv1":
            self.pt1 = 64
            self.pt2 = 80
        elif cv == "cv2":
            self.pt1 = 48
            self.pt2 = 64
        elif cv == "cv3":
            self.pt1 = 32
            self.pt2 = 48
        elif cv == "cv4":
            self.pt1 = 16
            self.pt2 = 32
        else:
            self.pt1 = 0
            self.pt2 = 16

        self.train = pd.concat([self.meta_data[:self.pt1], self.meta_data[self.pt2:]])
        self.val = self.meta_data[self.pt1:self.pt2]

    def get_train_indexes(self):
        return self.train.index.to_list()

    def get_val_indexes(self):
        return self.val.index.to_list()

    def get_test_indexes(self):
        assert False, "This dataset implemented for cross-validation. There are only training and validation sets."


class _FeTABalancedDistribution(_BaseClass):
    """
    There are 80 MRI images of 80 subjects in the FeTA2021. Gestational ages of subjects ranges 20 weeks to 35 weeks.
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

    Note: 28 was determined intuitively for diversity gestational weeks and smoother age distribution.
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

        train = sorted(train)

        # Images below might have bad qualities
        # "sub-007" and "sub-009"
        train.pop(6)  # sub-007 has bad resolution
        train.pop(7)  # sub-009 has bad resolution

        return train

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


class _Dhcp(_BaseClass):
    """
    dHCP dataset contains 738 neonatal subjects. Scan age of MRIs belongs to these subjects range from 26 to 45 weeks.
    Only 89 of them whose MRIs were recorded under 35 weeks were selected for this experiment. first 70% of them (~69)
    were selected for training, 15% of them (~13) for validation and 15% of them (~13) for test.
    """

    def __init__(self, meta_data):
        self.meta_data = meta_data

    def get_train_indexes(self):
        return sorted(self.meta_data[:63].index.to_list())

    def get_val_indexes(self):
        return sorted(self.meta_data[63:76].index.to_list())

    def get_test_indexes(self):
        return sorted(self.meta_data[76:].index.to_list())


class _DhcpFeta(_BaseClass):
    """This class loads the combination of dHCP and FeTA datasets from the participants file which includes subjects
    of both dataset.
    """

    def __init__(self, meta_data):
        self.feta = _FeTABalancedDistribution(meta_data[:80])
        self.dhcp = _Dhcp(meta_data[80:])

    def get_train_indexes(self):
        return sorted([*self.feta.get_train_indexes(), *self.dhcp.get_train_indexes()])

    def get_val_indexes(self):
        return sorted([*self.feta.get_val_indexes(), *self.dhcp.get_val_indexes()])

    def get_test_indexes(self):
        return sorted([*self.feta.get_test_indexes(), *self.dhcp.get_test_indexes()])


class _EarlyWeeks(_BaseClass):
    """This class provides the indexes of subjects gestational age < 24.9 weeks.
    """

    def __init__(self, meta_data):
        meta_data = meta_data[meta_data["Gestational age"] < 24.9]
        meta_data = meta_data.sort_values(by="participant_id")

        self.mial_srtk = meta_data[:10]
        self.simple_irtk = meta_data[10:]

    def get_train_indexes(self):
        train = [*self.mial_srtk[:8].index.to_list(), *self.simple_irtk[:10].index.to_list()]

        return sorted(train)

    def get_val_indexes(self):
        val = [*self.mial_srtk[8:9].index.to_list(), *self.simple_irtk[10:12].index.to_list()]

        return sorted(val)

    def get_test_indexes(self):
        test = [*self.mial_srtk[9:].index.to_list(), *self.simple_irtk[12:].index.to_list()]

        return sorted(test)


class _MiddleWeeks(_BaseClass):
    """This class provides the indexes of subjects 24.9 <= gestational age < 29.8 weeks .
    """

    def __init__(self, meta_data):
        meta_data = meta_data[(24.9 <= meta_data["Gestational age"]) & (meta_data["Gestational age"] < 29.8)]
        meta_data = meta_data.sort_values(by="participant_id")

        self.mial_srtk = meta_data[:20]
        self.simple_irtk = meta_data[20:]

    def get_train_indexes(self):
        train = [*self.mial_srtk[:14].index.to_list(), *self.simple_irtk[:11].index.to_list()]

        return sorted(train)

    def get_val_indexes(self):
        val = [*self.mial_srtk[14:17].index.to_list(), *self.simple_irtk[11:14].index.to_list()]

        return sorted(val)

    def get_test_indexes(self):
        test = [*self.mial_srtk[17:].index.to_list(), *self.simple_irtk[14:].index.to_list()]

        return sorted(test)


class _LateWeeks(_BaseClass):
    """This class provides the indexes of subjects 29.8 weeks <= gestational age .
    """

    def __init__(self, meta_data):
        meta_data = meta_data[29.8 <= meta_data["Gestational age"]]
        meta_data = meta_data.sort_values(by="participant_id")

        self.mial_srtk = meta_data[:8]
        self.simple_irtk = meta_data[8:]

    def get_train_indexes(self):
        train = [*self.mial_srtk[:5].index.to_list(), *self.simple_irtk[:6].index.to_list()]

        return sorted(train)

    def get_val_indexes(self):
        val = [*self.mial_srtk[5:7].index.to_list(), *self.simple_irtk[6:7].index.to_list()]

        return sorted(val)

    def get_test_indexes(self):
        test = [*self.mial_srtk[7:].index.to_list(), *self.simple_irtk[7:].index.to_list()]

        return sorted(test)


class MRIDatasets(Enum):
    FeTA = "FeTA"
    dHCP = "dHCP"
    dHCP_FeTA = "dF"

    FeTA_BalancedDistribution = "FBD"
    FeTA_EarlyWeeks = "FEW"
    FeTA_MiddleWeeks = "FMW"
    FeTA_LateWeeks = "FLW"


class MRIDataset(Dataset):
    """Load MRI datasets.
    """

    def __init__(self, dataset=None, split="train", path="feta_2.1", cv="cv1",  transform=None):
        """Creates train, validation or test sets from FeTA2.1 dataset.

        Parameters
        ----------
        dataset: Element of MRIDatasets
            Dataset class, FeTA, dHCP or others.
        split: str
            "train", "val" or "test"
        path: str
            Main folder path of the data.
        cv: str
            Fold code of the cross-validation like CV1, CV2, CV3, CV4, and CV5 for 5-fold cross-validation.
            Warnings: Only implemented for FeTA dataset.
        transform: torch or torchio transforms
        """

        assert dataset is not None, "Pass the dataset code as a MRIDataset value."

        dataset_x = None
        self.__path_base = path
        self.__transform = transform
        self.meta_data = pd.read_csv(os.path.join(self.__path_base, "participants.tsv"), sep="\t")
        self.__paths_file = get_file_names(self.__path_base)

        if dataset is (MRIDatasets.FeTA_EarlyWeeks or MRIDatasets.FeTA_MiddleWeeks or MRIDatasets.FeTA_LateWeeks):
            self.meta_data.drop(self.meta_data[self.meta_data["participant_id"] == "sub-007"].index, inplace=True)
            self.meta_data.drop(self.meta_data[self.meta_data["participant_id"] == "sub-009"].index, inplace=True)
            self.meta_data = self.meta_data.sort_values(by="participant_id").reset_index(drop=True)

        if dataset is MRIDatasets.FeTA:
            dataset_x = _FeTA(self.meta_data, cv)
        elif dataset is MRIDatasets.dHCP:
            dataset_x = _Dhcp(self.meta_data)
        elif dataset is MRIDatasets.dHCP_FeTA:
            dataset_x = _DhcpFeta(self.meta_data)
        elif dataset is MRIDatasets.FeTA_BalancedDistribution:
            dataset_x = _FeTABalancedDistribution(self.meta_data)
        elif dataset is MRIDatasets.FeTA_EarlyWeeks:
            dataset_x = _EarlyWeeks(self.meta_data)
        elif dataset is MRIDatasets.FeTA_MiddleWeeks:
            dataset_x = _MiddleWeeks(self.meta_data)
        elif dataset is MRIDatasets.FeTA_LateWeeks:
            dataset_x = _LateWeeks(self.meta_data)

        if split == "train":
            train_indexes = dataset_x.get_train_indexes()
            self.meta_data = self.meta_data.iloc[train_indexes]
        elif split == "val":
            val_indexes = dataset_x.get_val_indexes()
            self.meta_data = self.meta_data.iloc[val_indexes]
        else:
            test_indexes = dataset_x.get_test_indexes()
            self.meta_data = self.meta_data.iloc[test_indexes]

        self.meta_data = self.meta_data.sort_values(by="participant_id")
        self.meta_data = self.meta_data.reset_index().drop("index", axis=1)

        self.preprocessing = tio.HistogramStandardization('t2_feta_landmarks.pth',
                                                          masking_method=tio.ZNormalization.mean)

    def __getitem__(self, index):
        if isinstance(index, int):
            sub_id = self.meta_data.participant_id[index]
            mri_image, mri_mask = self.__get_data(sub_id)

            if self.__transform:
                mri_image, mri_mask = self.__apply_transform(mri_image, mri_mask)

            # Apply ZNormalization(see:https://torchio.readthedocs.io/transforms/preprocessing.html#znormalization).
            # mri_image = self.__apply_normalization(mri_image)
            mri_image = self.__apply_histogram_equalization(mri_image)

            return mri_image, mri_mask

        elif isinstance(index, slice):
            assert index.stop <= self.meta_data.shape[0], "Index out of range."

            sub_ids = self.meta_data.participant_id[index].tolist()
            mri_images = []
            mri_masks = []

            for sub_id in sub_ids:
                mri_image, mri_mask = self.__get_data(sub_id)

                if self.__transform:
                    mri_image, mri_mask = self.__apply_transform(mri_image, mri_mask)
                # Apply ZNormalization(see:https://torchio.readthedocs.io/transforms/preprocessing.html#znormalization).
                #mri_image = self.__apply_normalization(mri_image)
                mri_image = self.__apply_histogram_equalization(mri_image)

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

    @staticmethod
    def __apply_normalization(mri):
        preprocessing = tio.ZNormalization(masking_method=tio.ZNormalization.mean)
        mri = mri.view(1, *mri.shape)
        mri = preprocessing(mri)
        mri = mri.view(mri.shape[1:])

        return mri

    def __apply_histogram_equalization(self, mri):
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
        mri = subject.t2.data
        mri = mri.squeeze(0)

        return mri

    def __apply_transform(self, mri, mask):
        mri = mri.view(1, *mri.shape)
        mask = mask.view(1, *mask.shape)

        mri, mask = self.__transform((mri, mask))

        mri = mri.view(mri.shape[1:])
        mask = mask.view(mask.shape[1:])

        return mri, mask
