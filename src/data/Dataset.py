from abc import ABC, abstractmethod
import os

import pandas as pd
import torch
from torch.utils.data import Dataset
import torchio as tio

from utils.Utils import get_file_names


class _BaseClass(ABC):
    """Base class to create different dataset combinations.
    """

    @abstractmethod
    def get_train_indexes(self):
        """
        Returns train Data indexes.
        """
        pass

    @abstractmethod
    def get_val_indexes(self):
        """
        Returns validation Data indexes.
        """
        pass

    @abstractmethod
    def get_test_indexes(self):
        """
        Returns test Data indexes.
        """
        pass


class FeTA(_BaseClass):
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

    def __str__(self):
        return "Dataset_FeTA_5-fold_CV"

    def get_train_indexes(self):
        return self.train.index.to_list()

    def get_val_indexes(self):
        return self.val.index.to_list()

    def get_test_indexes(self):
        assert False, "This dataset implemented for cross-validation. There are only training and validation sets."


class FeTABalancedDistribution(_BaseClass):
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

    def __str__(self):
        return "Dataset_FeTABalancedDistribution"

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


class Dhcp(_BaseClass):
    """
    dHCP dataset contains 738 neonatal subjects. Scan age of MRIs belongs to these subjects range from 26 to 45 weeks.
    Only 89 of them whose MRIs were recorded under 35 weeks were selected for this experiment. first 70% of them (~69)
    were selected for training, 15% of them (~13) for validation and 15% of them (~13) for test.
    """

    def __init__(self, meta_data):
        self.meta_data = meta_data

    def __str__(self):
        return "Dataset_dHCP"

    def get_train_indexes(self):
        return sorted(self.meta_data[:63].index.to_list())

    def get_val_indexes(self):
        return sorted(self.meta_data[63:76].index.to_list())

    def get_test_indexes(self):
        return sorted(self.meta_data[76:].index.to_list())


class DhcpFeta(_BaseClass):
    """This class loads the combination of dHCP and FeTA datasets from the participants file which includes subjects
    of both dataset.
    """

    def __init__(self, meta_data):
        self.feta = FeTABalancedDistribution(meta_data[:80])
        self.dhcp = Dhcp(meta_data[80:])

    def __str__(self):
        return "Dataset_dHCP+FeTA"

    def get_train_indexes(self):
        return sorted([*self.feta.get_train_indexes(), *self.dhcp.get_train_indexes()])

    def get_val_indexes(self):
        return sorted([*self.feta.get_val_indexes(), *self.dhcp.get_val_indexes()])

    def get_test_indexes(self):
        return sorted([*self.feta.get_test_indexes(), *self.dhcp.get_test_indexes()])


class EarlyWeeks(_BaseClass):
    """This class provides the indexes of subjects gestational age < 24.9 weeks.
    """

    def __init__(self, meta_data):
        meta_data = meta_data[meta_data["Gestational age"] < 24.9]
        meta_data = meta_data.sort_values(by="participant_id")

        self.mial_srtk = meta_data[:10]
        self.simple_irtk = meta_data[10:]

    def __str__(self):
        return "Dataset_FeTA<24.9"

    def get_train_indexes(self):
        train = [*self.mial_srtk[:8].index.to_list(), *self.simple_irtk[:10].index.to_list()]

        return sorted(train)

    def get_val_indexes(self):
        val = [*self.mial_srtk[8:9].index.to_list(), *self.simple_irtk[10:12].index.to_list()]

        return sorted(val)

    def get_test_indexes(self):
        test = [*self.mial_srtk[9:].index.to_list(), *self.simple_irtk[12:].index.to_list()]

        return sorted(test)


class MiddleWeeks(_BaseClass):
    """This class provides the indexes of subjects 24.9 <= gestational age < 29.8 weeks .
    """

    def __init__(self, meta_data):
        meta_data = meta_data[(24.9 <= meta_data["Gestational age"]) & (meta_data["Gestational age"] < 29.8)]
        meta_data = meta_data.sort_values(by="participant_id")

        self.mial_srtk = meta_data[:20]
        self.simple_irtk = meta_data[20:]

    def __str__(self):
        return "Dataset_24.9<FeTA<29.8"

    def get_train_indexes(self):
        train = [*self.mial_srtk[:14].index.to_list(), *self.simple_irtk[:11].index.to_list()]

        return sorted(train)

    def get_val_indexes(self):
        val = [*self.mial_srtk[14:17].index.to_list(), *self.simple_irtk[11:14].index.to_list()]

        return sorted(val)

    def get_test_indexes(self):
        test = [*self.mial_srtk[17:].index.to_list(), *self.simple_irtk[14:].index.to_list()]

        return sorted(test)


class LateWeeks(_BaseClass):
    """This class provides the indexes of subjects 29.8 weeks <= gestational age .
    """

    def __init__(self, meta_data):
        meta_data = meta_data[29.8 <= meta_data["Gestational age"]]
        meta_data = meta_data.sort_values(by="participant_id")

        self.mial_srtk = meta_data[:8]
        self.simple_irtk = meta_data[8:]

    def __str__(self):
        return "Dataset_29.8<FeTA"

    def get_train_indexes(self):
        train = [*self.mial_srtk[:5].index.to_list(), *self.simple_irtk[:6].index.to_list()]

        return sorted(train)

    def get_val_indexes(self):
        val = [*self.mial_srtk[5:7].index.to_list(), *self.simple_irtk[6:7].index.to_list()]

        return sorted(val)

    def get_test_indexes(self):
        test = [*self.mial_srtk[7:].index.to_list(), *self.simple_irtk[7:].index.to_list()]

        return sorted(test)


class RandomSplit(_BaseClass):
    def __init__(self, meta_data):
        train, val, test =  torch.utils.data.random_split(meta_data, lengths=[60, 10, 10],
                                                 generator=torch.Generator().manual_seed(0))

        self.train_indexes = train.indices
        self.val_indexes = val.indices
        self.test_indexes = test.indices

    def __str__(self):
        return "Dataset_RandomSplit"

    def get_train_indexes(self):
        return self.train_indexes

    def get_val_indexes(self):
        return self.val_indexes

    def get_test_indexes(self):
        return self.test_indexes


class MRIDataset(Dataset):
    """Load MRI datasets.
    """

    def __init__(self, dataset=None, split=None, path="feta_2.1", cv=None, transform=None):
        """Creates train, validation or test sets from FeTA2.1 and/or dHCP dataset.

        Parameters
        ----------
        dataset: Classes of Dataset
            FeTA, Dhcp or others.
        split: str
            "train", "val" or "test"
        path: str
            Main folder path of the Data.
        cv: str
            Fold code of the cross-validation like CV1, CV2, CV3, CV4, and CV5 for 5-fold cross-validation.
            Warnings: Only implemented for FeTA dataset.
        transform: torch or torchio transforms
        """

        assert dataset is not None, "Choose a dataset"

        self.__path_base = path
        self.__transform = transform
        self.meta_data = pd.read_csv(os.path.join(self.__path_base, "participants.tsv"), sep="\t")
        self.__paths_file = get_file_names(self.__path_base)

        # dropped sub-ids
        low_quality_mris = ["sub-007", "sub-009"]
        if (dataset is EarlyWeeks) or (dataset is MiddleWeeks) or (dataset is LateWeeks):
            drop_indexes = self.meta_data[self.meta_data["participant_id"].isin(low_quality_mris)].index
            self.meta_data.drop(drop_indexes, inplace=True)
            self.meta_data = self.meta_data.sort_values(by="participant_id").reset_index(drop=True)

        # Cross Validation Dataset
        if cv:
            dataset_x = dataset(self.meta_data, cv)
        else:
            dataset_x = dataset(self.meta_data)

        if split == "train":
            train_indexes = dataset_x.get_train_indexes()
            self.meta_data = self.meta_data.iloc[train_indexes]
        elif split == "val":
            val_indexes = dataset_x.get_val_indexes()
            self.meta_data = self.meta_data.iloc[val_indexes]
        elif split == "test":
            test_indexes = dataset_x.get_test_indexes()
            self.meta_data = self.meta_data.iloc[test_indexes]

        self.meta_data = self.meta_data.sort_values(by="participant_id")
        self.meta_data = self.meta_data.reset_index(drop=True)

        self.dataset = tio.SubjectsDataset(self.__get_subjects(), transform=self.__transform)

        # Histogram equalization made training slow and didn't improve dice score.
        # self.HE = Pre('t2_feta_landmarks.pth')

    def __getitem__(self, index):
        subject = self.dataset[index]

        return subject

    def __len__(self):
        return self.meta_data.shape[0]

    def __get_subjects(self):
        subjects = []
        for index, row in self.meta_data.iterrows():
            path_mri, path_mask = self.__paths_file[row.participant_id]
            subject = tio.Subject(mri=tio.ScalarImage(path_mri),
                                  mask=tio.LabelMap(path_mask),
                                  sub_id=row.participant_id
                                  )

            subjects.append(subject)

        return subjects
