import torch.utils.data as data


class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_by_name(dataset_name, opt, is_for_train):
        if dataset_name == 'RealContinuous':
            from data.RealData import Dataset
            dataset = Dataset(opt, is_for_train=False)
        else:
            raise ValueError("Dataset [%s] not recognized." % dataset_name)
        print('Dataset {} was created'.format(dataset.name))
        return dataset


class DatasetBase(data.Dataset):
    def __init__(self, opt, is_for_train):
        super(DatasetBase, self).__init__()
        self._name = 'BaseDataset'
        self._opt = opt
        self._is_for_train = is_for_train

    @property
    def name(self):
        return self._name
