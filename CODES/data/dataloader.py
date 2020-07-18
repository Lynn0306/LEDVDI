import torch.utils.data
from data.datasets import DatasetFactory


class CustomDatasetDataLoader:
    def __init__(self, opt, is_for_train=True):
        self._opt = opt
        self._is_for_train = is_for_train
        self._num_threds = opt.n_threads
        self._create_dataset()

    def _create_dataset(self):
        self._dataset = DatasetFactory.get_by_name(self._opt.dataset_mode, self._opt, self._is_for_train)
        if self._is_for_train:
            self._dataloader = torch.utils.data.DataLoader(
                self._dataset,
                batch_size=self._opt.train_batch_size,
                ## TODO
                shuffle=self._is_for_train,
                # shuffle=False,
                num_workers=int(self._num_threds),
                drop_last=True)
        else:
            self._dataloader = torch.utils.data.DataLoader(
                self._dataset,
                batch_size=self._opt.test_batch_size,
                shuffle=self._is_for_train,
                num_workers=int(self._num_threds),
                drop_last=True)

    def load_data(self):
        return self._dataloader

    def __len__(self):
        return len(self._dataset)
