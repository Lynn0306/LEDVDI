from opts.test_opt import TestOpt
from data.dataloader import CustomDatasetDataLoader
from models.models import ModelsFactory
import os
import shutil
import torch

class Test:
    def __init__(self):
        self._opt = TestOpt().parse()
        self.cal_time = 0.0
        data_loader_test = CustomDatasetDataLoader(self._opt, is_for_train=False)

        self._dataset_test = data_loader_test.load_data()

        self._dataset_test_size = len(data_loader_test)
        print('# Test videos : %d' % self._dataset_test_size)

        self._model = ModelsFactory.get_by_name(self._opt.model, self._opt)

        self.output_dir = os.path.expanduser(self._opt.output_dir)
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)
        print('# Output dir: ', self.output_dir)

        self._test()

    def _test(self):
        # train epoch
        self._test_epoch()

    def _test_epoch(self):

        # set model to eval
        self._model.set_eval()

        for i_val_batch, val_batch in enumerate(self._dataset_test):
            self._model.set_input(val_batch)
            with torch.no_grad():
                self._model.forward(False, isTrain=False)



if __name__ == "__main__":
    Test()