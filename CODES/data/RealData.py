## real dataset.

import os
from data.datasets import DatasetBase
from data import transforms
from utils import cv_utils
import numpy as np
from collections import OrderedDict

class Dataset(DatasetBase):
    def __init__(self, opt, is_for_train=False):
        super(Dataset, self).__init__(opt, is_for_train=is_for_train)
        self._name = 'RealDataset'
        print('Loading dataset...')
        self.opt = opt
        self.sequence_num = self.opt.sequence_num
        self._read_dataset_paths()

    def _read_dataset_paths(self):
        self.root = os.path.expanduser(self._opt.test_data_dir)
        self.load_blur()
        self.load_event()

    def load_blur(self):
        self.blur_paths = OrderedDict()
        self.dataset_acc_num = [0]
        self.dataset_name = []
        for subroot in sorted(os.listdir(self.root)):
            imgroot = os.path.join(self.root, subroot, 'blurred')
            imglist = os.listdir(imgroot)
            imglist.sort(key=lambda x: float(x[:-4]))
            self.blur_paths[subroot] = imglist
            self.dataset_acc_num.append(len(imglist) + self.dataset_acc_num[-1] - self.sequence_num * 6)
            self.dataset_name.append(subroot)

    def load_event(self):
        self.eventdata = OrderedDict()
        for subroot in sorted(os.listdir(self.root)):
            eventpath = os.path.join(self.root, subroot, self.opt.event_name+'.mat')
            eventdata = cv_utils.read_mat(eventpath, 'event_bins')
            self.eventdata[subroot] = eventdata

    def __len__(self):
        return len(self.dataset_name)

    def __getitem__(self, index):
        ###################### TEST ########################
        dataname = self.dataset_name[index]
        # blurred images
        blur_paths = self.blur_paths.get(dataname)
        blurs = []
        num_data = len(blur_paths)
        for i in range(num_data):
            blur_path = os.path.join(self.root, dataname, 'blurred', blur_paths[i])
            blur = cv_utils.read_cv2_img(blur_path, input_nc=1)
            blurs.append(blur)
        blurs = np.concatenate(blurs, axis=0)
        # event images
        events = self.eventdata.get(dataname)
        sample = {'events': events,
                  'blurred': blurs,
                  'dataname': dataname}

        return sample



    def _create_transform(self):
        print('Set Augmentation...')
        self.transforms = transforms.Compose([
            transforms.RandomCrop([self.opt.crop_height, self.opt.crop_width]),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip()
        ])