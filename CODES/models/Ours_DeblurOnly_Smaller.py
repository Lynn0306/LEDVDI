## input: event, blur and last sharp
## output: delta L

import torch
from models.models import BaseModel
from networks.networks import NetworksFactory
from utils import cv_utils
from collections import OrderedDict
import os

class Model(BaseModel):
    def __init__(self, opt):
        super(Model, self).__init__(opt)
        self._name = 'Ours_DeblurOnly_Smaller'
        self.sequence_num = self._opt.sequence_num
        self.eventbins_between_frames = self._opt.eventbins_between_frames
        self.inter_num = self._opt.inter_num
        self.channel = self._opt.channel
        self._init_create_networks()
        self._init_inputs()

    def _init_create_networks(self):

        self._G = NetworksFactory.get_by_name('Ours_DeblurOnly',
                                              eventbins_between_frames = self.eventbins_between_frames,
                                              if_RGB=self._opt.channel, inter_num=self.inter_num)
        if len(self._opt.load_G) > 0:
            self._load_network(self._G, self._opt.load_G)
        else:
            raise ValueError("Weights file not found.")
        self._G.cuda()

    def _init_inputs(self):
        self._input_blurred = self._Tensor()
        self._input_event = self._Tensor()

    def set_input(self, input):
        self._input_blurred.resize_(input['blurred'].size()).copy_(input['blurred'])
        self._input_event.resize_(input['events'].size()).copy_(input['events'])

        self._input_blurred = self._input_blurred.cuda()
        self._input_event = self._input_event.cuda()
        self.dataname = input['dataname'][0]
        self.imgcount = 0

    def set_eval(self):
        self._G.eval()
        self._is_train = False

    def forward(self, keep_data_for_visuals=False, isTrain=True):
        #
        blur_list = torch.split(self._input_blurred, split_size_or_sections=self.channel, dim=1)
        self.last_sharps = blur_list[0]
        self.last_blur = blur_list[0]
        self.last_events = self._input_event[:, 0:self.inter_num * self.eventbins_between_frames]

        VideoWriter = cv_utils.save_videos(os.path.join(self._opt.output_dir, self.dataname+'.mp4'),
                                           1, (self._opt.width * 2, self._opt.height))

        for blur_index in range(len(blur_list)):
            self.cur_blur = blur_list[blur_index]
            self.cur_event = self._input_event[:, (blur_index * self.inter_num * self.eventbins_between_frames):
                                                ((blur_index+1) * self.inter_num * self.eventbins_between_frames)]

            self.est_sharp = self._G(self.cur_blur, self.cur_event, self.last_sharps, self.last_blur, self.last_events)

            # update.
            if blur_index % self.sequence_num == 0:
                self.last_sharps = self.cur_blur
                self.last_blur = self.cur_blur
                self.last_events = self.cur_event
            else:
                self.last_sharps = self.est_sharp.detach()
                self.last_blur = self.cur_blur
                self.last_events = self.cur_event

            ############# save imgs #############
            path = os.path.join(self._opt.output_dir, self.dataname)
            if not os.path.exists(path):
                os.makedirs(path)
            path_i = os.path.join(path, str(self.imgcount) + '_deblur.png')
            cv_utils.debug_save_tensor(self.est_sharp, path_i, rela=False, rgb=self.channel == 3)
            self.imgcount = self.imgcount + 1
            ############# save video #############
            concat = torch.cat((self.cur_blur, self.est_sharp), -1)
            VideoWriter.save_video(concat)