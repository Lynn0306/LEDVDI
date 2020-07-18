import argparse

class BaseOpt():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--name', type=str, default='DeblurAndInterpolation')

        # dataset
        self.parser.add_argument('--sequence_num', type=int, default=10,
                                 help='number of blurred frames in a sequence')
        self.parser.add_argument('--dataset_mode', type=str, default='Real',
                                 help='Beijing, GoProLMDB or Real')
        self.parser.add_argument('--event_name', type=str, default='EventBin3')
        self.parser.add_argument('--eventbins_between_frames', type=int, default=3,
                            help='number of event bins between 2 sharp frames')
        self.parser.add_argument('--width', type=int, default=240)
        self.parser.add_argument('--height', type=int, default=180)

        # dataloader
        self.parser.add_argument('--n_threads', default=4, type=int, help='# threads for data')

        # model
        self.parser.add_argument('--model', type=str, default='Ours_Reconstruction_Smaller',
                                 help='model to run')
        self.parser.add_argument('--load_G', type=str, default='', help='path of the pretrained model')

        # other options
        # self.parser.add_argument('--Gopro', action='store_true',
        #                          help='set this option to train on Gopro dataset.')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()

        self.opt.inter_num = 6
        self.opt.channel = 1

        # if self.opt.Gopro:
        #     self.opt.inter_num = 10
        #     self.opt.channel = 3

        self.opt.is_train = self.is_train

        return self.opt