from networks.networks import NetworkBase
from networks.submodules import *
from networks.FAC.kernelconv2d import KernelConv2D
import torch

class Net(NetworkBase):
    def __init__(self, if_RGB=1, inter_num=6, eventbins_between_frames=3):
        super(Net,self).__init__()
        self._name = 'Ours_DeblurOnly_Smaller'
        self.network = SubModule(if_RGB=if_RGB, inter_num=inter_num, eventbins_between_frames=eventbins_between_frames)

    def forward(self, blur, events, lastSharps, lastBlur, lastEvents):
        return self.network(blur, events, lastSharps, lastBlur, lastEvents)

class SubModule(nn.Module):
    def __init__(self, if_RGB=1, inter_num=6, eventbins_between_frames=3):
        super(SubModule, self).__init__()
        self.inter_num = inter_num
        self.events_nc = eventbins_between_frames * inter_num
        self.if_RGB = if_RGB
        ks = 3
        ks_2d = 5
        ch0 = 32
        ch1 = 64
        ch2 = 96
        ch3 = 128

        #############################
        # Dynamic Filter Generation
        #############################
        self.kconv1_1 = conv(if_RGB * 3 + 2 * self.events_nc, ch1, kernel_size=ks, stride=1)
        self.kconv1_2 = resnet_block(ch1, kernel_size=ks)
        self.kconv1_3 = resnet_block(ch1, kernel_size=ks)

        self.kconv2_1 = conv(ch1, ch2, kernel_size=ks, stride=2)
        self.kconv2_2 = resnet_block(ch2, kernel_size=ks)
        self.kconv2_3 = resnet_block(ch2, kernel_size=ks)

        self.kconv3_1 = conv(ch2, ch3, kernel_size=ks, stride=2)
        self.kconv3_2 = resnet_block(ch3, kernel_size=ks)
        self.kconv3_3 = resnet_block(ch3, kernel_size=ks)

        self.fac_deblur1 = nn.Sequential(
            conv(ch3, ch3, kernel_size=ks),
            resnet_block(ch3, kernel_size=ks),
            resnet_block(ch3, kernel_size=ks))
        self.fac_deblur2 = conv(ch3, ch3 * ks_2d ** 2, kernel_size=1)

        #############################
        # Event Feature Extraction
        #############################
        self.conv1_1 = conv(2 * self.events_nc, ch1, kernel_size=ks, stride=1)
        self.conv1_2 = resnet_block(ch1, kernel_size=ks)
        self.conv1_3 = resnet_block(ch1, kernel_size=ks)

        self.conv2_1 = conv(ch1, ch2, kernel_size=ks, stride=2)
        self.conv2_2 = resnet_block(ch2, kernel_size=ks)
        self.conv2_3 = resnet_block(ch2, kernel_size=ks)

        self.conv3_1 = conv(ch2, ch3, kernel_size=ks, stride=2)
        self.conv3_2 = resnet_block(ch3, kernel_size=ks)
        self.conv3_3 = resnet_block(ch3, kernel_size=ks)

        self.kconv_deblur = KernelConv2D.KernelConv2D(kernel_size=ks_2d)

        #############################
        # Multi-Residual Prediction
        #############################
        self.upconv2_u = upconv(ch3, ch2)
        self.upconv2_u_cat = conv(ch2 * 2, ch2, kernel_size=ks)
        self.upconv2_2 = resnet_block(ch2, kernel_size=ks)
        self.upconv2_1 = resnet_block(ch2, kernel_size=ks)

        self.upconv1_u = upconv(ch2, ch1)
        self.upconv1_u_cat = conv(ch1 * 2, ch1, kernel_size=ks)

        self.delta_blur_ = nn.Sequential(
            conv(ch1, ch0, kernel_size=ks),
            resnet_block(ch0, kernel_size=ks),
            resnet_block(ch0, kernel_size=ks),
            conv(ch0, if_RGB, kernel_size=ks)
        )
        self.delta_last_ = nn.Sequential(
            conv(ch1, ch0, kernel_size=ks),
            resnet_block(ch0, kernel_size=ks),
            resnet_block(ch0, kernel_size=ks),
            conv(ch0, if_RGB, kernel_size=ks)
        )

        #############################
        # GateNet
        #############################
        self.gate_block = nn.Sequential(
            nn.Conv3d(3*self.if_RGB + self.events_nc, ch1, kernel_size=ks, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(ch1, ch1, kernel_size=ks, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(ch1, 2*self.if_RGB, kernel_size=ks, padding=1),
            nn.Sigmoid()
        )
        self.act = nn.Tanh()


    def forward(self, blur, events, lastSharps, lastBlur, lastEvents):
        merge = torch.cat([blur, events, lastSharps, lastBlur, lastEvents], 1)

        #############################
        # Dynamic Filter Generation
        #############################
        kconv1 = self.kconv1_3(self.kconv1_2(self.kconv1_1(merge)))
        kconv2 = self.kconv2_3(self.kconv2_2(self.kconv2_1(kconv1)))
        kconv3 = self.kconv3_3(self.kconv3_2(self.kconv3_1(kconv2)))
        # fac
        kernel_deblur1 = self.fac_deblur1(kconv3)
        kernel_deblur2 = self.fac_deblur2(kernel_deblur1)

        #############################
        # Event Feature Extraction
        #############################
        events_input = torch.cat((events, lastEvents), 1)
        conv1_d = self.conv1_1(events_input)
        conv1_d = self.conv1_3(self.conv1_2(conv1_d))

        conv2_d = self.conv2_1(conv1_d)
        conv2_d = self.conv2_3(self.conv2_2(conv2_d))

        conv3_d = self.conv3_1(conv2_d)
        conv3_d = self.conv3_3(self.conv3_2(conv3_d))

        conv3_d_k = self.kconv_deblur(conv3_d, kernel_deblur2)

        #############################
        # Multi-Residual Prediction
        #############################
        upconv2 = self.upconv2_u(conv3_d_k)
        upconv2 = torch.cat((upconv2, conv2_d), 1)
        upconv2 = self.upconv2_1(self.upconv2_2(self.upconv2_u_cat(upconv2)))
        upconv1 = self.upconv1_u(upconv2)
        upconv1 = torch.cat((upconv1, conv1_d), dim=1)
        upconv1 = self.upconv1_u_cat(upconv1)

        # estimate center C_{i,0}: multiply blur
        res_blur = self.delta_blur_(upconv1)
        output_centers_blur = self.act(blur * res_blur)
        # estimate center P_{i,0,0}: multiply last sharp
        res_last = self.delta_last_(upconv1)
        output_centers_lastsharp = self.act(lastSharps * res_last)
        # F_{i,0,k}
        output_centers = torch.cat((output_centers_blur, output_centers_lastsharp), 1)
        output_centers_ = output_centers.unsqueeze(2)

        #############################
        # GateNet
        #############################
        gates = torch.cat((output_centers_, blur.unsqueeze(2), events.unsqueeze(2)), 1)
        gates = self.gate_block(gates)
        output_merge = output_centers_ * gates
        output_merge = output_merge[:,:self.if_RGB,0] + output_merge[:,self.if_RGB:,0]

        return output_merge