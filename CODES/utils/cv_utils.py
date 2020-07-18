import cv2
import scipy.io as scio
import numpy as np
import torchvision
import torch
import math
import os
from PIL import Image

def read_cv2_img(path, input_nc):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is not None:
        if input_nc == 1:
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = np.expand_dims(img, 0)
        elif input_nc == 3:
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img = img.transpose(2,0,1)
        img = img.astype(np.float32) / 255.0
    else:
        print('No image: ', path)
    return img

def read_cv2_onechannel(path, channel):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img[:,:,channel]
        img = np.expand_dims(img, 0)
        img = img.astype(np.float32) / 255.0
    else:
        print('No image: ', path)
    return img

def read_mat(path, arrayname):
    print(path)
    data = scio.loadmat(path)
    data = data[arrayname].astype(np.float32)
    data = data / 20.0
    data = np.clip(data, -1, 1)
    return data

def read_mat_gopro(path, arrayname):
    data = scio.loadmat(path)
    data = data[arrayname]
    return data

def read_mat_flow(path, arrayname):
    data = scio.loadmat(path)
    data = data[arrayname]
    data = data.transpose(2,0,1)
    data = data[0:2]
    return data

def tensor2im(img, imtype=np.uint8, idx=0, nrows=None, need_normalize = False, if_RGB=1):
    # select a sample or create grid if img is a batch
    if len(img.shape) == 4:
        nrows = nrows if nrows is not None else int(math.sqrt(img.size(0)))
        img = img[idx] if idx >= 0 else torchvision.utils.make_grid(img, nrows)
    if need_normalize:
        if img.max() != img.min():
            img = (img - img.min()) / (img.max() - img.min())
    else:
        img = img.clamp(0.0, 1.0)
    img = img.cpu().detach().float()
    img = img * 255.0
    image_numpy = img.numpy()
    # if if_RGB == 3:
    #     image_numpy = image_numpy.transpose(1,2,0)
    return image_numpy.astype(imtype)

def save_image(image_numpy, image_path, index):
    if not os.path.exists(image_path):
        os.mkdir(image_path)
    if image_numpy.shape[0] == 3:
        image_numpy = image_numpy.transpose(1,2,0)
    else:
        image_numpy = image_numpy[0]
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(os.path.join(image_path, '%04d.png' % index))

def debug_show_tensor(img, name, rela=False):
    img = img.detach().cpu().squeeze().numpy()
    if rela:
        img = (img - img.min()) / (img.max() - img.min())
    else:
        img = np.clip(img, 0.0, 1.0)
    img = (img*255.0).astype(np.uint8)
    cv2.imshow(name, img)

def debug_save_tensor(img, name, rela=False, rgb = False):
    img = img.detach().cpu().squeeze().numpy()
    if rgb:
        img = img.transpose(1,2,0)
        img = img[:,:,::-1]
    if rela:
        img = (img - img.min()) / (img.max() - img.min())
    else:
        img = np.clip(img, 0.0, 1.0)
    img = (img*255.0).astype(np.uint8)
    cv2.imwrite(name, img)

class save_videos():
    def __init__(self, name, inter_num, size):
        self.videowriter = cv2.VideoWriter(name, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), inter_num, size, False)

    def save_video(self, imgs):
        imgs = imgs[0].cpu().numpy()
        imgs = np.clip(imgs, 0.0, 1.0)
        imgs = (imgs*255.0).astype(np.uint8)
        for i in range(imgs.shape[0]):
            self.videowriter.write(imgs[i])


