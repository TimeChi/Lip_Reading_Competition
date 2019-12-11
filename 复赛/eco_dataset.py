import torch.utils.data as data
from PIL import Image
import os
import os.path
import cv2
import numpy as np
import random

class TSNDataSet(data.Dataset):
    def __init__(self, root_path, mode='train',
                 num_segments=24, img_size=184, transform=None,
                 lip_dict=None, video_list=None):

        self.root_path = root_path
        self.mode = mode
        self.num_segments = num_segments
        self.transform = transform
        self.lip_dict = lip_dict
        self.video_list = video_list
        self.img_size = img_size

    def padding_img_full(self, pilimg, size=256):
        img = cv2.cvtColor(np.asarray(pilimg), cv2.COLOR_RGB2BGR)
        h, w, _ = img.shape
        image = np.zeros((size, size, 3), np.uint8)
        if h > w:
            ratio = size / h
            resize_h = size
            resize_w = int(ratio * w)
            pad_left = (size - resize_w) // 2
            pad_right = pad_left + resize_w
            img = cv2.resize(img, (resize_w, resize_h))
            image[:, pad_left:pad_right, :] = img
        elif h < w:
            ratio = size / w
            resize_w = size
            resize_h = int(ratio * h)
            pad_top = (size - resize_h) // 2
            pad_bottom = pad_top + resize_h
            img = cv2.resize(img, (resize_w, resize_h))
            image[pad_top:pad_bottom, :, :] = img
        else:
            image = cv2.resize(img, (size, size))
        if random.random() > 0.6 and self.mode == 'train':
            image = self.addGaussianNoise(image, 0.01)
        else:
            if random.random() > 0.6 and self.mode == 'train':
                image = cv2.bilateralFilter(src=image, d=0, sigmaColor=random.randint(15, 30), sigmaSpace=15)

        pilimage = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return pilimage

    def addGaussianNoise(self, image, percetage):
        G_Noiseimg = image.copy()
        w = image.shape[1]
        h = image.shape[0]
        G_NoiseNum = int(percetage * image.shape[0] * image.shape[1])
        for i in range(G_NoiseNum):
            temp_x = np.random.randint(0, h)
            temp_y = np.random.randint(0, w)
            G_Noiseimg[temp_x][temp_y][np.random.randint(3)] = np.random.randn(1)[0]
        return G_Noiseimg

    def __getitem__(self, index):
        record = self.video_list[index]

        lip_path = os.path.join(self.root_path, record)
        pic_names = os.listdir(lip_path)
        real_pic_size = len(pic_names)

        coeff = 1
        doubl_copy = False
        if real_pic_size < 5 and self.mode == 'train' and random.random() > 0.3:
            coeff = 2
            # doubl_copy = True
        pic_size = real_pic_size*coeff #图片的数量
        step = int(self.num_segments / pic_size)  # 每张图复制几次
        great = self.num_segments % pic_size  # 前几张图多复制一次

        start = 0
        if great > 2 and self.mode == 'train' and random.random() > 0.3:
            start = random.randint(0, pic_size-great)

        drop_flag, drop_nums = False, 0
        if great > 2 and self.mode == 'train' and pic_size > 12 and random.random() > 0.3:
            drop_nums = random.randint(0, 2)
            drop_flag = True


        images = list()
        last_img = None
        dx,iddx = 1, 0
        resize_random = random.random()
        for idx in range(pic_size):
            if coeff == 2 and idx >=real_pic_size:
                iddx -= real_pic_size
                if doubl_copy == True:
                    dx = 1
                    doubl_copy = False

            img_path = os.path.join(lip_path, str(idx + dx + iddx) + '.png')
            while (os.path.exists(img_path) == False):
                dx += 1
                img_path = os.path.join(lip_path, str(idx + dx + iddx) + '.png')

            if drop_nums != 0 and drop_flag == False and idx > pic_size // 2 and random.random() > 0.5:
                img = last_img
                drop_nums -= 1
            else:
                img = Image.open(img_path)
            if drop_flag == True:
                drop_flag = False

            if drop_nums != 0:
                last_img = img

            # img_data =[img.convert('RGB')]
            box = (img.size[0] * 0.04, img.size[1] * 0.1, img.size[0] * 0.98, img.size[1])
            region = img.crop(box)
            if resize_random > 0.5 and self.mode == 'train':
                region = self.padding_img_full(region, min(
                    max(int(max(region.size[0], region.size[1]) * random.uniform(0.3, 0.8)), self.img_size*200//184), self.img_size*215//184))
            else:
                region = self.padding_img_full(region, self.img_size*200//184)

            img_data = [region.convert('RGB')]
            if start == 0 :
                if idx < great:  # 前几张多复制一次
                    for _ in range(step + 1):
                        images.extend(img_data)
                else:
                    for _ in range(step):
                        images.extend(img_data)
            else:
                if idx < start:
                    for _ in range(step):
                        images.extend(img_data)
                elif idx < start + great:
                    for _ in range(step + 1):
                        images.extend(img_data)
                else:
                    for _ in range(step):
                        images.extend(img_data)


        process_data = self.transform(images)
        if self.mode == 'train' or self.mode == 'val':
            record_label = int(self.lip_dict[record])
            return process_data,record_label,record#图像，标签，文件名
        else:
            return process_data,record

    def __len__(self):
        return len(self.video_list)


class TSNDataSet_infer(data.Dataset):
    def __init__(self, root_path,
                 num_segments=3, img_size=184, transform=None,
                 lip_dict=None, video_list=None):

        self.root_path = root_path
        self.num_segments = num_segments
        self.transform = transform
        self.lip_dict = lip_dict
        self.video_list = video_list
        self.img_size = img_size

    def padding_img_full(self, pilimg, size=256):
        img = cv2.cvtColor(np.asarray(pilimg), cv2.COLOR_RGB2BGR)
        h, w, _ = img.shape
        image = np.zeros((size, size, 3), np.uint8)
        if h > w:
            ratio = size / h
            resize_h = size
            resize_w = int(ratio * w)
            pad_left = (size - resize_w) // 2
            pad_right = pad_left + resize_w
            img = cv2.resize(img, (resize_w, resize_h))
            image[:, pad_left:pad_right, :] = img
        elif h < w:
            ratio = size / w
            resize_w = size
            resize_h = int(ratio * h)
            pad_top = (size - resize_h) // 2
            pad_bottom = pad_top + resize_h
            img = cv2.resize(img, (resize_w, resize_h))
            image[pad_top:pad_bottom, :, :] = img
        else:
            image = cv2.resize(img, (size, size))

        pilimage = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return pilimage

    def __getitem__(self, index):
        record = self.video_list[index]

        lip_path = os.path.join(self.root_path, record)
        pic_names = os.listdir(lip_path)
        real_pic_size = len(pic_names)


        pic_size = real_pic_size #图片的数量
        step = int(self.num_segments / pic_size)  # 每张图复制几次
        great = self.num_segments % pic_size  # 前几张图多复制一次


        images = list()
        dx = 1
        for idx in range(pic_size):
            img_path = os.path.join(lip_path, str(idx + dx) + '.png')

            while (os.path.exists(img_path) == False):
                dx += 1
                img_path = os.path.join(lip_path, str(idx + dx) + '.png')

            img = Image.open(img_path)

            # img_data =[img.convert('RGB')]
            box = (img.size[0] * 0.04, img.size[1] * 0.1, img.size[0] * 0.98, img.size[1])
            region = img.crop(box)
            region = self.padding_img_full(region, self.img_size*200//184)

            img_data = [region.convert('RGB')]

            if idx < great:  # 前几张多复制一次
                for _ in range(step + 1):
                    images.extend(img_data)
            else:
                for _ in range(step):
                    images.extend(img_data)

        process_data = self.transform(images)
        return process_data,record

    def __len__(self):
        return len(self.video_list)