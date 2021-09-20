from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import json
import cv2
import torch
import random
import numpy as np
from scipy.io import loadmat, savemat
from collections import OrderedDict
import torch.utils.data as data

from pycocotools.coco import COCO

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import hoe_heatmap_gen

logger = logging.getLogger(__name__)

class AZI_Dataset(data.Dataset):
    def __init__(self, cfg, root, is_train, transform=None):
        self.image_width = cfg.MODEL.IMAGE_SIZE[0]
        self.image_height = cfg.MODEL.IMAGE_SIZE[1]
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.pixel_std = 200
        self.root = root
        self.is_train = is_train
        if is_train:
            json_path = os.path.join(root, 'annotations', 'train_azi.json')
            dataType = 'train2017'
        else:
            json_path = os.path.join(root, 'annotations', 'val_azi.json')
            dataType = 'val2017'
        json_file = open(json_path, 'r')
        self.img_list = list(json.load(json_file).items())
        logger.info('=> load {} samples'.format(len(self.img_list)))
        print('=> load {} samples'.format(len(self.img_list)))

        annFile = os.path.join(root, 'annotations', 'instances_{}.json'.format(dataType))
        self.coco_ann = COCO(annFile)

        # set parameters for key points

        # for data processing
        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP

        # generate heatmap label
        self.target_type = cfg.MODEL.TARGET_TYPE
        self.image_size = np.array(cfg.MODEL.IMAGE_SIZE)
        self.sigma = cfg.MODEL.SIGMA
        self.azimuth_sigma = cfg.DATASET.AZIMUTH_SIGMA
        self.polar_sigma = cfg.DATASET.POLAR_SIGMA
        self.azimuth_step = cfg.DATASET.AZIMUTH_STEP
        self.polar_step = cfg.DATASET.POLAR_STEP

        self.transform = transform

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25
        return center, scale

    def __len__(self,):
        return len(self.img_list)

    def _load_image(self, index):
        azimuth_degree = self.img_list[index][1]['azimuth']
        polar_degree = self.img_list[index][1]['polar']
        obj_scale = self.img_list[index][1]['horse_scale']
        str_id = self.img_list[index][0]
        img_id = int(str_id.split('_')[0])
        ann_id = int(str_id.split('_')[1])

        img_ann = self.coco_ann.loadImgs(img_id)[0]
        kps_ann = self.coco_ann.loadAnns(ann_id)[0]

        img_name = img_ann['file_name']
        if self.is_train:
            img_path = os.path.join(self.root, 'images', 'train2017', img_name)
        else:
            img_path = os.path.join(self.root, 'images', 'val2017', img_name)

        # label of orienation degree
        azimuth_degree = int(azimuth_degree) // self.azimuth_step
        polar_degree = int(polar_degree) // self.polar_step
        bbox = kps_ann['bbox']
        
        center, scale = self._box2cs(bbox)
        return img_path, center, scale, azimuth_degree, polar_degree, obj_scale

    def __getitem__(self, index):
        imgfile, c, s, azimuth_degree, polar_degree, obj_scale = self._load_image(index)
        data_numpy = cv2.imread(imgfile, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        if data_numpy is None:
            logger.error('=> fail to read {}'.format(imgfile))
            raise ValueError('Fail to read {}'.format(imgfile))

        # Not use score
        # score = 0

        if self.is_train:

            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            # Not use r
            r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                
                c[0] = data_numpy.shape[1] - c[0] - 1
                
                n_azimuth_digit = 360//self.azimuth_step
                
                azimuth_degree = (n_azimuth_digit - azimuth_degree) % n_azimuth_digit

        azimuth_degree = hoe_heatmap_gen(azimuth_degree, 360//self.azimuth_step, sigma=self.azimuth_sigma)
        polar_degree = hoe_heatmap_gen(polar_degree, 360//self.polar_step, sigma=self.polar_sigma)
        
        trans = get_affine_transform(c, s, 0, self.image_size)
        input = cv2.warpAffine(
            data_numpy,
            trans,
            (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)
        if self.transform:
            input = self.transform(input)
        input = input.float()

        meta = {
            'image_path': imgfile,
            'center': c,
            'scale': s,
        }
        if(self.image_size[0] != self.image_size[1]):
          raise Exception('Please debug here the obj_scale')

        # print("src_size", data_numpy.shape[0]*obj_scale, "imgfile", imgfile, "obj_scale", obj_scale, "dest_img", self.image_size[0],"src_size", s[0]*200, "out", data_numpy.shape[0]*obj_scale * (self.image_size[0]/(s[0]*200))/self.image_size[0])
        
        obj_scale = data_numpy.shape[0]*obj_scale * (self.image_size[0]/(s[0]*200))/self.image_size[0]
        obj_scale = np.array([obj_scale], dtype=np.float32)
        return input, 0, 0, azimuth_degree, polar_degree, obj_scale, meta

if __name__ == '__main__':
    import argparse
    from config import cfg
    from config import update_config
    import torchvision.transforms as transforms
    import torch

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.cfg = "experiments/w32_256x192_adam_lr1e-3.yaml"
    args.opts, args.modelDir, args.logDir, args.dataDir = "", "", "", ""
    update_config(cfg, args)
    normalize = transforms.Normalize(
        mean=[0.485, 0.485, 0.485], std=[0.229, 0.229, 0.229]
    )
    train_dataset = AZI_Dataset(
        cfg, cfg.DATASET.TRAIN_ROOT, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    for i, b in enumerate(train_loader):
        if i == 5:
            break
        else:
            print(b[0].shape, b[1].shape, b[2].shape)
            print('fdsa')