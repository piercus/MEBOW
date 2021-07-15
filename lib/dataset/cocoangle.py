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

class COCO_ANGLE_Dataset(data.Dataset):
    def __init__(self, cfg, root, is_train, transform=None):
        self.image_width = cfg.MODEL.IMAGE_SIZE[0]
        self.image_height = cfg.MODEL.IMAGE_SIZE[1]
        self.aspect_ratio = self.image_width * 1.0 / self.image_height
        self.pixel_std = 200
        self.root = root
        self.is_train = is_train
        if is_train:
            json_path = os.path.join(root, 'annotations', 'train_hoe.json')
            dataType = 'train2017'
        else:
            json_path = os.path.join(root, 'annotations', 'val_hoe.json')
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
        self.hoe_sigma = cfg.DATASET.HOE_SIGMA

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
        degree = self.img_list[index][1]
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
        degree = int(degree) // 5
        bbox = kps_ann['bbox']
        
        center, scale = self._box2cs(bbox)
        return img_path, center, scale, degree

    def __getitem__(self, index):
        imgfile, c, s, degree = self._load_image(index)
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

                degree = (72 - degree) % 72

        degree = hoe_heatmap_gen(degree, 72, sigma=self.hoe_sigma)
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

        return input, 0, 0, degree, meta

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
    train_dataset = COCO_ANGLE_Dataset(
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