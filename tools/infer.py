# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import cv2

import _init_paths
from config import cfg
from config import update_config

from core.loss import JointsMSELoss
from core.loss import DepthLoss
from core.loss import hoe_diff_loss
from core.loss import Bone_loss


from utils.utils import create_logger
from utils.utils import get_model_summary
from utils.transforms import get_affine_transform

import dataset
import models
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')
    
    parser.add_argument('--input',
                        help='input file name',
                        type=str)
    
    parser.add_argument('--input-box',
                        help='input box',
                        type=str)
    
    args = parser.parse_args()

    return args

# thi sis inference part 
def infer(imgfile, bbox, image_size, model, transform=None, step=5):

    # switch to evaluate mode
    model.eval()
    plane_output2, hoe_output2 = model(torch.zeros(1,3,256,256))
    print("numpy_hoe_output2", hoe_output2.detach().cpu().numpy())    
    print("imgfile", imgfile)
    print("bbox", bbox)
    data_numpy = cv2.imread(imgfile, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

    if data_numpy is None:
        logger.error('=> fail to read {}'.format(imgfile))
        raise ValueError('Fail to read {}'.format(imgfile))
    
    def _xywh2cs(cfg, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > cfg["aspect_ratio"] * h:
            h = w * 1.0 / cfg["aspect_ratio"]
        elif w < cfg["aspect_ratio"] * h:
            w = h * cfg["aspect_ratio"]
        scale = np.array(
            [w * 1.0 / cfg["pixel_std"], h * 1.0 / cfg["pixel_std"]],
            dtype=np.float32)
        if center[0] != -1:
            scale = scale * 1.25
        return center, scale

    def _box2cs(box):
        x, y, w, h = box[:4]
        return _xywh2cs({
          "pixel_std": 200, 
          "aspect_ratio": image_size[0] * 1.0 / image_size[1]
        }, x, y, w, h)
    
    center, scale = _box2cs(bbox)
    print('center, scale', center, scale)
    trans = get_affine_transform(center, scale, 0, image_size)
    
    input = cv2.warpAffine(
        data_numpy,
        trans,
        (int(image_size[0]), int(image_size[1])),
        flags=cv2.INTER_LINEAR)
    
    if transform:
        input = transform(input)
    
    input = input.float()
    
    with torch.no_grad():
      # compute output
      print('input', input)

      plane_output, hoe_output = model(torch.unsqueeze(input, 0))
      print("numpy_hoe_output", hoe_output.detach().cpu().numpy())

      index_degree = hoe_output.argmax(axis = 1) 
      degree2 = index_degree * step
      size = hoe_output.shape[1]
      deg = degree2.cpu().numpy()
      hoe_base = hoe_output.cpu().numpy()[0]
      print('values are :',deg)
      conf = hoe_base[index_degree]
      print('confidence ~0° is : {:.2%}'.format(conf))
      for i in range(1,10):
        conf+= hoe_base[(index_degree-i)%size]+hoe_base[(index_degree+i)%size]
        print('confidence ~{}° is : {:.2%}'.format(i*step, conf))

      return deg


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {} without strict'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'model_best.pth'
        )
        logger.info('=> loading model from {} with strict'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))
    

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    
    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    if args.input is None:
      raise Exception('--input must be defined')
    if args.input_box is None:
      raise Exception('--input_box must be defined')

    infer(
        args.input, [int(s) for s in args.input_box.split(',')], cfg.MODEL.IMAGE_SIZE, model, transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

if __name__ == '__main__':
    main()
