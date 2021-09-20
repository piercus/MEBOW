from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pose_hrnet import PoseHighResolutionNet, Bottleneck, blocks_dict, HighResolutionModule, BasicBlock
BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

class AziHighResolutionNet(PoseHighResolutionNet):

    def __init__(self, cfg, **kwargs):
        super(AziHighResolutionNet, self).__init__(cfg, **kwargs)

        self.azimuth_fc = nn.Linear(512, 360//cfg.DATASET.AZIMUTH_STEP)
        self.polar_fc = nn.Linear(512, 360//cfg.DATASET.POLAR_STEP)
        self.obj_scale_fc = nn.Linear(512, 1)

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3, 1, 1, bias=False
                            ),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU(inplace=True)
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(
                                inchannels, outchannels, 3, 2, 1, bias=False
                            ),
                            nn.BatchNorm2d(outchannels),
                            nn.ReLU(inplace=True)
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        feature_map_2 = y_list[0]

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        feature_map_3 = y_list[0]

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)
        feature_map_4 = y_list[0]
        x = self.final_layer(y_list[0])
        if self.use_featuremap:
            x_cat = torch.cat([feature_map_2, feature_map_3, feature_map_4], 1)
        else:
            x_cat = x
        # y = self.hoe_layer1(x)
        y = self.hoe_layer2(x_cat)
        y = self.hoe_layer3(y)
        y = self.hoe_layer4(y)
        y = self.hoe_avgpool(y)
        y = y.view(y.size(0), -1)
        azimuth = self.azimuth_fc(y)
        azimuth = F.softmax(azimuth, dim=1)
        polar = self.polar_fc(y)
        polar = F.softmax(polar, dim=1)
        obj_scale = self.obj_scale_fc(y)
        return azimuth, polar, obj_scale

    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        # This is for the keypoint network, we use the ImageNet pretrained model
        # if os.path.isfile(pretrained):
        #     pretrained_state_dict = torch.load(pretrained)
        #     logger.info('=> loading pretrained model {}'.format(pretrained))
        #     need_init_state_dict = {}
        #     for name, m in pretrained_state_dict.items():
        #         if name.split('.')[0] in self.pretrained_layers \
        #            or self.pretrained_layers[0] is '*':
        #             need_init_state_dict[name] = m
        #     self.load_state_dict(need_init_state_dict, strict=False)

        # This is the orientation model, we use the keypoint model as pretrained model
        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            self.load_state_dict(pretrained_state_dict, strict=False)
        elif pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(pretrained))

    def get_hoe_params(self):
        for m in self.named_modules():
            if "hoe" in m[0] or "final_layer" in m[0]:
                print(m[0])
                for p in m[1].parameters():
                    if p.requires_grad:
                        yield p

def get_pose_net(cfg, is_train, **kwargs):
    model = AziHighResolutionNet(cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)
    return model

# just for debug
if __name__ == "__main__":
    import sys
    sys.path.append(".")
    sys.path.append("./lib")
    import argparse
    import experiments
    import config
    from config import cfg
    from config import update_config
    import torchvision.transforms as transforms
    import torch

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.cfg = "experiments/coco/segm-4_lr1e-3.yaml"
    args.opts, args.modelDir, args.logDir, args.dataDir = "", "", "", ""
    update_config(cfg, args)
    model = AziHighResolutionNet(cfg)
    model.eval()
    input = torch.rand(1, 3, 256, 192)
    azimuth, polar, obj_scale = model(input)
    print(azimuth.shape, polar.shape, obj_scale.shape)