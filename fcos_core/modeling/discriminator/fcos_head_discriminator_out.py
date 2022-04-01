import torch
import logging
import time
import torch.nn.functional as F
from torch import nn
from fcos_core.structures.bounding_box import BoxList
# from fcos_core.structures.boxlist_ops import boxlist_iou, remove_small_boxes
def see( data):
    print('max: ', torch.max(data))
    print('mean: ', torch.mean(data))
    print('min: ', torch.min(data), '\n')
from .layer import GradientReversal, FocalLoss

class FCOSDiscriminator_out(nn.Module):
    def __init__(
            self,
            num_convs=2,
            in_channels=6,
            cls_map_pre = None,
            grad_reverse_lambda=-1.0,
            out_weight=0.0,
            out_loss='ce',
            grl_applied_domain='both',
            outmap_op='sigmoid',
            output_reg_da=False,
            output_cls_da=False,
            output_centerness_da=False,
            base_dis_tower = False,
            ):
        self.cls_map_pre = cls_map_pre
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSDiscriminator_out, self).__init__()
        dis_tower = []
        # if base_dis_tower:
        if not base_dis_tower:
            for i in range(num_convs):
                dis_tower.append(
                    nn.Conv2d(
                        in_channels,
                        in_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1
                    )
                )
                # dis_tower.append(nn.BatchNorm2d(in_channels))
                # dis_tower.append(nn.ReLU())
                dis_tower.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            self.cls_logits = nn.Conv2d(
                in_channels, 1, kernel_size=3, stride=1,
                padding=1
            )
        else: # seg_tower
            ndf = 64
            dis_tower.append(nn.Conv2d(in_channels, ndf, kernel_size=3, stride=2, padding=1))
            dis_tower.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            for i in range(num_convs - 1):
                dis_tower.append(
                    nn.Conv2d(
                        ndf * (2**i), ndf * (2**(i+1)), kernel_size=3, stride=2, padding=1
                    )
                )
                dis_tower.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

            if self.cls_map_pre == 'roi':
                dis_tower.append(nn.AdaptiveAvgPool2d((7, 7)))
                # self.cls_logits = nn.Linear( 512, 1  )
                self.cls_logits = nn.Conv2d(
                    ndf * (2 ** (num_convs - 1)), 1, kernel_size=3, stride=1, padding=1
                )

            else:
                dis_tower.append(nn.AdaptiveAvgPool2d((1, 1)))
                # self.cls_logits = nn.Linear( 512, 1  )
                self.cls_logits = nn.Conv2d(
                    ndf * (2**(num_convs - 1)), 1, kernel_size=1, stride=1, padding=0
                )
        self.add_module('dis_tower', nn.Sequential(*dis_tower))

        # initialization
        for modules in [self.dis_tower, self.cls_logits]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    # torch.nn.init.normal_(l.weight, std=0.01)
                    # torch.nn.init.constant_(l.bias, 0)
                    torch.nn.init.kaiming_normal(l.weight)
                    torch.nn.init.constant_(l.bias, 0)

        self.grad_reverse = GradientReversal(grad_reverse_lambda)
        self.loss_ce = nn.BCEWithLogitsLoss()
        self.loss_ce_no_reduce = nn.BCEWithLogitsLoss(reduction='none')

        # hyperparameters
        assert out_loss == 'ce' or out_loss == 'focal' or out_loss == 'ce_no_reduce'
        self.out_weight = out_weight
        self.out_loss = out_loss

        ####################### Additional Settings #########################
        self.outmap_op = outmap_op
        self.output_reg_da = output_reg_da
        self.output_cls_da = output_cls_da
        self.output_centerness_da = output_centerness_da

    def forward(self, target, score_map=None, domain='source'):
        assert target == 0 or target == 1 or target == 0.1 or target == 0.9
        assert domain == 'source' or domain == 'target'

        # Get score maps
        box_regression_map = score_map["box_regression"].clone() # .clone() will retain the grad

        # print(torch.mean(box_regression_map.sigmoid()))

        box_cls_map = score_map["box_cls"].clone()
        centerness_map = score_map["centerness"].clone()
        map_list = []
        #
        # if self.cls_map_pre == 'softmax':
        #     box_cls_map = box_cls_map.softmax(dim=1)

        if self.cls_map_pre == 'softmax':
            if self.output_reg_da:
                map_list.append(box_regression_map.sigmoid())
            if self.output_cls_da:
                map_list.append(box_cls_map.softmax(dim=1))
            if self.output_centerness_da:
                map_list.append(centerness_map.sigmoid())

            output_map = torch.cat(map_list, dim=1)
            n, c, h, w = output_map.shape
        elif self.cls_map_pre == 'naive':
            map_list.append(box_cls_map)
            map_list.append(centerness_map.sigmoid())
            map_list.append(box_regression_map)
            output_map = torch.cat(map_list, dim=1)
            n, c, h, w = output_map.shape

        # Generate output feature map for da
        else:
            if self.output_reg_da:
                map_list.append(box_regression_map)
            if self.output_cls_da:
                map_list.append(box_cls_map)
            if self.output_centerness_da:
                map_list.append(centerness_map)
            assert map_list != []
            output_map = torch.cat(map_list, dim=1)
            n, c, h, w = output_map.shape
            if self.outmap_op == 'sigmoid':
                output_map = output_map.sigmoid()
            elif self.outmap_op == 'maxpool':
                maxpooling = nn.AdaptiveMaxPool3d((1, h, w))
                box_cls_map = maxpooling(box_cls_map)
                output_map = torch.cat((box_regression_map, box_cls_map, centerness_map), dim=1).sigmoid()
            elif self.outmap_op == 'attreg':
                assert len(map_list) == 3, "attreg must align cls, reg, ctr"
                maxpooling = nn.AdaptiveMaxPool3d((1, h, w))
                box_cls_map = maxpooling(box_cls_map.sigmoid())
                attmap = (self.out_weight * box_cls_map * centerness_map.sigmoid()).sigmoid()
                output_map = torch.cat((attmap, box_regression_map.sigmoid()), dim=1)
            elif self.outmap_op == 'none':
                output_map = output_map
            else:
                raise NotImplementedError
        #

        output_map = self.grad_reverse(output_map)

        # Forward
        x = self.dis_tower(output_map)
        x = self.cls_logits(x)

        target = torch.full(x.shape, target, dtype=torch.float, device=x.device)
        if self.out_loss == 'focal':
            loss = FocalLoss(x, target)
        elif self.out_loss == 'ce':
            loss = self.loss_ce(x, target)
        elif self.out_loss == 'ce_no_reduce':
            loss = self.loss_ce_no_reduce(x, target)
        else:
            raise NotImplementedError

        # return loss
        return loss
