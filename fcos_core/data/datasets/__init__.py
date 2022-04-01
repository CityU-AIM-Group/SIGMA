# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .coco import COCODataset
from .voc import PascalVOCDataset
from .sim10k import Sim10kDataset
from .kitti import KittiDataset
from .concat_dataset import ConcatDataset

from .watercolor import WatercolorDataset
from .voc_watercolor import WaterColorDataset


__all__ = ["COCODataset", "ConcatDataset", "PascalVOCDataset", "Sim10kDataset", "KittiDataset"]
