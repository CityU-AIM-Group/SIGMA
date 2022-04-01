# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os

from yacs.config import CfgNode as CN


# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
_C.MODEL.RPN_ONLY = False
_C.MODEL.MASK_ON = False
_C.MODEL.ATSS_ON = False
_C.MODEL.FCOS_ON = False
_C.MODEL.DA_ON = True
_C.MODEL.RETINANET_ON = False
_C.MODEL.KEYPOINT_ON = False
_C.MODEL.DEVICE = "cuda"
_C.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"
_C.MODEL.CLS_AGNOSTIC_BBOX_REG = False

# If the WEIGHT starts with a catalog://, like :R-50, the code will look for
# the path in paths_catalog. Else, it will use it as the specified absolute
# path
_C.MODEL.WEIGHT = ""
_C.MODEL.USE_SYNCBN = False


# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the smallest side of the image during training
_C.INPUT.MIN_SIZE_TRAIN = (800,)  # (800,)
# The range of the smallest side for multi-scale training
_C.INPUT.MIN_SIZE_RANGE_TRAIN = (-1, -1)  # -1 means disabled and it will use MIN_SIZE_TRAIN
# Maximum size of the side of the image during training
_C.INPUT.MAX_SIZE_TRAIN = 1333
# Size of the smallest side of the image during testing
_C.INPUT.MIN_SIZE_TEST = 800
# Maximum size of the side of the image during testing
_C.INPUT.MAX_SIZE_TEST = 1333
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [102.9801, 115.9465, 122.7717]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [1., 1., 1.]
# Convert image to BGR format (for Caffe2 models), in range 0-255
_C.INPUT.TO_BGR255 = True


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# # List of the dataset names for training, as present in paths_catalog.py
# _C.DATASETS.TRAIN = ()
# List of the dataset names for training of source domain, as present in paths_catalog.py
_C.DATASETS.TRAIN_SOURCE = ()
# List of the dataset names for training of target domain, as present in paths_catalog.py
_C.DATASETS.TRAIN_TARGET = ()
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ()

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4
# If > 0, this enforces that each collated batch should have a size divisible
# by SIZE_DIVISIBILITY
_C.DATALOADER.SIZE_DIVISIBILITY = 0
# If True, each batch should contain only images for which the aspect ratio
# is compatible. This groups portrait images together, and landscape images
# are not batched with portrait images.
_C.DATALOADER.ASPECT_RATIO_GROUPING = True


# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()

# The backbone conv body to use
# The string must match a function that is imported in modeling.model_builder
# (e.g., 'FPN.add_fpn_ResNet101_conv5_body' to specify a ResNet-101-FPN
# backbone)
_C.MODEL.BACKBONE.CONV_BODY = "R-50-C4"

# Add StopGrad at a specified stage so the bottom layers are frozen
_C.MODEL.BACKBONE.FREEZE_CONV_BODY_AT = 2
# GN for backbone
_C.MODEL.BACKBONE.USE_GN = False


# ---------------------------------------------------------------------------- #
# FPN options
# ---------------------------------------------------------------------------- #
_C.MODEL.FPN = CN()
_C.MODEL.FPN.USE_GN = False
_C.MODEL.FPN.USE_RELU = False


# ---------------------------------------------------------------------------- #
# Group Norm options
# ---------------------------------------------------------------------------- #
_C.MODEL.GROUP_NORM = CN()
# Number of dimensions per group in GroupNorm (-1 if using NUM_GROUPS)
_C.MODEL.GROUP_NORM.DIM_PER_GP = -1
# Number of groups in GroupNorm (-1 if using DIM_PER_GP)
_C.MODEL.GROUP_NORM.NUM_GROUPS = 32
# GroupNorm's small constant in the denominator
_C.MODEL.GROUP_NORM.EPSILON = 1e-5


# ---------------------------------------------------------------------------- #
# RPN options
# ---------------------------------------------------------------------------- #
_C.MODEL.RPN = CN()
_C.MODEL.RPN.USE_FPN = False
# Base RPN anchor sizes given in absolute pixels w.r.t. the scaled network input
_C.MODEL.RPN.ANCHOR_SIZES = (32, 64, 128, 256, 512)
# Stride of the feature map that RPN is attached.
# For FPN, number of strides should match number of scales
_C.MODEL.RPN.ANCHOR_STRIDE = (16,)
# RPN anchor aspect ratios
_C.MODEL.RPN.ASPECT_RATIOS = (0.5, 1.0, 2.0)
# Remove RPN anchors that go outside the image by RPN_STRADDLE_THRESH pixels
# Set to -1 or a large value, e.g. 100000, to disable pruning anchors
_C.MODEL.RPN.STRADDLE_THRESH = 0
# Minimum overlap required between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a positive example (IoU >= FG_IOU_THRESHOLD
# ==> positive RPN example)
_C.MODEL.RPN.FG_IOU_THRESHOLD = 0.7
# Maximum overlap allowed between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a negative examples (IoU < BG_IOU_THRESHOLD
# ==> negative RPN example)
_C.MODEL.RPN.BG_IOU_THRESHOLD = 0.3
# Total number of RPN examples per image
_C.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
# Target fraction of foreground (positive) examples per RPN minibatch
_C.MODEL.RPN.POSITIVE_FRACTION = 0.5
# Number of top scoring RPN proposals to keep before applying NMS
# When FPN is used, this is *per FPN level* (not total)
_C.MODEL.RPN.PRE_NMS_TOP_N_TRAIN = 12000
_C.MODEL.RPN.PRE_NMS_TOP_N_TEST = 6000
# Number of top scoring RPN proposals to keep after applying NMS
_C.MODEL.RPN.POST_NMS_TOP_N_TRAIN = 2000
_C.MODEL.RPN.POST_NMS_TOP_N_TEST = 1000
# NMS threshold used on RPN proposals
_C.MODEL.RPN.NMS_THRESH = 0.7
# Proposal height and width both need to be greater than RPN_MIN_SIZE
# (a the scale used during training or inference)
_C.MODEL.RPN.MIN_SIZE = 0
# Number of top scoring RPN proposals to keep after combining proposals from
# all FPN levels
_C.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN = 2000
_C.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST = 2000
# Custom rpn head, empty to use default conv or separable conv
_C.MODEL.RPN.RPN_HEAD = "SingleConvRPNHead"


# ---------------------------------------------------------------------------- #
# ROI HEADS options
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_HEADS = CN()
_C.MODEL.ROI_HEADS.USE_FPN = False
# Overlap threshold for an RoI to be considered foreground (if >= FG_IOU_THRESHOLD)
_C.MODEL.ROI_HEADS.FG_IOU_THRESHOLD = 0.5
# Overlap threshold for an RoI to be considered background
# (class = 0 if overlap in [0, BG_IOU_THRESHOLD))
_C.MODEL.ROI_HEADS.BG_IOU_THRESHOLD = 0.5
# Default weights on (dx, dy, dw, dh) for normalizing bbox regression targets
# These are empirically chosen to approximately lead to unit variance targets
_C.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS = (10., 10., 5., 5.)
# RoI minibatch size *per image* (number of regions of interest [ROIs])
# Total number of RoIs per training minibatch =
#   TRAIN.BATCH_SIZE_PER_IM * TRAIN.IMS_PER_BATCH
# E.g., a common configuration is: 512 * 2 * 8 = 8192
_C.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
# Target fraction of RoI minibatch that is labeled foreground (i.e. class > 0)
_C.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25

# Only used on test mode

# Minimum score threshold (assuming scores in a [0, 1] range); a value chosen to
# balance obtaining high recall with not having too many low precision
# detections that will slow down inference post processing steps (like NMS)
_C.MODEL.ROI_HEADS.SCORE_THRESH = 0.05
# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
_C.MODEL.ROI_HEADS.NMS = 0.5
# Maximum number of detections to return per image (100 is based on the limit
# established for the COCO dataset)
_C.MODEL.ROI_HEADS.DETECTIONS_PER_IMG = 100


_C.MODEL.ROI_BOX_HEAD = CN()
_C.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR = "ResNet50Conv5ROIFeatureExtractor"
_C.MODEL.ROI_BOX_HEAD.PREDICTOR = "FastRCNNPredictor"
_C.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 0
_C.MODEL.ROI_BOX_HEAD.POOLER_SCALES = (1.0 / 16,)
_C.MODEL.ROI_BOX_HEAD.NUM_CLASSES = 81
# Hidden layer dimension when using an MLP for the RoI box head
_C.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM = 1024
# GN
_C.MODEL.ROI_BOX_HEAD.USE_GN = False
# Dilation
_C.MODEL.ROI_BOX_HEAD.DILATION = 1
_C.MODEL.ROI_BOX_HEAD.CONV_HEAD_DIM = 256
_C.MODEL.ROI_BOX_HEAD.NUM_STACKED_CONVS = 4


_C.MODEL.ROI_MASK_HEAD = CN()
_C.MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR = "ResNet50Conv5ROIFeatureExtractor"
_C.MODEL.ROI_MASK_HEAD.PREDICTOR = "MaskRCNNC4Predictor"
_C.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO = 0
_C.MODEL.ROI_MASK_HEAD.POOLER_SCALES = (1.0 / 16,)
_C.MODEL.ROI_MASK_HEAD.MLP_HEAD_DIM = 1024
_C.MODEL.ROI_MASK_HEAD.CONV_LAYERS = (256, 256, 256, 256)
_C.MODEL.ROI_MASK_HEAD.RESOLUTION = 14
_C.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR = True
# Whether or not resize and translate masks to the input image.
_C.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS = False
_C.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS_THRESHOLD = 0.5
# Dilation
_C.MODEL.ROI_MASK_HEAD.DILATION = 1
# GN
_C.MODEL.ROI_MASK_HEAD.USE_GN = False

_C.MODEL.ROI_KEYPOINT_HEAD = CN()
_C.MODEL.ROI_KEYPOINT_HEAD.FEATURE_EXTRACTOR = "KeypointRCNNFeatureExtractor"
_C.MODEL.ROI_KEYPOINT_HEAD.PREDICTOR = "KeypointRCNNPredictor"
_C.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO = 0
_C.MODEL.ROI_KEYPOINT_HEAD.POOLER_SCALES = (1.0 / 16,)
_C.MODEL.ROI_KEYPOINT_HEAD.MLP_HEAD_DIM = 1024
_C.MODEL.ROI_KEYPOINT_HEAD.CONV_LAYERS = tuple(512 for _ in range(8))
_C.MODEL.ROI_KEYPOINT_HEAD.RESOLUTION = 14
_C.MODEL.ROI_KEYPOINT_HEAD.NUM_CLASSES = 17
_C.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR = True

# ---------------------------------------------------------------------------- #
# ResNe[X]t options (ResNets = {ResNet, ResNeXt}
# Note that parts of a resnet may be used for both the backbone and the head
# These options apply to both
# ---------------------------------------------------------------------------- #
_C.MODEL.RESNETS = CN()

# Number of groups to use; 1 ==> ResNet; > 1 ==> ResNeXt
_C.MODEL.RESNETS.NUM_GROUPS = 1

# Baseline width of each group
_C.MODEL.RESNETS.WIDTH_PER_GROUP = 64

# Place the stride 2 conv on the 1x1 filter
# Use True only for the original MSRA ResNet; use False for C2 and Torch models
_C.MODEL.RESNETS.STRIDE_IN_1X1 = True

# Residual transformation function
_C.MODEL.RESNETS.TRANS_FUNC = "BottleneckWithFixedBatchNorm"
# ResNet's stem function (conv1 and pool1)
_C.MODEL.RESNETS.STEM_FUNC = "StemWithFixedBatchNorm"

# Apply dilation in stage "res5"
_C.MODEL.RESNETS.RES5_DILATION = 1

_C.MODEL.RESNETS.BACKBONE_OUT_CHANNELS = 256 * 4
_C.MODEL.RESNETS.RES2_OUT_CHANNELS = 256
_C.MODEL.RESNETS.STEM_OUT_CHANNELS = 64
# ---------------------------------------------------------------------------- #
# ATSS Options
# ---------------------------------------------------------------------------- #
_C.MODEL.ATSS = CN()
_C.MODEL.ATSS.NUM_CLASSES = 81  # the number of classes including background

# Anchor parameter
_C.MODEL.ATSS.ANCHOR_SIZES = (64, 128, 256, 512, 1024)
_C.MODEL.ATSS.ASPECT_RATIOS = (1.0,)
_C.MODEL.ATSS.ANCHOR_STRIDES = (8, 16, 32, 64, 128)
_C.MODEL.ATSS.STRADDLE_THRESH = 0
_C.MODEL.ATSS.OCTAVE = 2.0
_C.MODEL.ATSS.SCALES_PER_OCTAVE = 1

# Head parameter
_C.MODEL.ATSS.NUM_CONVS = 4
_C.MODEL.ATSS.USE_DCN_IN_TOWER = False

# Focal loss parameter
_C.MODEL.ATSS.LOSS_ALPHA = 0.25
_C.MODEL.ATSS.LOSS_GAMMA = 2.0

# how to select positves: ATSS (Ours) , SSC (FCOS), IoU (RetinaNet), TOPK
_C.MODEL.ATSS.POSITIVE_TYPE = 'ATSS'

# IoU parameter to select positves
_C.MODEL.ATSS.FG_IOU_THRESHOLD = 0.5
_C.MODEL.ATSS.BG_IOU_THRESHOLD = 0.4

# topk for selecting candidate positive samples from each level
_C.MODEL.ATSS.TOPK = 9

# regressing from a box ('BOX') or a point ('POINT')
_C.MODEL.ATSS.REGRESSION_TYPE = 'BOX'

# Weight for bbox_regression loss
_C.MODEL.ATSS.REG_LOSS_WEIGHT = 2.0

# Inference parameter
_C.MODEL.ATSS.PRIOR_PROB = 0.01
_C.MODEL.ATSS.INFERENCE_TH = 0.05
_C.MODEL.ATSS.NMS_TH = 0.6
_C.MODEL.ATSS.PRE_NMS_TOP_N = 1000


# Focal loss parameter
_C.MODEL.ATSS.LOSS_ALPHA = 0.25
_C.MODEL.ATSS.LOSS_GAMMA = 5.0
# ---------------------------------------------------------------------------- #
# FCOS Options
# ---------------------------------------------------------------------------- #
_C.MODEL.FCOS = CN()
_C.MODEL.FCOS.NUM_CLASSES = 81  # the number of classes including background
_C.MODEL.FCOS.FPN_STRIDES = [8, 16, 32, 64, 128]
_C.MODEL.FCOS.PRIOR_PROB = 0.01
_C.MODEL.FCOS.INFERENCE_TH = 0.05
_C.MODEL.FCOS.NMS_TH = 0.6
_C.MODEL.FCOS.PRE_NMS_TOP_N = 1000

# Focal loss parameter: alpha
_C.MODEL.FCOS.LOSS_ALPHA = 0.25
# Focal loss parameter: gamma
_C.MODEL.FCOS.LOSS_GAMMA = 2.0

# the number of convolutions used in the cls and bbox tower
_C.MODEL.FCOS.NUM_CONVS = 4

# ---------------------------------------------------------------------------- #
# Domain Adaption Options
# ---------------------------------------------------------------------------- #
_C.MODEL.ADV = CN()

# discriminator of different feature layers
_C.MODEL.ADV.USE_DIS_P7 = False
_C.MODEL.ADV.USE_DIS_P6 = False
_C.MODEL.ADV.USE_DIS_P5 = False
_C.MODEL.ADV.USE_DIS_P4 = False
_C.MODEL.ADV.USE_DIS_P3 = False

# global and center-aware alignment
_C.MODEL.ADV.USE_DIS_GLOBAL = False
_C.MODEL.ADV.USE_DIS_CENTER_AWARE = False
_C.MODEL.ADV.CENTER_AWARE_WEIGHT = 20
_C.MODEL.ADV.CENTER_AWARE_TYPE = "ca_feature"
_C.MODEL.ADV.GA_DIS_LAMBDA = 0.01
_C.MODEL.ADV.CA_DIS_LAMBDA = 0.1
_C.MODEL.ADV.GRL_APPLIED_DOMAIN = "both"

# the number of convolutions used in the discriminator
_C.MODEL.ADV.DIS_P7_NUM_CONVS = 4
_C.MODEL.ADV.DIS_P6_NUM_CONVS = 4
_C.MODEL.ADV.DIS_P5_NUM_CONVS = 4
_C.MODEL.ADV.DIS_P4_NUM_CONVS = 4
_C.MODEL.ADV.DIS_P3_NUM_CONVS = 4

# the number of convolutions used in the center-aware discriminator
_C.MODEL.ADV.CA_DIS_P7_NUM_CONVS = 4
_C.MODEL.ADV.CA_DIS_P6_NUM_CONVS = 4
_C.MODEL.ADV.CA_DIS_P5_NUM_CONVS = 4
_C.MODEL.ADV.CA_DIS_P4_NUM_CONVS = 4
_C.MODEL.ADV.CA_DIS_P3_NUM_CONVS = 4

# adversarial parameter: GRL weight for global discriminator
_C.MODEL.ADV.GRL_WEIGHT_P7 = 0.1
_C.MODEL.ADV.GRL_WEIGHT_P6 = 0.1
_C.MODEL.ADV.GRL_WEIGHT_P5 = 0.1
_C.MODEL.ADV.GRL_WEIGHT_P4 = 0.1
_C.MODEL.ADV.GRL_WEIGHT_P3 = 0.1

# adversarial parameter: GRL weight for center-aware discriminator
_C.MODEL.ADV.CA_GRL_WEIGHT_P7 = 0.1
_C.MODEL.ADV.CA_GRL_WEIGHT_P6 = 0.1
_C.MODEL.ADV.CA_GRL_WEIGHT_P5 = 0.1
_C.MODEL.ADV.CA_GRL_WEIGHT_P4 = 0.1
_C.MODEL.ADV.CA_GRL_WEIGHT_P3 = 0.1

# OUTPUT DA
_C.MODEL.ADV.USE_DIS_OUT = False
_C.MODEL.ADV.BASE_DIS_TOWER = False # True refers to the 3*3 convs
_C.MODEL.ADV.OUT_DIS_LAMBDA = 0.1
_C.MODEL.ADV.OUT_WEIGHT = 0.5
_C.MODEL.ADV.OUT_LOSS = 'ce'
_C.MODEL.ADV.OUTMAP_OP = 'sigmoid'
_C.MODEL.ADV.OUTPUT_REG_DA = True
_C.MODEL.ADV.OUTPUT_CLS_DA = True
_C.MODEL.ADV.OUTPUT_CENTERNESS_DA = True
# ---------------------------------------------------------------------------- #
# RetinaNet Options (Follow the Detectron version)
# ---------------------------------------------------------------------------- #
_C.MODEL.RETINANET = CN()

# This is the number of foreground classes and background.
_C.MODEL.RETINANET.NUM_CLASSES = 81

# Anchor aspect ratios to use
_C.MODEL.RETINANET.ANCHOR_SIZES = (32, 64, 128, 256, 512)
_C.MODEL.RETINANET.ASPECT_RATIOS = (0.5, 1.0, 2.0)
_C.MODEL.RETINANET.ANCHOR_STRIDES = (8, 16, 32, 64, 128)
_C.MODEL.RETINANET.STRADDLE_THRESH = 0

# Anchor scales per octave
_C.MODEL.RETINANET.OCTAVE = 2.0
_C.MODEL.RETINANET.SCALES_PER_OCTAVE = 3

# Use C5 or P5 to generate P6
_C.MODEL.RETINANET.USE_C5 = True

# Convolutions to use in the cls and bbox tower
# NOTE: this doesn't include the last conv for logits
_C.MODEL.RETINANET.NUM_CONVS = 4

# Weight for bbox_regression loss
_C.MODEL.RETINANET.BBOX_REG_WEIGHT = 4.0

# Smooth L1 loss beta for bbox regression
_C.MODEL.RETINANET.BBOX_REG_BETA = 0.11

# During inference, #locs to select based on cls score before NMS is performed
# per FPN level
_C.MODEL.RETINANET.PRE_NMS_TOP_N = 1000

# IoU overlap ratio for labeling an anchor as positive
# Anchors with >= iou overlap are labeled positive
_C.MODEL.RETINANET.FG_IOU_THRESHOLD = 0.5

# IoU overlap ratio for labeling an anchor as negative
# Anchors with < iou overlap are labeled negative
_C.MODEL.RETINANET.BG_IOU_THRESHOLD = 0.4

# Focal loss parameter: alpha
_C.MODEL.RETINANET.LOSS_ALPHA = 0.25

# Focal loss parameter: gamma
_C.MODEL.RETINANET.LOSS_GAMMA = 2.0

# Prior prob for the positives at the beginning of training. This is used to set
# the bias init for the logits layer
_C.MODEL.RETINANET.PRIOR_PROB = 0.01

# Inference cls score threshold, anchors with score > INFERENCE_TH are
# considered for inference
_C.MODEL.RETINANET.INFERENCE_TH = 0.05

# NMS threshold used in RetinaNet
_C.MODEL.RETINANET.NMS_TH = 0.4


# ---------------------------------------------------------------------------- #
# FBNet options
# ---------------------------------------------------------------------------- #
_C.MODEL.FBNET = CN()
_C.MODEL.FBNET.ARCH = "default"
# custom arch
_C.MODEL.FBNET.ARCH_DEF = ""
_C.MODEL.FBNET.BN_TYPE = "bn"
_C.MODEL.FBNET.SCALE_FACTOR = 1.0
# the output channels will be divisible by WIDTH_DIVISOR
_C.MODEL.FBNET.WIDTH_DIVISOR = 1
_C.MODEL.FBNET.DW_CONV_SKIP_BN = True
_C.MODEL.FBNET.DW_CONV_SKIP_RELU = True

# > 0 scale, == 0 skip, < 0 same dimension
_C.MODEL.FBNET.DET_HEAD_LAST_SCALE = 1.0
_C.MODEL.FBNET.DET_HEAD_BLOCKS = []
# overwrite the stride for the head, 0 to use original value
_C.MODEL.FBNET.DET_HEAD_STRIDE = 0

# > 0 scale, == 0 skip, < 0 same dimension
_C.MODEL.FBNET.KPTS_HEAD_LAST_SCALE = 0.0
_C.MODEL.FBNET.KPTS_HEAD_BLOCKS = []
# overwrite the stride for the head, 0 to use original value
_C.MODEL.FBNET.KPTS_HEAD_STRIDE = 0

# > 0 scale, == 0 skip, < 0 same dimension
_C.MODEL.FBNET.MASK_HEAD_LAST_SCALE = 0.0
_C.MODEL.FBNET.MASK_HEAD_BLOCKS = []
# overwrite the stride for the head, 0 to use original value
_C.MODEL.FBNET.MASK_HEAD_STRIDE = 0

# 0 to use all blocks defined in arch_def
_C.MODEL.FBNET.RPN_HEAD_BLOCKS = 0
_C.MODEL.FBNET.RPN_BN_TYPE = ""


# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.MAX_ITER = 40000

# _C.SOLVER.BASE_LR = 0.001
# _C.SOLVER.BIAS_LR_FACTOR = 2

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0


# _C.SOLVER.GAMMA = 0.1
# _C.SOLVER.STEPS = (30000,)

# _C.SOLVER.WARMUP_FACTOR = 1.0 / 3
# _C.SOLVER.WARMUP_ITERS = 500
# _C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.CHECKPOINT_PERIOD = 2500

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 16

# Backbone
_C.SOLVER.BACKBONE = CN()
_C.SOLVER.BACKBONE.BASE_LR = 0.005
_C.SOLVER.BACKBONE.BIAS_LR_FACTOR = 2
_C.SOLVER.BACKBONE.GAMMA = 0.1
_C.SOLVER.BACKBONE.STEPS = (30000,)
_C.SOLVER.BACKBONE.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.BACKBONE.WARMUP_ITERS = 500
_C.SOLVER.BACKBONE.WARMUP_METHOD = "linear"
# FCOS
_C.SOLVER.FCOS = CN()
_C.SOLVER.FCOS.BASE_LR = 0.005
_C.SOLVER.FCOS.BIAS_LR_FACTOR = 2
_C.SOLVER.FCOS.GAMMA = 0.1
_C.SOLVER.FCOS.STEPS = (30000,)
_C.SOLVER.FCOS.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.FCOS.WARMUP_ITERS = 500
_C.SOLVER.FCOS.WARMUP_METHOD = "linear"
# Discriminator
_C.SOLVER.DIS = CN()
_C.SOLVER.DIS.BASE_LR = 0.005
_C.SOLVER.DIS.BIAS_LR_FACTOR = 2
_C.SOLVER.DIS.GAMMA = 0.1
_C.SOLVER.DIS.STEPS = (30000,)
_C.SOLVER.DIS.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.DIS.WARMUP_ITERS = 500
_C.SOLVER.DIS.WARMUP_METHOD = "linear"
# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()

_C.TEST.EXPECTED_RESULTS = []
_C.TEST.EXPECTED_RESULTS_SIGMA_TOL = 4
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST.IMS_PER_BATCH = 4
# Number of detections per image
_C.TEST.DETECTIONS_PER_IMG = 100
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

_C.SOLVER.ADAPT_VAL_ON = True
_C.SOLVER.VAL_ITER = 100
_C.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")
_C.TENSORBOARD_EXPERIMENT= "./exps/demo/logs/"
_C.CLS_MAP_PRE = 'softmax'

_C.MODEL.ADV.CON_DIS_LAMBDA = 0.1

_C.MODEL.ADV.USE_DIS_P7_CON = False
_C.MODEL.ADV.USE_DIS_P6_CON = False
_C.MODEL.ADV.USE_DIS_P5_CON = False
_C.MODEL.ADV.USE_DIS_P4_CON = False
_C.MODEL.ADV.USE_DIS_P3_CON = False
_C.MODEL.ADV.PATCH_STRIDE = None

_C.SOLVER.BACKBONE.SWA = False
_C.SOLVER.BACKBONE.SWA = False




# CLASS CONDITIONED ALIGNMENT

_C.MODEL.ADV.USE_DIS_CON = False
_C.MODEL.ADV.CON_NUM_SHARED_CONV_P7 = 4
_C.MODEL.ADV.CON_NUM_SHARED_CONV_P6 = 4
_C.MODEL.ADV.CON_NUM_SHARED_CONV_P5 = 4
_C.MODEL.ADV.CON_NUM_SHARED_CONV_P4 = 4
_C.MODEL.ADV.CON_NUM_SHARED_CONV_P3 = 4

_C.MODEL.ADV.CON_WITH_GA = False

# 'concat', 'mul_detached', 'mul
_C.MODEL.ADV.CON_FUSUIN_CFG = 'concat'


_C.MODEL.MIDDLE_HEAD = CN()
_C.MODEL.MIDDLE_HEAD.CONDGRAPH_ON = False
_C.MODEL.MIDDLE_HEAD.NUM_CONVS_IN =2
_C.MODEL.MIDDLE_HEAD.NUM_CONVS_OUT =1
_C.MODEL.MIDDLE_HEAD.GCN1_OUT_CHANNEL = 256
_C.MODEL.MIDDLE_HEAD.GCN2_OUT_CHANNEL = 256
_C.MODEL.MIDDLE_HEAD.GCN_EDGE_PROJECT = 128


# GCN configs

# 'softmax', 'l2', 'gcn'
_C.MODEL.MIDDLE_HEAD.GCN_EDGE_NORM = 'softmax'
# 'relu' 'softmax'
_C.MODEL.MIDDLE_HEAD.GCN_OUT_ACTIVATION = 'relu'

# _C.MODEL.GCN.INCHANNEL = 256
# _C.MODEL.GCN.OUTCHANNEL = 128

_C.MODEL.MIDDLE_HEAD.CAT_ACT_MAP = True
_C.MODEL.MIDDLE_HEAD.GCN_SHORTCUT = False

_C.MODEL.MIDDLE_HEAD.RETURN_ACT_LOGITS = False

_C.MODEL.MIDDLE_HEAD.COND_WITH_BIAS = False
_C.MODEL.MIDDLE_HEAD.PROTO_WITH_BG = True


_C.MODEL.MIDDLE_HEAD.ACT_LOSS = None
_C.MODEL.MIDDLE_HEAD.ACT_LOSS_WEIGHT = 1.0
_C.MODEL.MIDDLE_HEAD.GCN_LOSS_WEIGHT = 1.0
_C.MODEL.MIDDLE_HEAD.CON_LOSS_WEIGHT = 1.0
_C.MODEL.MIDDLE_HEAD.GCN_LOSS_WEIGHT_TG = 1.0

_C.MODEL.MIDDLE_HEAD.PROTO_MOMENTUM = 0.95
_C.MODEL.MIDDLE_HEAD.PROTO_CHANNEL = 256
# _C.MODEL.MIDDLE_HEAD.DUAL_GRAPH_ON = False

# validation configsF
_C.SOLVER.INITIAL_AP50 = 10
# 'AP', 'AP50', 'AP75'
_C.SOLVER.VAL_TYPE = 'AP50'
_C.SOLVER.VAL_ITER = 250
_C.OUTPUT_DIR = "./experiments/debug/"
_C.MODEL.FCOS.REG_CTR_ON = False


_C.SOLVER.MIDDLE_HEAD = CN()
_C.MODEL.MIDDLE_HEAD.GM = CN()
# _C.DEBUG = CN()
# _C.DEBUG.ACT_MAP = False
# _C.DEBUG.TSNE = False
# _C.DEBUG.CNT = False
# _C.DEBUG.CLS = False



_C.SOLVER.MIDDLE_HEAD.BASE_LR = 0.005
_C.SOLVER.MIDDLE_HEAD.BIAS_LR_FACTOR = 2
_C.SOLVER.MIDDLE_HEAD.GAMMA = 0.1
_C.SOLVER.MIDDLE_HEAD.STEPS = (30000,)
_C.SOLVER.MIDDLE_HEAD.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.MIDDLE_HEAD.WARMUP_ITERS = 500
_C.SOLVER.MIDDLE_HEAD.WARMUP_METHOD = "linear"
_C.SOLVER.MIDDLE_HEAD.PLABEL_TH = (0.9,)

_C.MODEL.MIDDLE_HEAD.CON_TG_CFG = 'KLdiv'


_C.MODEL.BACKBONE.VGG_W_BN = False
_C.MODEL.FCOS.NUM_CONVS_REG = 4
_C.MODEL.FCOS.NUM_CONVS_CLS = 4

_C.TEST.MODE ='common'

_C.MODEL.DEBUG_CFG = None
# _C.MODEL.MIDDLE_HEAD.TRANSFER_CFG = (None,)
# _C.MODEL.MIDDLE_HEAD.PROTO_MEAN_VAR = False
_C.MODEL.MIDDLE_HEAD.IN_NORM = 'GN'
_C.MODEL.MIDDLE_HEAD.GLOBAL_GRAPH_ON = False

_C.MODEL.MIDDLE_HEAD.GM_ON = False
_C.MODEL.MIDDLE_HEAD_CFG = None

_C.MODEL.MIDDLE_HEAD.GM.MATCHING_LOSS_CFG = 'MSE'
_C.MODEL.MIDDLE_HEAD.GM.MATCHING_CFG = 'o2o'
_C.MODEL.MIDDLE_HEAD.GM.NUM_NODES_PER_LVL_SR = 100
_C.MODEL.MIDDLE_HEAD.GM.NUM_NODES_PER_LVL_TG = 100
_C.MODEL.MIDDLE_HEAD.GM.BG_RATIO = 8


_C.MODEL.MIDDLE_HEAD.GM.WITH_QUADRATIC_MATCHING = False
_C.MODEL.MIDDLE_HEAD.GM.WITH_CLUSTER_UPDATE = False
_C.MODEL.MIDDLE_HEAD.GM.WITH_SEMANTIC_COMPLETION = False
_C.MODEL.MIDDLE_HEAD.GM.WITH_COMPLETE_GRAPH = False
_C.MODEL.MIDDLE_HEAD.GM.WITH_DOMAIN_INTERACTION = True
_C.MODEL.MIDDLE_HEAD.GM.WITH_COND_CLS = False
_C.MODEL.MIDDLE_HEAD.GM.WITH_CTR = False

# loss weight
_C.MODEL.MIDDLE_HEAD.GM.NODE_LOSS_WEIGHT = 0.1
_C.MODEL.MIDDLE_HEAD.GM.MATCHING_LOSS_WEIGHT = 0.1
_C.MODEL.MIDDLE_HEAD.GM.WITH_SCORE_WEIGHT = True
_C.MODEL.MIDDLE_HEAD.GM.WITH_NODE_DIS = False
_C.MODEL.MIDDLE_HEAD.GM.WITH_GLOBAL_GRAPH = False
_C.MODEL.MIDDLE_HEAD.GM.NODE_DIS_LAMBDA = 0.01
_C.MODEL.MIDDLE_HEAD.GM.NODE_DIS_WEIGHT = 0.1
_C.MODEL.MIDDLE_HEAD.GM.NODE_DIS_PLACE = 'feat' # 'intra', 'inter'
_C.MODEL.MIDDLE_HEAD.GM.MIN_AP50_SAVE = 40 # 'intra', 'inter'




# _C.MODEL.MIDDLE_HEAD.GLOBAL_GCN = False
# _C.MODEL.MIDDLE_HEAD.COSINE_UPDATE_ON = False
# _C.MODEL.MIDDLE_HEAD.PROTO_ALIGN = False
# _C.MODEL.MIDDLE_HEAD.PROTO_ITER = 1

# _C.MODEL.MIDDLE_HEAD.USE_RNN = None
# _C.MODEL.MIDDLE_HEAD.GCN_SELF_TRAINING = False
# _C.MODEL.MIDDLE_HEAD.COND_HIDDEN_CHANNEL = 512
# _C.MODEL.MIDDLE_HEAD.TARGET_SAMPLING_CFG = 'score_threshold'

# _C.MODEL.MIDDLE_HEAD.DBSCAN_EPS = 3
# _C.MODEL.MIDDLE_HEAD.DBSCAN_THR = 0.05
