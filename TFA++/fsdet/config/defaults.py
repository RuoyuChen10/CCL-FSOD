from detectron2.config import CfgNode as CN
from detectron2.config.defaults import _C

# adding additional default values built on top of the default values in detectron2

_CC = _C

# FREEZE Parameters
_CC.MODEL.BACKBONE.FREEZE = False
_CC.MODEL.PROPOSAL_GENERATOR.FREEZE = False
_CC.MODEL.ROI_HEADS.FREEZE_FEAT = False

# choose from "FastRCNNOutputLayers" and "CosineSimOutputLayers"
_CC.MODEL.ROI_HEADS.OUTPUT_LAYER = "FastRCNNOutputLayers"
# scale of cosine similarity (set to -1 for learnable scale)
_CC.MODEL.ROI_HEADS.COSINE_SCALE = 20.0

# Backward Compatible options.
_CC.MUTE_HEADER = True

# Relationship MATRIX path
_CC.KNOWLEDGE = False
_CC.KNOWLEDGE_MATRIX = None
_CC.CLUSTER = True
_CC.TAU = 0.2
_CC.CONTRAST_NORM = True
# _CC.KNOWLEDGE = CN()
# _CC.KNOWLEDGE.SKETCH_MATRIX_PATH = None
# _CC.KNOWLEDGE.TEXT_MATRIX_PATH   = None
# _CC.KNOWLEDGE.VISION_MATRIX_PATH = None

# If TSNE?
_CC.TSNE_SAVE_PATH = None

# If prototype
_CC.PROTOTYPE = False

# Which SHOT?
_CC.SHOT = 0

# [linear, exp, no]
_CC.NORM_MATRIX = "no"

_CC.BANK_CONTAINER = 2

# Explanation Operation
_CC.COUNTERFACTUAL = CN()
_CC.COUNTERFACTUAL.OPERATOR = False
_CC.COUNTERFACTUAL.START_ITER = 1000
_CC.COUNTERFACTUAL.ERASE_THRESHOLD = 0.6
_CC.COUNTERFACTUAL.VISUALIZATION = False
_CC.COUNTERFACTUAL.COUNTER_NUMBER = 3
_CC.COUNTERFACTUAL.ERASE_RATE = 0.05
_CC.COUNTERFACTUAL.ERASE_METHOD = "random"

_CC.LAMBDA1 = 1.
_CC.LAMBDA2 = 2.
_CC.LAMBDA3 = 2.
