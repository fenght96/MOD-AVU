from ..builder import DETECTORS
from .two_stream_single_stage import TwoStreamSingleStageDetector

@DETECTORS.register_module()
class TwoStreamFCOS(TwoStreamSingleStageDetector):
    """Implementation of `FCOS <https://arxiv.org/abs/1904.01355>`_"""

    def __init__(self,
                 backbone_1,
                 backbone_2,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(TwoStreamFCOS, self).__init__(backbone_1, backbone_2, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained)
