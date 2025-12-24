from mmcv.utils import Registry

from mmdet.models.builder import BACKBONES, HEADS, LOSSES, NECKS, DETECTORS
import warnings


FUSIONMODELS = Registry("fusion_models")
VTRANSFORMS = Registry("vtransforms")
FUSERS = Registry("fusers")
CUSTOMS = Registry("custom")
FUSION_LAYERS = Registry("fusion_layers")

def build_detector(cfg, train_cfg=None, test_cfg=None):
    """Build detector."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    return DETECTORS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))


def build_backbone(cfg):
    return BACKBONES.build(cfg)


def build_neck(cfg):
    return NECKS.build(cfg)


def build_vtransform(cfg):
    return VTRANSFORMS.build(cfg)


def build_fuser(cfg):
    return FUSERS.build(cfg)


def build_custom(cfg):
    return CUSTOMS.build(cfg)


def build_head(cfg):
    return HEADS.build(cfg)


def build_loss(cfg):
    return LOSSES.build(cfg)


def build_fusion_model(cfg, train_cfg=None, test_cfg=None):
    return FUSIONMODELS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg)
    )

def build_fusion_layer(cfg):
    """Build fusion layer."""
    return FUSION_LAYERS.build(cfg)

def build_model(cfg, train_cfg=None, test_cfg=None):
    return build_fusion_model(cfg, train_cfg=train_cfg, test_cfg=test_cfg)
