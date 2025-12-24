# ---------------------------------------------
#  Modified by Qihang Ma
# ---------------------------------------------

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner.base_module import BaseModule

from mmdet.models.utils.builder import TRANSFORMER
from torch.nn.init import normal_
from mmcv.runner.base_module import BaseModule
from torchvision.transforms.functional import rotate
from .spatial_cross_attention import MSDeformableAttention3D
from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn import PLUGIN_LAYERS, Conv2d,Conv3d, ConvModule, caffe2_xavier_init

@TRANSFORMER.register_module()
class TransformerMSOcc(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 encoder=None,
                 decoder=None,
                 embed_dims=256,
                 use_can_bus=True,
                 can_bus_norm=True,
                 use_cams_embeds=True,
                 **kwargs):
        super(TransformerMSOcc, self).__init__(**kwargs)
        self.encoder = build_transformer_layer_sequence(encoder)

        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False

        self.use_cams_embeds = use_cams_embeds
        self.init_layers()

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        #yh0923
        # self.level_embeds = nn.Parameter(torch.Tensor(
        #     self.num_feature_levels, self.embed_dims))
        # self.cams_embeds = nn.Parameter(
        #     torch.Tensor(self.num_cams, self.embed_dims))
        self.level_embeds = nn.Parameter(torch.zeros(self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(torch.zeros(self.num_cams, self.embed_dims))


    def init_weights(self):
        """Initialize the transformer weights."""
        # for p in self.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)
        # for m in self.modules():
        #     if isinstance(m, MSDeformableAttention3D):
        #         try:
        #             m.init_weight()
        #         except AttributeError:
        #             m.init_weights()
        # normal_(self.level_embeds)
        # normal_(self.cams_embeds)

        #yh0923
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        # 专门初始化这两个 embedding
        nn.init.normal_(self.level_embeds, mean=0.0, std=0.02)   # 类似 Transformer/BERT 风格
        nn.init.normal_(self.cams_embeds, mean=0.0, std=0.02)

    @auto_fp16(apply_to=('mlvl_feats', 'occ_queries', 'prev_occ', 'occ_pos'))
    def get_bev_features(
            self,
            mlvl_feats,
            occ_queries,
            occ_h,
            occ_w,
            occ_z,
            occ_pos=None,
            prev_occ=None,
            **kwargs):
        """
        obtain occ features.
        """
        bs = mlvl_feats[0].size(0)            #bs是1，  mlvl_feats[0]是torch.Size([1, 6, 256, 16, 44])
        occ_queries = occ_queries.permute(1, 0, 2) # [b, n, c] -> [n, b, c] （适配Transformer输入）torch.Size([40000, 1, 256])
        occ_pos = occ_pos.flatten(2).permute(2, 0, 1) #调整张量维度，匹配Transformer的序列化输入要求（序列长度在前）。

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)  # 添加摄像头位置编码
            feat = feat + self.level_embeds[None,
                                            None, lvl:lvl + 1, :].to(feat.dtype)  # 添加层级编码
            spatial_shapes.append(spatial_shape)  #各层原始空间尺寸
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)  #所有层特征拼接后的序列,沿序列维度拼接  torch.Size([6, 1, 704, 256])
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=occ_pos.device)   # 转为张量
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)   torch.Size([6, 704, 1, 256])

        occ_embed = self.encoder(         #OccEncoder mmdet3d/models/model_utils/occencoder.py
            occ_queries,
            feat_flatten,
            feat_flatten,
            occ_h=occ_h,
            occ_w=occ_w,
            occ_z=occ_z,
            occ_pos=occ_pos,
            spatial_shapes=spatial_shapes,   # 各层空间尺寸
            level_start_index=level_start_index,  # 层级起始索引
            prev_occ=prev_occ,  # 时序,先前的特征
            **kwargs
        )

        return occ_embed    #torch.Size([1, 40000, 256])

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'object_query_embed', 'prev_bev', 'bev_pos'))
    def forward(self,
                mlvl_feats,
                occ_queries,
                occ_h,
                occ_w,
                occ_z,
                occ_pos=None,
                reg_branches=None,
                cls_branches=None,
                prev_occ=None,
                **kwargs):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, num_cams, embed_dims, h, w].
            bev_queries (Tensor): (bev_h*bev_w, c)
            bev_pos (Tensor): (bs, embed_dims, bev_h, bev_w)
            object_query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - bev_embed: BEV features
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """

        occ_embed = self.get_bev_features(
            mlvl_feats,
            occ_queries,
            occ_h,
            occ_w,
            occ_z,
            occ_pos=occ_pos,
            prev_occ=prev_occ,
            **kwargs)  # occ_embed shape: bs, occ_h*bev_w*occ_z, embed_dims

        return occ_embed