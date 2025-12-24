from typing import Any, Dict

import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch.nn import functional as F
import einops

from mmdet3d.models.builder import (
    build_detector,
    build_backbone,
    build_neck,
    build_vtransform,
    build_fuser,
    build_custom,
    build_head,
)
from mmdet3d.ops import Voxelization, DynamicScatter
from mmdet3d.models import FUSIONMODELS


from .base import Base3DFusionModel

__all__ = ["BEVFusion"]


@FUSIONMODELS.register_module()
class BEVFusion(Base3DFusionModel):
    def __init__(
        self,
        encoders: Dict[str, Any],
        fuser: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__()

        self.encoders = nn.ModuleDict()  ## 使用ModuleDict管理多模态编码器
        if encoders.get("camera") is not None:
            self.encoders["camera"] = nn.ModuleDict(
                {
                    "detector": build_detector(encoders["camera"]["detector"]),
                    # "img_backbone": build_backbone(encoders["camera"]["detector"]["img_backbone"]),
                    # "img_neck": build_neck(encoders["camera"]["detector"]["img_neck"]),
                    # "img_view_transformer": build_neck(encoders["camera"]["detector"]["img_view_transformer"]),
                    # "img_bev_encoder_backbone":build_backbone(encoders["camera"]["detector"]["img_bev_encoder_backbone"]),
                    # "img_bev_encoder_neck": build_neck(encoders["camera"]["detector"]["img_bev_encoder_neck"])
                    # "pts_bbox_head":build_neck(encoders["camera"]["detector"]["pts_bbox_head"])
                }
            )
            if encoders["camera"].get("img_bev_encoder_backbone") is not None:
                self.encoders["camera"]["img_bev_encoder_backbone"] = build_backbone(encoders["camera"]["detector"]["img_bev_encoder_backbone"])
            if encoders["camera"].get("img_bev_encoder_neck") is not None:
                self.encoders["camera"]["img_bev_encoder_neck"] = build_neck(encoders["camera"]["detector"]["img_bev_encoder_neck"])

        if encoders.get("lidar") is not None:
            if encoders["lidar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders["lidar"]["voxelize"])  # 硬体素化​​: 限制每个体素的最大点数（max_num_points），超出则截断。
            else:
                voxelize_module = DynamicScatter(**encoders["lidar"]["voxelize"])  #软体素化（DynamicScatter）​​: 动态分配点数（无硬性限制）。
            self.encoders["lidar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,  # 体素化模块
                    "backbone": build_backbone(encoders["lidar"]["backbone"]),   # 稀疏卷积主干(SparseEncoder)
                }
            )
            self.voxelize_reduce = encoders["lidar"].get("voxelize_reduce", True)
            if encoders["lidar"].get("bev_backbone") is not None:
                self.encoders["lidar"]["bev_backbone"] = build_backbone(encoders["lidar"]["bev_backbone"])
            if encoders["lidar"].get("bev_neck") is not None:
                self.encoders["lidar"]["bev_neck"] = build_neck(encoders["lidar"]["bev_neck"])

        if fuser is not None:
            self.fuser = build_fuser(fuser)
        else:
            self.fuser = None

        if decoder is not None:
            self.decoder = nn.ModuleDict(
                {
                    "backbone": build_backbone(decoder["backbone"]),
                    "neck": build_neck(decoder["neck"]),
                }
            )
        else:
            self.decoder = None

        self.heads = nn.ModuleDict()
        for name in heads:
            if heads[name] is not None:
                self.heads[name] = build_head(heads[name])

        if "loss_scale" in kwargs:
            self.loss_scale = kwargs["loss_scale"]
        else:
            self.loss_scale = dict()
            for name in heads:
                if heads[name] is not None:
                    self.loss_scale[name] = 1.0
        self.init_weights()

    def init_weights(self) -> None:
        if "camera" in self.encoders:
            det = self.encoders["camera"]["detector"]
            det.img_backbone.init_weights()

    def extract_camera_features(
        self,
        img_inputs,
        img_metas,
        points=None,
        **kwargs,
    ) -> torch.Tensor:
        kwargs.pop('return_loss', None)#0818
        x = self.encoders["camera"]["detector"](img_inputs,img_metas,points,
                      **kwargs)
    

        return x  #shape:torch.Size([1, 4096, 180, 180])  图像BEV特征

    def extract_lidar_features(self, x) -> torch.Tensor:
        feats, coords, sizes = self.voxelize(x)  #体素化
        batch_size = coords[-1, 0] + 1   #得到总批次大小
        x = self.encoders["lidar"]["backbone"](feats, coords, batch_size, sizes=sizes)  #3D稀疏卷积（Backbone）​

        if "bev_backbone" in self.encoders["lidar"]:  # BEV特征提取（可选）​
            x = self.encoders["lidar"]["bev_backbone"](x)
        if "bev_neck" in self.encoders["lidar"]:
            x = self.encoders["lidar"]["bev_neck"](x)

        return x  #torch.Size([1, 256, 180, 180])  点云BEV

    @torch.no_grad() # 禁用梯度计算（推理阶段优化）
    @force_fp32()   # 强制使用 FP32 计算（避免精度损失）
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):   #points是一个 ​​列表​​（List[torch.Tensor]），其中每个元素是一个批次的点云，形状为 [N, C]（N 是点数，C 是特征维度
            ret = self.encoders["lidar"]["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize 硬体素化（保留点数信息）
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret  #仅返回 (feats, coords)，不统计点数。
                n = None  
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))  #在体素坐标 c 的最前面插入批次索引 k，得到 (batch_idx, x, y, z)。
            if n is not None: 
                sizes.append(n)

        feats = torch.cat(feats, dim=0)  # 合并所有体素特征
        coords = torch.cat(coords, dim=0)  # 合并所有体素坐标
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)  # 合并所有体素点数
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )        #对体素内所有点的特征求均值（sum(feats) / num_points）
                feats = feats.contiguous()

        return feats, coords, sizes

    @auto_fp16(apply_to=("img", "points"))
    def forward(
        self,
        img_inputs,
        points,
        lidar2ego,
        lidar_aug_matrix,
        metas,
        img_metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        # if isinstance(img_inputs, list):
        #     raise NotImplementedError
        # else:
        outputs = self.forward_single(   #单张图像输入
            img_inputs,
            points,
            lidar2ego,
            lidar_aug_matrix,
            metas,
            img_metas,
            gt_masks_bev,
            gt_bboxes_3d,
            gt_labels_3d,
            **kwargs,
        )
        return outputs

    @auto_fp16(apply_to=("img", "points"))
    def forward_single(
        self,
        img_inputs,
        points,
        lidar2ego,
        lidar_aug_matrix,
        metas,
        img_metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        features = []
        for sensor in (self.encoders if self.training else list(self.encoders.keys())[::-1]):
            if sensor == "camera":
                feature = self.extract_camera_features(  #提取图像函数
                            img_inputs,
                            img_metas,
                            points,
                            **kwargs,
                )

            elif sensor == "lidar":
                feature = self.extract_lidar_features(points)   #提取点云特征
            else:
                raise ValueError(f"unsupported sensor: {sensor}")
            features.append(feature)   #features里有0是图像，1是点云

        if not self.training:
            # avoid OOM
            features = features[::-1]  #反转顺序（如 [lidar, camera]）

        if self.fuser is not None:
            x = self.fuser(features)  # 多模态特征融合  torch.Size([4, 512, 180, 180])
        else:
            assert len(features) == 1, features
            x = features[0]   # 单模态直接使用

        batch_size = x.shape[0]

        if self.decoder is not None:
            x = self.decoder["backbone"](x)  #x[0].shape是torch.Size([4, 128, 180, 180])   x[1].shape是torch.Size([4, 256, 180, 180])  x[2].shape是torch.Size([4, 512, 90, 90])
            x = self.decoder["neck"](x)

        if self.training:  #训练模式
            outputs = {}
            for head_type, head in self.heads.items():  #head是CenterHead    head_type先是"object"，然后是"occ"
                if head_type == "object":  # 3D检测头  
                    pred_dict = head(x, metas)  
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict) # 计算损失
                elif head_type == "map":  # BEV地图分割头
                    losses = head(x, gt_masks_bev) 
                elif head_type == "occ":  # 占据预测头
                    # TODO: [yz] this is so weird!
                    occ_pred = head(x, lidar_aug_matrix, lidar2ego, kwargs['occ_aug_matrix'])  
                    losses = head.loss(occ_pred, kwargs['voxel_semantics'], kwargs['mask_camera'])
                else:
                    raise ValueError(f"unsupported head: {head_type}")
                for name, val in losses.items():
                    if val.requires_grad:
                        outputs[f"loss/{head_type}/{name}"] = val * self.loss_scale[head_type]
                    else:
                        outputs[f"stats/{head_type}/{name}"] = val
            # if depth_pred is not None and "gt_depth" in kwargs:
            #     loss_depth = self.encoders["camera"]["detector"].img_view_transformer.get_depth_loss(
            #         kwargs["gt_depth"], depth_pred
            #         )
            #     outputs["loss/depth"] = loss_depth*0.1

            return outputs  #字典形式，包含各任务的损失值
        
        else:  #推理模式（self.training == False）
            outputs = [{} for _ in range(batch_size)]
            for head_type, head in self.heads.items():
                if head_type == "object":
                    pred_dict = head(x, metas)
                    bboxes = head.get_bboxes(pred_dict, metas)
                    for k, (boxes, scores, labels) in enumerate(bboxes):
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                            }
                        )
                elif head_type == "map":
                    logits = head(x)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                            }
                        )
                elif head_type == "occ":
                    # TODO: [yz] this is so weird!
                    occ_pred = head(x, lidar_aug_matrix, lidar2ego, kwargs['occ_aug_matrix'])
                    occ_pred = head.get_occ(occ_pred)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "occ_pred": occ_pred,  # already in cpu
                            }
                        )
                else:
                    raise ValueError(f"unsupported head: {head_type}")
            return outputs   #预测结果列表形式，每个元素是样本的预测结果（如 boxes_3d, masks_bev）
