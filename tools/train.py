#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import random
import argparse

import numpy as np
import torch

# 1) 进程就位：按 LOCAL_RANK 绑定本地 GPU（务必最前）
if torch.cuda.is_available():
    _lr = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(_lr)
    print(f"[PID {os.getpid()}] LOCAL_RANK={_lr}, cuda.current_device={torch.cuda.current_device()}",
          flush=True)

# mmcv / mmdet3d
from mmcv import Config
from mmcv.runner import init_dist, get_dist_info
from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger, convert_sync_batchnorm, recursive_eval

# torchpack（仅用于配置解析的“第一优先”）
from torchpack.utils.config import configs as tp_configs


def build_run_dir(default_root: str, cfg_path: str) -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    name = os.path.splitext(os.path.basename(cfg_path))[0]
    run_dir = os.path.join(default_root, name, stamp)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def set_random_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_cfg_compatible(cfg_path: str, extra_opts: list):
    """
    优先用 torchpack 解析（适配 DAOCC 等工程的 YAML）；
    失败则回退到 mmcv Config.fromfile（标准 mmdet3d 配置）。
    """
    # --- 尝试 torchpack 风格 ---
    try:
        tp_configs.load(cfg_path, recursive=True)
        if extra_opts:
            tp_configs.update(extra_opts)  # 形如 ["key1=a", "a.b=3"]
        cfg_dict = recursive_eval(tp_configs)  # 转成普通 dict（含 data 等字段）
        cfg = Config(cfg_dict, filename=cfg_path)
        if hasattr(cfg, "data"):
            return cfg
        # 没有 data 字段，视为不符合预期，转入 mmcv 回退
    except Exception:
        pass

    # --- 回退：mmcv/mmdet3d 标准配置 ---
    cfg = Config.fromfile(cfg_path)
    # 额外覆盖（mmcv 风格的 key=value 不解析为层级的话，也让它透传）
    # 这里不做 merge_from_dict，避免把未知 key 打碎结构，保持简洁。
    return cfg


def main():
    # 2) 参数
    parser = argparse.ArgumentParser(description="mmdet3d training (torchrun + DDP, cfg-compatible)")
    parser.add_argument("config", help="config file path")
    parser.add_argument("--run-dir", type=str, default=None, help="directory to save logs/checkpoints")
    parser.add_argument("--dist", action="store_true", help="enable distributed training")
    # 其余未识别参数（key=value）原样传入 torchpack 的 configs.update
    args, extra_opts = parser.parse_known_args()

    # 3) 读取配置（兼容两种来源）
    cfg = load_cfg_compatible(args.config, extra_opts)

    # cudnn benchmark
    cudnn_benchmark = bool(getattr(cfg, "cudnn_benchmark", False))
    torch.backends.cudnn.benchmark = cudnn_benchmark

    # 4) 分布式初始化（torchrun 注入 env://）
    rank, world_size = 0, 1
    if args.dist:
        init_dist(launcher="pytorch")
        rank, world_size = get_dist_info()

    # 5) run_dir、日志（只在 rank0 IO）
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    if rank == 0:
        run_dir = args.run_dir or build_run_dir(default_root="./work_dirs", cfg_path=args.config)
        os.makedirs(run_dir, exist_ok=True)
        cfg.run_dir = run_dir
        cfg.dump(os.path.join(run_dir, "configs.yaml"))
        log_file = os.path.join(run_dir, f"{timestamp}.log")
        logger = get_root_logger(log_file=log_file)
        logger.info(f"Config merged:\n{cfg.text}")
        logger.info("-" * 66)
        logger.info(f"[CUSTOM] distributed: {'ON' if args.dist else 'OFF'} | "
                    f"rank/world: {rank}/{world_size} | cudnn_benchmark={cudnn_benchmark}")
        logger.info("-" * 66)
    else:
        cfg.run_dir = args.run_dir or "./work_dirs/_tmp"

    # 6) 随机种子
    if getattr(cfg, "seed", None) is not None:
        set_random_seed(cfg.seed, bool(getattr(cfg, "deterministic", False)))

    # 7) 构建数据集 / 模型
    #   这里假定 cfg.data.train 按 mmdet3d 规范存在（torchpack->recursive_eval 已转好）
    datasets = [build_dataset(cfg.data.train)]

    model = build_model(cfg.model)
    model.init_weights() # 初始化权重，顶层调用

    # SyncBN（可选）
    if getattr(cfg, "sync_bn", None):
        sbn_cfg = cfg.sync_bn
        if not isinstance(sbn_cfg, dict):
            sbn_cfg = dict(exclude=[])
        model = convert_sync_batchnorm(model, exclude=sbn_cfg.get("exclude", []))

    # 8) 训练
    train_model(
        model=model,
        dataset=datasets,
        cfg=cfg,
        distributed=args.dist,
        validate=True,
        timestamp=timestamp,
    )


if __name__ == "__main__":
    main()
