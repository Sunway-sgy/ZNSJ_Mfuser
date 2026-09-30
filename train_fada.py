# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp

os.chdir(osp.abspath(osp.dirname(osp.dirname(__file__))))
import sys

sys.path.append(os.curdir)
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner

from mmseg.registry import RUNNERS
from mmseg.models import BaseSegmentor
from mmseg.registry import MODELS
from mmengine.analysis import get_model_complexity_info
from mmengine.analysis.print_helper import _format_size
import torch
import torch.nn as nn
# from torchsummary import summary
# from thop import profile
import rein


def parse_args():
    parser = argparse.ArgumentParser(description="Train a segmentor")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work-dir", help="the dir to save logs and models")
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="resume from the latest checkpoint in the work_dir automatically",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=False,
        help="enable automatic-mixed-precision training",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument("--local_rank", "--local-rank", type=int, default=0)
    parser.add_argument(
        "--profile-flops",
        action="store_true",
        help="profile FLOPs before training",
    )
    parser.add_argument(
        "--flops-only",
        action="store_true",
        help="profile FLOPs and exit without training",
    )
    parser.add_argument(
        "--flops-shape",
        type=int,
        nargs=2,
        default=[512, 512],
        metavar=("H", "W"),
        help="input shape used for FLOPs profiling. Default: 512 512",
    )
    parser.add_argument(
        "--flops-batch-size",
        type=int,
        default=1,
        help="dummy batch size used for FLOPs profiling. Default: 1",
    )
    parser.add_argument(
        "--flops-method",
        choices=["profiler", "mmengine"],
        default="profiler",
        help="FLOPs profiling method. Default: profiler",
    )
    parser.add_argument(
        "--flops-topk",
        type=int,
        default=20,
        help="print top-k ops by FLOPs in profiler mode. Default: 20",
    )
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def _format_count(num):
    units = ["", "K", "M", "G", "T", "P"]
    value = float(num)
    for unit in units:
        if abs(value) < 1000.0:
            return f"{value:.3f} {unit}".strip()
        value /= 1000.0
    return f"{value:.3f} E"


def _make_img_metas(batch_size, height, width):
    return [
        dict(
            ori_shape=(height, width),
            img_shape=(height, width),
            pad_shape=(height, width),
            batch_input_shape=(height, width),
            scale_factor=1.0,
            flip=False,
            flip_direction=None,
        )
        for _ in range(batch_size)
    ]


class FlopsWrapper(nn.Module):

    def __init__(self, model, height, width):
        super().__init__()
        self.model = model.module if hasattr(model, "module") else model
        self.height = height
        self.width = width

    def forward(self, img):
        img_metas = _make_img_metas(img.shape[0], self.height, self.width)
        if hasattr(self.model, "encode_decode"):
            return self.model.encode_decode(img, img_metas)
        try:
            return self.model(img, data_samples=None, mode="tensor")
        except TypeError:
            return self.model(img, img_metas, return_loss=False)


def profile_model_flops(model, args):
    height, width = args.flops_shape
    batch_size = args.flops_batch_size
    device = torch.device("cuda", args.local_rank) \
        if torch.cuda.is_available() else torch.device("cpu")

    was_training = model.training
    model.eval()
    model.to(device)
    wrapper = FlopsWrapper(model, height, width).to(device).eval()

    params = sum(p.numel() for p in wrapper.parameters())
    print_log(
        "Profiling FLOPs with input "
        f"batch={batch_size}, shape=3x{height}x{width}, "
        f"method={args.flops_method}",
        logger="current",
    )
    print_log(f"Params: {_format_count(params)} ({params:,})", logger="current")

    dummy = torch.randn(batch_size, 3, height, width, device=device)

    if args.flops_method == "mmengine":
        info = get_model_complexity_info(
            wrapper,
            inputs=dummy,
            show_table=False,
            show_arch=False,
        )
        flops = info["flops"] if isinstance(info, dict) else info
        print_log(f"FLOPs: {_format_count(flops)}FLOPs", logger="current")
    else:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.no_grad():
            with torch.profiler.profile(
                activities=activities, with_flops=True
            ) as prof:
                wrapper(dummy)
                if device.type == "cuda":
                    torch.cuda.synchronize()

        events = prof.key_averages()
        total_flops = sum(getattr(evt, "flops", 0) or 0 for evt in events)
        print_log(f"FLOPs: {_format_count(total_flops)}FLOPs", logger="current")
        print_log(f"GFLOPs: {total_flops / 1e9:.6f}", logger="current")
        ranked = sorted(
            events, key=lambda evt: getattr(evt, "flops", 0) or 0,
            reverse=True)
        for evt in ranked[:args.flops_topk]:
            flops = getattr(evt, "flops", 0) or 0
            if flops > 0:
                print_log(
                    f"FLOPs op: {evt.key:45s} {flops / 1e9:.6f} GFLOPs",
                    logger="current",
                )

    print_log(
        "Note: FLOPs profilers only count supported ops. "
        "Custom CUDA/deformable-attention kernels may be under-counted.",
        logger="current",
    )
    model.train(was_training)


def main():
    args = parse_args()
    args.profile_flops = args.profile_flops or args.flops_only

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join(
            "./work_dirs", osp.splitext(osp.basename(args.config))[0]
        )

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == "AmpOptimWrapper":
            print_log(
                "AMP training is already enabled in your config.",
                logger="current",
                level=logging.WARNING,
            )
        else:
            assert optim_wrapper == "OptimWrapper", (
                "`--amp` is only supported when the optimizer wrapper type is "
                f"`OptimWrapper` but got {optim_wrapper}."
            )
            cfg.optim_wrapper.type = "AmpOptimWrapper"
            cfg.optim_wrapper.loss_scale = "dynamic"

    # resume training
    cfg.resume = args.resume

    # build the runner from config
    if "runner_type" not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    if args.profile_flops:
        rank = torch.distributed.get_rank() \
            if torch.distributed.is_available() and \
            torch.distributed.is_initialized() else 0
        if rank == 0:
            profile_model_flops(runner.model, args)
        if torch.distributed.is_available() and \
                torch.distributed.is_initialized():
            torch.distributed.barrier()
        if args.flops_only:
            return

    # start training
    runner.train()


if __name__ == "__main__":
    main()
