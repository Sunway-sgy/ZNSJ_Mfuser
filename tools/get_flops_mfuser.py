import argparse
import os
import sys
from contextlib import contextmanager
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "configs"
    / "mfuser"
    / "mfuser_clip_vit-l_1e-4_20k-g2c-512.py"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Profile FLOPs for MFuser.")
    parser.add_argument(
        "config",
        nargs="?",
        default=str(DEFAULT_CONFIG),
        help="Config path. Defaults to mfuser_clip_vit-l_1e-4_20k-g2c-512.",
    )
    parser.add_argument(
        "--shape",
        type=int,
        nargs=2,
        default=[512, 512],
        metavar=("H", "W"),
        help="Input image shape. Default: 512 512.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Dummy batch size. Default: 1.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device used for the dummy forward. This model is written for CUDA.",
    )
    parser.add_argument(
        "--method",
        default="profiler",
        choices=["profiler", "mmcv"],
        help="FLOPs method. profiler counts supported PyTorch ops; mmcv uses module hooks.",
    )
    parser.add_argument(
        "--load-weights",
        action="store_true",
        help="Load pretrained weights. Disabled by default because weights do not affect FLOPs.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=20,
        help="Print top-k ops for profiler mode. Default: 20.",
    )
    return parser.parse_args()


def format_count(num):
    units = ["", "K", "M", "G", "T", "P"]
    value = float(num)
    for unit in units:
        if abs(value) < 1000.0:
            return f"{value:.3f} {unit}".strip()
        value /= 1000.0
    return f"{value:.3f} E"


def format_flops(num):
    return f"{format_count(num)}FLOPs"


@contextmanager
def skip_pretrained_loading(enabled=True):
    """Avoid slow checkpoint loading during architecture-only profiling."""
    if not enabled:
        yield
        return

    import torch
    import models.backbones.clip.models as clip_models
    import models.backbones.dino_v2 as dino_v2

    original_torch_load = torch.load
    original_dino_load_state_dict = dino_v2.DinoVisionTransformer.load_state_dict
    original_clip_vision_init = clip_models.CLIPVisionTransformer.init_weights
    original_clip_text_init = clip_models.CLIPTextEncoder.init_weights
    original_clip_text_context_init = clip_models.CLIPTextContextEncoder.init_weights

    def fast_torch_load(f, *args, **kwargs):
        if os.fspath(f).replace("\\", "/").endswith("pretrained/dinov2_vitl14_pretrain.pth"):
            print("[skip-weights] Skip DINOv2 checkpoint loading for FLOPs.")
            return {}
        return original_torch_load(f, *args, **kwargs)

    def no_load_state_dict(self, state_dict, strict=True):
        return [], []

    def no_init_weights(self, pretrained=None):
        print(f"[skip-weights] Skip {self.__class__.__name__}.init_weights().")

    torch.load = fast_torch_load
    dino_v2.DinoVisionTransformer.load_state_dict = no_load_state_dict
    clip_models.CLIPVisionTransformer.init_weights = no_init_weights
    clip_models.CLIPTextEncoder.init_weights = no_init_weights
    clip_models.CLIPTextContextEncoder.init_weights = no_init_weights
    try:
        yield
    finally:
        torch.load = original_torch_load
        dino_v2.DinoVisionTransformer.load_state_dict = original_dino_load_state_dict
        clip_models.CLIPVisionTransformer.init_weights = original_clip_vision_init
        clip_models.CLIPTextEncoder.init_weights = original_clip_text_init
        clip_models.CLIPTextContextEncoder.init_weights = original_clip_text_context_init


def set_model_class_names(cfg):
    if cfg.model.get("class_names", None) is not None:
        return

    # This config is GTA-to-Cityscapes and uses the Cityscapes 19 classes.
    from mmseg.datasets.cityscapes import CityscapesDataset

    cfg.model.class_names = list(CityscapesDataset.CLASSES)


def build_profile_model(cfg_path, height, width, device, load_weights):
    import torch.nn as nn
    from mmcv import Config

    sys.path.insert(0, str(REPO_ROOT))

    import models  # noqa: F401
    from mmseg.models import build_segmentor

    cfg = Config.fromfile(cfg_path)
    set_model_class_names(cfg)
    cfg.model["save_dir"] = cfg.get("work_dir", "./work_dirs")

    with skip_pretrained_loading(enabled=not load_weights):
        model = build_segmentor(
            cfg.model,
            train_cfg=cfg.get("train_cfg"),
            test_cfg=cfg.get("test_cfg"),
        )

    model.eval()
    model.to(device)

    class FlopsWrapper(nn.Module):
        def __init__(self, segmentor):
            super().__init__()
            self.model = segmentor

        def make_img_metas(self, batch_size):
            return [
                {
                    "ori_shape": (height, width, 3),
                    "img_shape": (height, width, 3),
                    "pad_shape": (height, width, 3),
                    "scale_factor": 1.0,
                    "flip": False,
                    "flip_direction": None,
                }
                for _ in range(batch_size)
            ]

        def forward(self, img):
            return self.model.encode_decode(img, self.make_img_metas(img.shape[0]))

    return FlopsWrapper(model).to(device).eval()


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def profile_with_torch_profiler(model, batch_size, shape, device, topk):
    import torch

    height, width = shape
    dummy = torch.randn(batch_size, 3, height, width, device=device)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.inference_mode():
        with torch.profiler.profile(activities=activities, with_flops=True) as prof:
            model(dummy)
            if device == "cuda":
                torch.cuda.synchronize()

    events = prof.key_averages()
    total_flops = sum(getattr(evt, "flops", 0) or 0 for evt in events)
    print(f"Total FLOPs: {format_flops(total_flops)}")
    print(f"Total GFLOPs: {total_flops / 1e9:.6f}")

    if topk > 0:
        print(f"\nTop {topk} ops by FLOPs:")
        ranked = sorted(events, key=lambda evt: getattr(evt, "flops", 0) or 0, reverse=True)
        for evt in ranked[:topk]:
            flops = getattr(evt, "flops", 0) or 0
            if flops == 0:
                continue
            print(f"{evt.key:45s} {flops / 1e9:12.6f} GFLOPs")


def profile_with_mmcv(model, batch_size, shape, device):
    import torch
    from mmcv.cnn import get_model_complexity_info

    height, width = shape

    def input_constructor(input_shape):
        _, h, w = input_shape
        return {"img": torch.randn(batch_size, 3, h, w, device=device)}

    flops, params = get_model_complexity_info(
        model,
        (3, height, width),
        input_constructor=input_constructor,
        as_strings=True,
        print_per_layer_stat=False,
    )
    print(f"Total FLOPs: {flops}")
    print(f"Total Params: {params}")


def main():
    args = parse_args()
    import torch

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This MFuser config hardcodes CUDA in the model.")

    height, width = args.shape
    device = torch.device(args.device)

    print(f"Config: {cfg_path}")
    print(f"Input: batch={args.batch_size}, shape=3x{height}x{width}, device={device}")
    print(f"Method: {args.method}")
    print(f"Load weights: {args.load_weights}")

    model = build_profile_model(cfg_path, height, width, device, args.load_weights)
    params = count_params(model)
    print(f"Params: {format_count(params)} ({params:,})")

    if args.method == "profiler":
        profile_with_torch_profiler(model, args.batch_size, (height, width), args.device, args.topk)
    else:
        profile_with_mmcv(model, args.batch_size, (height, width), args.device)

    print(
        "\nNote: PyTorch/mmcv FLOPs counters only include supported ops. "
        "Custom CUDA ops such as selective_scan and some deformable-attention kernels may be under-counted."
    )


if __name__ == "__main__":
    main()
