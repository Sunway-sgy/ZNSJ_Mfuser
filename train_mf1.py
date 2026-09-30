import os
import os.path as osp

import copy
import time
import mmcv
import torch
import torch.nn as nn
import argparse

from mmcv.runner import init_dist
from mmcv.utils import Config, DictAction, get_git_hash

from mmseg import __version__
from mmseg.models import build_segmentor
from mmseg.datasets import build_dataset
from mmseg.utils import collect_env, get_root_logger
from mmseg.apis import set_random_seed, train_segmentor

import models
from fvcore.nn import FlopCountAnalysis
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--load-from', help='the checkpoint file to load weights from')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=2023, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--finetune', default=False, action='store_true')
    parser.add_argument(
        '--profile-flops',
        action='store_true',
        help='profile FLOPs before training')
    parser.add_argument(
        '--flops-only',
        action='store_true',
        help='profile FLOPs and exit without training')
    parser.add_argument(
        '--flops-shape',
        type=int,
        nargs=2,
        default=[512, 512],
        metavar=('H', 'W'),
        help='input shape used for FLOPs profiling. Default: 512 512')
    parser.add_argument(
        '--flops-batch-size',
        type=int,
        default=1,
        help='dummy batch size used for FLOPs profiling. Default: 1')
    parser.add_argument(
        '--flops-method',
        choices=['profiler', 'mmcv'],
        default='profiler',
        help='FLOPs profiling method. Default: profiler')
    parser.add_argument(
        '--flops-topk',
        type=int,
        default=20,
        help='print top-k ops by FLOPs in profiler mode. Default: 20')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def _format_count(num):
    units = ['', 'K', 'M', 'G', 'T', 'P']
    value = float(num)
    for unit in units:
        if abs(value) < 1000.0:
            return f'{value:.3f} {unit}'.strip()
        value /= 1000.0
    return f'{value:.3f} E'


def _make_img_metas(batch_size, height, width):
    return [
        dict(
            ori_shape=(height, width, 3),
            img_shape=(height, width, 3),
            pad_shape=(height, width, 3),
            scale_factor=1.0,
            flip=False,
            flip_direction=None)
        for _ in range(batch_size)
    ]


class FlopsWrapper(nn.Module):

    def __init__(self, model, height, width):
        super().__init__()
        self.model = model
        self.height = height
        self.width = width

    def forward(self, img):
        return self.model.encode_decode(
            img, _make_img_metas(img.shape[0], self.height, self.width))


def profile_model_flops(model, args, logger):
    height, width = args.flops_shape
    batch_size = args.flops_batch_size

    was_training = model.training
    model.eval()
    model.cuda()
    wrapper = FlopsWrapper(model, height, width).cuda().eval()

    logger.info('Profiling FLOPs with input '
                f'batch={batch_size}, shape=3x{height}x{width}, '
                f'method={args.flops_method}')

    if args.flops_method == 'mmcv':
        from mmcv.cnn import get_model_complexity_info

        def input_constructor(input_shape):
            _, h, w = input_shape
            return dict(
                img=torch.randn(batch_size, 3, h, w).cuda())

        flops, params = get_model_complexity_info(
            wrapper,
            (3, height, width),
            input_constructor=input_constructor,
            as_strings=True,
            print_per_layer_stat=False)
        logger.info(f'FLOPs: {flops}')
        logger.info(f'Params: {params}')
    else:
        dummy = torch.randn(batch_size, 3, height, width).cuda()
        activities = [torch.profiler.ProfilerActivity.CPU]
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.no_grad():
            with torch.profiler.profile(
                    activities=activities, with_flops=True) as prof:
                wrapper(dummy)
                torch.cuda.synchronize()

        events = prof.key_averages()
        total_flops = sum(getattr(evt, 'flops', 0) or 0 for evt in events)
        logger.info(f'FLOPs: {_format_count(total_flops)}FLOPs')
        logger.info(f'GFLOPs: {total_flops / 1e9:.6f}')
        ranked = sorted(
            events, key=lambda evt: getattr(evt, 'flops', 0) or 0,
            reverse=True)
        for evt in ranked[:args.flops_topk]:
            flops = getattr(evt, 'flops', 0) or 0
            if flops > 0:
                logger.info(f'FLOPs op: {evt.key:45s} '
                            f'{flops / 1e9:.6f} GFLOPs')

    logger.info('Note: FLOPs profilers only count supported ops. '
                'selective_scan and some custom CUDA/deformable-attention '
                'kernels may be under-counted.')
    model.train(was_training)


def main():
    args = parse_args()
    args.profile_flops = args.profile_flops or args.flops_only

    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # create work_dir
    if args.work_dir is not None:
        cfg.work_dir = osp.join(args.work_dir,
                                osp.splitext(osp.basename(args.config))[0])
    elif cfg.get('work_dir', None) is not None:
        cfg.work_dir = osp.join(cfg.get('work_dir'),
                                osp.splitext(osp.basename(args.config))[0])
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    if args.load_from is not None:
        cfg.load_from = args.load_from
        
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)    

    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))

    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()

    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic: '
                    f'{args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    # build dataset
    if args.flops_only:
        from mmseg.datasets.cityscapes import CityscapesDataset
        datasets = []
        cfg.model.class_names = list(CityscapesDataset.CLASSES)
    else:
        datasets = [build_dataset(cfg.data.train)]
        cfg.model.class_names = list(datasets[0].CLASSES)

    # build model
    cfg.model['save_dir'] = cfg.work_dir
    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    
    num_parameters = sum([p.numel() for p in model.parameters()])
    num_trainable_parameters = sum([p.numel() for p in model.parameters() if p.requires_grad])
    logger.info(f'#Params: {num_parameters / 1e6:.2f} M')
    logger.info(f'#TrainableParams: {num_trainable_parameters / 1e6:.2f} M')

    # model parm init
    if hasattr(model, 'backbone'):
        if hasattr(model.backbone, 'init_weights'):
            print("backbone init_weights")
            model.backbone.init_weights()
            model.backbone = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model.backbone)
    if hasattr(model, 'text_encoder'):
        if hasattr(model.text_encoder, 'init_weights'):
            print("text_encoder init_weights")
            model.text_encoder.init_weights()
    if hasattr(model, 'model'):
        model.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model.model)
    # logger.info(model)

    if args.profile_flops:
        rank = torch.distributed.get_rank() \
            if torch.distributed.is_available() and \
            torch.distributed.is_initialized() else 0
        if rank == 0:
            profile_model_flops(model, args, logger)
        if torch.distributed.is_available() and \
                torch.distributed.is_initialized():
            torch.distributed.barrier()
        if args.flops_only:
            return

    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))

    if cfg.checkpoint_config is not None:
        # save mmseg version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    if args.finetune:
        model.eval()
    with torch.autograd.set_detect_anomaly(True):
        train_segmentor(
            model,
            datasets,
            cfg,
            distributed=distributed,
            validate=(not args.no_validate),
            timestamp=timestamp,
            meta=meta)

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    main()
