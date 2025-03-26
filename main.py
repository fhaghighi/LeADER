import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models
from data_loader import DiseasedPatchesAndEmbeddings,VinDrCXRImagesAndEmbeddings,PadChestImagesAndEmbeddings,CheXpertImagesAndEmbeddings,ShenzhenImagesAndEmbeddings,MIMICImagesAndEmbeddings,RSNAPneumoniaImagesAndEmbeddings,General_Local_GLobal_KD
from swin_transformer import SwinTransformer
import utils
import convnext as convnext
import yaml
from models import SimpleWrapper, AnatomyModelWrapper, ProjectionHead

def get_args_parser():
    parser = argparse.ArgumentParser('David', add_help=False)
    parser.add_argument('--arch', default='swin_base', type=str,
        help="""Name of architecture to train.""",choices=['swin_base','convnext_base'])
    parser.add_argument('--out_dim', default=1376, type=int, help="""Dimensionality of
        the disease head output """)
    parser.add_argument('--anatomy_out_dim', default=2048, type=int, help="""Dimensionality of
        the anatomy head output""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the heads.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in heads")
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True)
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4)
    parser.add_argument('--clip_grad', type=float, default=3.0)
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training).""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'])
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.8, 1.))
    parser.add_argument('--dataset', default='VinDR_CXR_patch', action="append")
    parser.add_argument('--disease_embeddings_path', default=None, type=str,help='path to disease embeddings')

    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=10, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument("--pretrained_weights", default=None, type=str, help="pretrained encoder path")
    parser.add_argument('--use_head_pretrained_weights', action='store_true',help='use KD pretrained head weights for disease branch')
    parser.add_argument('--anatomy_model_use_head', action='store_true',help='use pretrained head weights for anatomy branch')
    parser.add_argument("--anatomy_expert_pretrained_weights", default=None, type=str, help="pretrained encoder path")
    parser.add_argument('--kd_disease_loss_weight', type=float, default=1, help="kd disease loss weight")
    parser.add_argument('--kd_anatomy_loss_weight', type=float, default=1, help="kd anatomy loss weight")
    parser.add_argument('--head_layers', default=1, type=int, help="""number of layers for KD head""")
    parser.add_argument('--resume', default=None, type=str, help='set to True for continue training')
    parser.add_argument('--anatomy_model_arch', default="resnet50", type=str, help='anatomy expert model')
    parser.add_argument('--loss_mode', default="DA", type=str, help='DA|D|A')
    return parser


def train(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # ============ preparing data ... ============
    with open("./datasets_config.yaml", 'r') as stream:
        datasets_config = yaml.safe_load(stream)

    transform = DataAugmentations(args.global_crops_scale
    )

    concat_datasets = []
    for key in list(datasets_config.keys()):
        if key in args.dataset:
            if key == "chexpert":
                dataset = CheXpertImagesAndEmbeddings(images_path=datasets_config[key]['images_path'], file_path=datasets_config[key]['train_list'],
                                                   embedding_path=datasets_config[key]['disease_embedding_path'], augment=transform)

            elif key == "vindrcxr":
                dataset = VinDrCXRImagesAndEmbeddings(images_path=datasets_config[key]['images_path'], file_path=datasets_config[key]['train_list'],
                                                   embedding_path=datasets_config[key]['disease_embedding_path'], augment=transform)
            elif key == "rsna":
                dataset = RSNAPneumoniaImagesAndEmbeddings(images_path=datasets_config[key]['images_path'], file_path=datasets_config[key]['train_list'],
                                                   embedding_path=datasets_config[key]['disease_embedding_path'], augment=transform)
            elif key == "mimic":
                dataset = MIMICImagesAndEmbeddings(images_path=datasets_config[key]['images_path'], file_path=datasets_config[key]['train_list'],
                                                   embedding_path=datasets_config[key]['disease_embedding_path'], augment=transform)
            elif key == "shenzhen":
                dataset = ShenzhenImagesAndEmbeddings(images_path=datasets_config[key]['images_path'], file_path=datasets_config[key]['train_list'],
                                                   embedding_path=datasets_config[key]['disease_embedding_path'], augment=transform)

            elif key=="nih14" or key=="padchest":
                dataset = PadChestImagesAndEmbeddings(images_path=datasets_config[key]['images_path'], file_path=datasets_config[key]['train_list'],
                                                   embedding_path=datasets_config[key]['disease_embedding_path'], augment=transform)
            else:
                dataset = General_Local_GLobal_KD(images_path=datasets_config[key]['images_path'], file_path=datasets_config[key]['train_list'], embeddings_path=datasets_config[key]['disease_embedding_path'], augment=transform, img_prfix=value[3],mode=value[4])


            concat_datasets.append(dataset)

    train_dataset = torch.utils.data.ConcatDataset(concat_datasets)
    sampler = torch.utils.data.DistributedSampler(train_dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(train_dataset)} images.")

    # ============ building models ... ============

    if args.arch == "swin_base":
        student = SwinTransformer(img_size=224, patch_size=4, in_chans=3, num_classes=0, embed_dim=128,
                                         depths=[2, 2, 18, 2],
                                         num_heads=[4, 8, 16, 32], window_size=7, mlp_ratio=4.,
                                         qkv_bias=True, qk_scale=None, drop_rate=0, drop_path_rate=0.1, ape=False,
                                         patch_norm=True, use_checkpoint=False)
        embed_dim = 1024
    elif args.arch == "swin_small":
        student = SwinTransformer(img_size=224, patch_size=4, in_chans=3, num_classes=0, embed_dim=96,
                                         depths=[2, 2, 18, 2],
                                         num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4.,
                                         qkv_bias=True, qk_scale=None, drop_rate=0, drop_path_rate=0.1, ape=False,
                                         patch_norm=True, use_checkpoint=False)
        embed_dim = 768
    elif args.arch == "swin_tiny":
        student = SwinTransformer(img_size=224, patch_size=4, in_chans=3, num_classes=0, embed_dim=96,
                                         depths=[2, 2, 6, 2],
                                         num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4.,
                                         qkv_bias=True, qk_scale=None, drop_rate=0, drop_path_rate=0.1, ape=False,
                                         patch_norm=True, use_checkpoint=False)
        embed_dim = 768
    elif args.arch in convnext.__dict__.keys():
        student = convnext.__dict__[args.arch](drop_path_rate=0)
        if args.arch == "convnext_base":
            embed_dim = 1024
        else:
            embed_dim = 768
    else:
        print(f"Unknow architecture: {args.arch}")

    if args.pretrained_weights is not None:
        print("=> loading checkpoint '{}'".format(args.pretrained_weights))
        if os.path.isfile(args.pretrained_weights):
            state_dict = torch.load(args.pretrained_weights, map_location="cpu")
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            elif "teacher" in state_dict:
                state_dict = state_dict["teacher"]
            elif "student" in state_dict:
                state_dict = state_dict["student"]

            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        if args.use_head_pretrained_weights:
            state_dict = {k.replace("head.", "disease_head."): v for k, v in state_dict.items()}

        msg = student.load_state_dict(state_dict, strict=False)
        print("missing keys:", msg)

    student = SimpleWrapper(student, ProjectionHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
        nlayers=args.head_layers
    ) , ProjectionHead(
        embed_dim,
        args.anatomy_out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
        nlayers=args.head_layers
    ))


    student= student.cuda()
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu],broadcast_buffers=False)
    anatomy_expert_model = AnatomyModelWrapper(args)
    state_dict = torch.load(args.anatomy_expert_pretrained_weights, map_location="cpu")
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    elif 'teacher' in state_dict:
        state_dict = state_dict['teacher']
    elif 'model' in state_dict:
        state_dict = state_dict['model']
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("encoder.", "base_model."): v for k, v in state_dict.items()}

    msg = anatomy_expert_model.base_model.load_state_dict(state_dict, strict=False)
    print("=> loaded anatomy expert  pre-trained model '{}'".format(args.anatomy_expert_pretrained_weights))
    print("=> missing keys'{}'".format(msg))
    for p in anatomy_expert_model.parameters():
        p.requires_grad = False
    anatomy_expert_model = anatomy_expert_model.cuda()

    # ============ preparing loss ... ============
    disease_kd_loss = torch.nn.MSELoss().cuda()
    anatomy_kd_loss = torch.nn.MSELoss().cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )

    print(f"Loss, optimizer and schedulers ready.")

    # ============ training ... ============
    start_time = time.time()
    print("Starting training !")
    for epoch in range(args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch ... ============
        train_stats = train_one_epoch(student, anatomy_expert_model,disease_kd_loss,anatomy_kd_loss,
            data_loader, optimizer, lr_schedule, wd_schedule,
            epoch, fp16_scaler, args)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def train_one_epoch(student,anatomy_expert_model, disease_kd_loss, anatomy_kd_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, epoch,
                    fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, (images, embds_targets,imageLabel) in enumerate(metric_logger.log_every(data_loader, 1, header)):
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:
                param_group["weight_decay"] = wd_schedule[it]
        images = images.cuda(non_blocking=True)
        embds_targets = embds_targets.float().cuda(non_blocking=True)

        with torch.cuda.amp.autocast(fp16_scaler is not None):
            disease_embeddings_outputs,anatomy_embeddings_outputs = student(images)
            anatomy_embeddings_targets = anatomy_expert_model(images)
            embds_targets = nn.functional.normalize(embds_targets, dim=1)
            loss_kd_disease = disease_kd_loss(disease_embeddings_outputs, embds_targets)
            loss_kd_anatomy = anatomy_kd_loss(anatomy_embeddings_outputs, anatomy_embeddings_targets)
            loss = loss_kd_disease+loss_kd_anatomy

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__ == '__main__':
    parser = argparse.ArgumentParser('LeADER', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train(args)
