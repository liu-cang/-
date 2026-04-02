import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.sampler import BatchSampler, RandomSampler
from datasets.coco_style_dataset import CocoStyleDataset, CocoStyleDatasetTeaching
from models.backbones import ResNet50MultiScale, ResNet18MultiScale, ResNet101MultiScale
from models.positional_encoding import PositionEncodingSine
# from models.deformable_detr import DeformableDETR
# # from models.deformable_detr_ours import DeformableDETR
# from models.deformable_transformer import DeformableTransformer
from models.criterion import SetCriterion, Set_UNK_Criterion
from datasets.augmentations import weak_aug, strong_aug, base_trans

import numpy as np

def build_sampler(args, dataset, split):
    if split == 'train':
        if args.distributed:
            sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
        else:
            sampler = RandomSampler(dataset)
        batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
    else:
        if args.distributed:
            sampler = DistributedSampler(dataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)
        batch_sampler = BatchSampler(sampler, args.eval_batch_size, drop_last=False)
    return batch_sampler


def info_dataset(dataset, dataset_name, domain, split):
    print('='*50)
    print(f'Name:{dataset_name}, Domain:{domain}, Split:{split}')
    for k in dataset.coco.cats.keys():
        cats_key = k
        name = dataset.coco.cats[k]['name']
        cats_id = dataset.coco.cats[k]['id']
        catToimg = dataset.coco.catToImgs[k]
        print(f'Classes:{name}, Cats_id:{cats_id}, instances:{len(catToimg)}')
    print('='*50)

def build_dataloader(args, dataset_name, domain, split, trans):
    dataset = CocoStyleDataset(root_dir=args.data_root,
                               dataset_name=dataset_name,
                               domain=domain,
                               split=split,
                               transforms=trans)
    info_dataset(dataset, dataset_name, domain, split)
    batch_sampler = build_sampler(args, dataset, split)
    data_loader = DataLoader(dataset=dataset,
                             batch_sampler=batch_sampler,
                             collate_fn=CocoStyleDataset.collate_fn,
                             num_workers=args.num_workers)
    return data_loader


def build_dataloader_sfuod(args, dataset_name, domain, split, trans, unk_version):
    dataset = CocoStyleDataset(root_dir=args.data_root,
                               dataset_name=dataset_name,
                               domain=domain,
                               split=split,
                               transforms=trans,
                               sfuod=True, unk_version=unk_version)
    info_dataset(dataset, dataset_name, domain, split)
    batch_sampler = build_sampler(args, dataset, split)
    data_loader = DataLoader(dataset=dataset,
                             batch_sampler=batch_sampler,
                             collate_fn=CocoStyleDataset.collate_fn,
                             num_workers=args.num_workers)
    return data_loader


def build_dataloader_teaching(args, dataset_name, domain, split):
    dataset = CocoStyleDatasetTeaching(root_dir=args.data_root,
                                       dataset_name=dataset_name,
                                       domain=domain,
                                       split=split,
                                       weak_aug=weak_aug,
                                       strong_aug=strong_aug,
                                       final_trans=base_trans)
    info_dataset(dataset, dataset_name, domain, split)
    batch_sampler = build_sampler(args, dataset, split)
    data_loader = DataLoader(dataset=dataset,
                             batch_sampler=batch_sampler,
                             collate_fn=CocoStyleDatasetTeaching.collate_fn_teaching,
                             num_workers=args.num_workers)
    return data_loader

def build_dataloader_teaching_sfuod(args, dataset_name, domain, split, unk_version):
    dataset = CocoStyleDatasetTeaching(root_dir=args.data_root,
                                       dataset_name=dataset_name,
                                       domain=domain,
                                       split=split,
                                       weak_aug=weak_aug,
                                       strong_aug=strong_aug,
                                       final_trans=base_trans,
                                       sfuod=True, unk_version=unk_version)
    info_dataset(dataset, dataset_name, domain, split)
    batch_sampler = build_sampler(args, dataset, split)
    data_loader = DataLoader(dataset=dataset,
                             batch_sampler=batch_sampler,
                             collate_fn=CocoStyleDatasetTeaching.collate_fn_teaching,
                             num_workers=args.num_workers)
    return data_loader


def build_model(args, device):
    if args.backbone == 'resnet50':
        backbone = ResNet50MultiScale()
    elif args.backbone == 'resnet18':
        backbone = ResNet18MultiScale()
    elif args.backbone == 'resnet101':
        backbone = ResNet101MultiScale()
    else:
        raise ValueError('Invalid args.backbone name: ' + args.backbone)
    position_encoding = PositionEncodingSine()
    
    # if args.mode == 'teaching_unknown_specialist' or args.mode == 'sfuod_exp':
    if args.mode == 'teaching_unknown_specialist' or args.mode == 'teaching_unknown_specialist2':
        from models.deformable_detr_ours import DeformableDETR
        from models.deformable_transformer_ours import DeformableTransformer
    elif args.mode == 'teaching_upuk':
        from models.deformable_detr_upuk import DeformableDETR
        from models.deformable_transformer import DeformableTransformer
    else:
        from models.deformable_detr import DeformableDETR
        from models.deformable_transformer import DeformableTransformer
    
    transformer = DeformableTransformer(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        feedforward_dim=args.feedforward_dim,
        dropout=args.dropout
    )
    
    
    model = DeformableDETR(
        backbone=backbone,
        position_encoding=position_encoding,
        transformer=transformer,
        num_classes=args.num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels
    )
    model.to(device)
    return model


def build_criterion(args, device, only_class_loss=False, high_quality_matches=False):
    criterion = SetCriterion(
        num_classes=args.num_classes,
        coef_class=args.coef_class,
        coef_boxes=0.0 if only_class_loss else args.coef_boxes,
        coef_giou=0.0 if only_class_loss else args.coef_giou,
        alpha_focal=args.alpha_focal,
        high_quality_matches=high_quality_matches,
        device=device
    )
    criterion.to(device)
    return criterion

def build_unk_criterion(args, device, only_class_loss=False):
    criterion = Set_UNK_Criterion(
        num_classes=args.num_classes,
        coef_class=args.coef_class,
        coef_boxes=0.0 if only_class_loss else args.coef_boxes,
        coef_giou=0.0 if only_class_loss else args.coef_giou,
        alpha_focal=args.alpha_focal,
        high_quality_matches=False,
        device=device
    )
    criterion.to(device)
    return criterion


def build_optimizer(args, model):
    params_backbone = [param for name, param in model.named_parameters()
                       if 'backbone' in name]
    params_linear_proj = [param for name, param in model.named_parameters()
                          if 'reference_points' in name or 'sampling_offsets' in name]
    params = [param for name, param in model.named_parameters()
              if 'backbone' not in name and 'reference_points' not in name and 'sampling_offsets' not in name]
    param_dicts = [
        {'params': params, 'lr': args.lr},
        {'params': params_backbone, 'lr': args.lr_backbone},
        {'params': params_linear_proj, 'lr': args.lr_linear_proj},
    ]
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    return optimizer

def build_optimizer_ours(args, model):
    params_backbone = [param for name, param in model.named_parameters()
                       if 'backbone' in name]
    params_linear_proj = [param for name, param in model.named_parameters()
                          if 'reference_points' in name or 'sampling_offsets' in name]
    params_additional = [param for name, param in model.named_parameters()
                          if 'res_embed_proj' in name or 'cross_attn_block' in name or 'collab_layers' in name or 'pt_wise' in name]
    params = [param for name, param in model.named_parameters()
              if 'backbone' not in name and 'reference_points' not in name and 'sampling_offsets' not in name
              and 'res_embed_proj' not in name and 'cross_attn_block' not in name and 'collab_layers' not in name and 'pt_wise' not in name]
    param_dicts = [
        {'params': params, 'lr': args.lr},
        {'params': params_backbone, 'lr': args.lr_backbone},
        {'params': params_linear_proj, 'lr': args.lr_linear_proj},
        {'params': params_additional, 'lr': args.lr_targets},
    ]
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    return optimizer




def build_teacher(args, student_model, device, learnable=False):
    teacher_model = build_model(args, device)
    state_dict, student_state_dict = teacher_model.state_dict(), student_model.state_dict()
    for key, value in state_dict.items():
        if learnable:
            state_dict[key] = student_state_dict[key].clone()
        else:
            state_dict[key] = student_state_dict[key].clone().detach()
    teacher_model.load_state_dict(state_dict)
    return teacher_model

# def build_teacher(args, student_model, device):
#     teacher_model = build_model(args, device)
#     state_dict, student_state_dict = teacher_model.state_dict(), student_model.state_dict()
#     for key, value in state_dict.items():
#         state_dict[key] = student_state_dict[key].clone().detach()
#     teacher_model.load_state_dict(state_dict)
#     return teacher_model

def get_copy(model):
    copy = build_model(model.args, model.device)
    copy.load_state_dict(model.state_dict())
    return copy
