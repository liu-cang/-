import time
import datetime
import json

import copy

import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader

from datasets.coco_style_dataset import DataPreFetcher
from datasets.coco_eval import CocoEvaluator
# from datasets.sfuod_eval import SFUODevaluator

from models.criterion import post_process, get_pseudo_labels, get_known_pseudo_labels, get_unknown_pseudo_labels, get_unknown_pseudo_labels_attn, get_topk_outputs, SetCriterion
from utils.distributed_utils import is_main_process
from utils.box_utils import box_cxcywh_to_xyxy, convert_to_xywh
from collections import defaultdict
from typing import List

from datasets.masking import Masking
from scipy.optimize import linear_sum_assignment
from utils.box_utils import box_cxcywh_to_xyxy, generalized_box_iou
from utils import selective_reinitialize



def train_one_epoch_standard(model: torch.nn.Module,
                             criterion: torch.nn.Module,
                             data_loader: DataLoader,
                             optimizer: torch.optim.Optimizer,
                             device: torch.device,
                             epoch: int,
                             clip_max_norm: float = 0.0,
                             print_freq: int = 20,
                             flush: bool = True):
    """
    Train the standard detection model, using only labelled training set source.
    """
    start_time = time.time()
    model.train()
    criterion.train()
    fetcher = DataPreFetcher(data_loader, device=device)
    images, masks, annotations = fetcher.next()
    # Training statistics
    epoch_loss = torch.zeros(1, dtype=torch.float, device=device, requires_grad=False)
    epoch_loss_dict = defaultdict(float)
    for i in range(len(data_loader)):
        # Forward
        out = model(images, masks)
        # Loss
        loss, loss_dict = criterion(out, annotations)
        # Backward
        optimizer.zero_grad()
        loss.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        # Record loss
        epoch_loss += loss.detach()
        for k, v in loss_dict.items():
            epoch_loss_dict[k] += v.detach().cpu().item()
        # Data pre-fetch
        images, masks, annotations = fetcher.next()
        # Log
        if is_main_process() and (i + 1) % print_freq == 0:
            print('Training epoch ' + str(epoch) + ' : [ ' + str(i + 1) + '/' + str(len(data_loader)) + ' ] ' +
                  'total loss: ' + str(loss.detach().cpu().numpy()), flush=flush)
    # Final process of training statistic
    epoch_loss /= len(data_loader)
    for k, v in epoch_loss_dict.items():
        epoch_loss_dict[k] /= len(data_loader)
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Training epoch ' + str(epoch) + ' finished. Time cost: ' + total_time_str +
          ' Epoch loss: ' + str(epoch_loss.detach().cpu().numpy()), flush=flush)
    return epoch_loss, epoch_loss_dict


def train_one_epoch_teaching_standard(student_model: torch.nn.Module,
                                      teacher_model: torch.nn.Module,
                                      criterion_pseudo: torch.nn.Module,
                                      target_loader: DataLoader,
                                      optimizer: torch.optim.Optimizer,
                                      thresholds: List[float],
                                      alpha_ema: float,
                                      device: torch.device,
                                      epoch: int,
                                      clip_max_norm: float = 0.0,
                                      print_freq: int = 20,
                                      flush: bool = True,
                                      fix_update_iter: int = 1):
    """
    Train the student model with the teacher model, using only unlabeled training set target .
    """
    start_time = time.time()
    student_model.train()
    teacher_model.train()
    criterion_pseudo.train()
    target_fetcher = DataPreFetcher(target_loader, device=device)
    target_images, target_masks, _ = target_fetcher.next()
    target_teacher_images, target_student_images = target_images[0], target_images[1]
    # Record epoch losses
    epoch_loss = torch.zeros(1, dtype=torch.float, device=device, requires_grad=False)

    # Training data statistics
    epoch_target_loss_dict = defaultdict(float)
    total_iters = len(target_loader)

    for iter in range(total_iters):
        # Target teacher forward
        with torch.no_grad():
            teacher_out = teacher_model(target_teacher_images, target_masks)
            pseudo_labels = get_pseudo_labels(teacher_out['logit_all'][-1], teacher_out['boxes_all'][-1], thresholds)
            #todo OW-DETR Pseudo Labeling for unknown
            # pseudo_labels = get_ow_pseudo_labels(target_teacher_images, teacher_out['resnet_1024_feat'], teacher_out['logit_all'][-1], teacher_out['boxes_all'][-1], thresholds)

        #todo Unknown Pseudo Label is needed
        # ps_labels=torch.tensor([])
        # for ps_dict in pseudo_labels:
        #     ps_labels = torch.cat([ps_labels, ps_dict['labels'].cpu()], dim=0)
        # print('[Pseudo Labels]', ps_labels.unique(return_counts=True))
        
        # Target student forward
        target_student_out = student_model(target_student_images, target_masks)
        target_loss, target_loss_dict = criterion_pseudo(target_student_out, pseudo_labels)

        loss = target_loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), clip_max_norm)
        optimizer.step()

        # Record epoch losses
        epoch_loss += loss.detach()

        # update loss_dict
        for k, v in target_loss_dict.items():
            epoch_target_loss_dict[k] += v.detach().cpu().item()

        if iter % fix_update_iter == 0:
            with torch.no_grad():
                state_dict, student_state_dict = teacher_model.state_dict(), student_model.state_dict()
                for key, value in state_dict.items():
                    state_dict[key] = alpha_ema * value + (1 - alpha_ema) * student_state_dict[key].detach()
                teacher_model.load_state_dict(state_dict)

        # Data pre-fetch
        target_images, target_masks, _ = target_fetcher.next()
        if target_images is not None:
            target_teacher_images, target_student_images = target_images[0], target_images[1]

        # Log
        if is_main_process() and (iter + 1) % print_freq == 0:
            print('Teaching epoch ' + str(epoch) + ' : [ ' + str(iter + 1) + '/' + str(total_iters) + ' ] ' +
                  'total loss: ' + str(loss.detach().cpu().numpy()), flush=flush)

    # Final process of loss dict
    epoch_loss /= total_iters
    for k, v in epoch_target_loss_dict.items():
        epoch_target_loss_dict[k] /= total_iters
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Teaching epoch ' + str(epoch) + ' finished. Time cost: ' + total_time_str +
          ' Epoch loss: ' + str(epoch_loss.detach().cpu().numpy()), flush=flush)
    return epoch_loss, epoch_target_loss_dict


# with torch.no_grad():
#     teacher_out_k = teacher_model(target_teacher_images, target_masks)
#     k_pseudo_labels, k_batch_indices = get_known_pseudo_labels(teacher_out_k['logit_all'][-1], teacher_out_k['boxes_all'][-1], thresholds)
#     u_pseudo_labels = get_unknown_pseudo_labels(teacher_out_k['logit_all'][-1], teacher_out_k['boxes_all'][-1], teacher_out_k['hidden_states_last'], k_batch_indices)

# if len(u_pseudo_labels) > 0:
#     pseudo_labels = []
#     for k_anno, u_anno in zip(k_pseudo_labels, u_pseudo_labels):
#         # print('k_anno:',k_anno['scores'].shape[0])
#         # print('u_anno:',u_anno['scores'].shape[0])
#         total_scores = torch.cat([k_anno['scores'], u_anno['scores']])
#         # print('total_anno:',total_scores.shape[0])
#         total_labels = torch.cat([k_anno['labels'], u_anno['labels']])
#         total_boxes = torch.cat([k_anno['boxes'], u_anno['boxes']])
#         pseudo_labels.append({'scores': total_scores, 'labels': total_labels, 'boxes': total_boxes})
# else:
#     pseudo_labels = k_pseudo_labels


def train_one_epoch_teaching_unknown_specialist(student_model: torch.nn.Module,
                                      teacher_model: torch.nn.Module,
                                  init_student_model: torch.nn.Module,
                                  criterion_pseudo: torch.nn.Module,
                                  criterion_pseudo_weak: torch.nn.Module,
                                  target_loader: DataLoader,
                                  optimizer: torch.optim.Optimizer,
                                  thresholds: List[float],
                                  coef_masked_img: float,
                                  alpha_ema: float,
                                  device: torch.device,
                                  epoch: int,
                                  keep_modules: List[str],
                                  clip_max_norm: float = 0.0,
                                  print_freq: int = 20,
                                  masking: Masking = None,
                                  flush: bool = True,
                                  fix_update_iter: int = 1,
                                  max_update_iter: int = 5,
                                  dynamic_update: bool = False,
                                  stu_buffer_cost: List[float] = None,
                                  stu_buffer_img: List[torch.Tensor] = None,
                                  stu_buffer_mask: List[torch.Tensor] = None,
                                  res_dict: dict = None,
                                  use_pseudo_label_weights: bool = False,
                                  use_loss_student: bool = False,
                                  unk_thresh: float = 0.3):
    """
    Train the student model with the teacher model, using only unlabeled training set target .
    """
    start_time = time.time()
    student_model.train()
    teacher_model.train()
    init_student_model.train()
    criterion_pseudo.train()
    criterion_pseudo_weak.train()
    target_fetcher = DataPreFetcher(target_loader, device=device)
    target_images, target_masks, _ = target_fetcher.next()
    target_teacher_images, target_student_images = target_images[0], target_images[1]
    # Record epoch losses
    epoch_loss = torch.zeros(1, dtype=torch.float, device=device, requires_grad=False)

    # Training data statistics
    epoch_target_loss_dict = defaultdict(float)
    total_iters = len(target_loader)

    for iter in range(total_iters):
        # Target teacher forward
        with torch.no_grad():
            teacher_out = teacher_model(target_teacher_images, target_masks, bi_attn=False)
            # pseudo_labels = get_pseudo_labels(teacher_out['logit_all'][-1], teacher_out['boxes_all'][-1], thresholds)
        #* Ours .. 
            teacher_out_k = teacher_model(target_teacher_images, target_masks, bi_attn=False)
            k_pseudo_labels, k_batch_indices = get_known_pseudo_labels(teacher_out_k['logit_all'][-1], teacher_out_k['boxes_all'][-1], thresholds)
            u_pseudo_labels = get_unknown_pseudo_labels(teacher_out_k['logit_all'][-1], teacher_out_k['boxes_all'][-1], teacher_out_k['hidden_states_last'], k_batch_indices, unk_threshold=unk_thresh)
            #* u_pseudo_labels = get_unknown_pseudo_labels_attn(target_teacher_images, teacher_out_k['logit_all'][-1], teacher_out_k['boxes_all'][-1], teacher_out_k['resnet_1024_feat'], k_batch_indices, unk_threshold=unk_thresh)
        if len(u_pseudo_labels) > 0:
            pseudo_labels = []
            for k_anno, u_anno in zip(k_pseudo_labels, u_pseudo_labels):
                total_scores = torch.cat([k_anno['scores'], u_anno['scores']])
                total_labels = torch.cat([k_anno['labels'], u_anno['labels']])
                total_boxes = torch.cat([k_anno['boxes'], u_anno['boxes']])
                pseudo_labels.append({'scores': total_scores, 'labels': total_labels, 'boxes': total_boxes})
        else:
            pseudo_labels = k_pseudo_labels
        #* ============================================
        
        target_student_out = student_model(target_student_images, target_masks)
        # loss from pseudo labels of current teacher
        target_loss, target_loss_dict = criterion_pseudo(target_student_out, pseudo_labels)

        #? Masked target student forward
        #* masked_target_images = masking(target_student_images)
        #* masked_target_student_out = student_model(masked_target_images, target_masks)
        #* masked_target_loss, masked_target_loss_dict = criterion_pseudo(masked_target_student_out, pseudo_labels)

        # Final loss
        #* loss = target_loss + coef_masked_img * masked_target_loss
        loss = target_loss

        # Dynamic update EMA teacher : Create buffer cost and buffer image in student model
        if dynamic_update:
            with torch.no_grad():
                # print("[Engine] Student_Foward: Dunamic Update")
                student_out = student_model(target_teacher_images, target_masks)
            # variance logit
            student_out_var = student_out['logit_all'].var(dim=0)
            var_total = student_out_var.mean().item()
            stu_buffer_cost.append(var_total)

            # Store batch data to buffer
            stu_buffer_img.append(target_teacher_images.clone().detach())
            stu_buffer_mask.append(target_masks.clone().detach())

            if len(stu_buffer_cost) == 1:
                with torch.no_grad():
                    init_student_model.load_state_dict(student_model.state_dict())

            if len(stu_buffer_cost) >= 1:
                with torch.no_grad():
                    init_student_out = init_student_model(target_teacher_images, target_masks)
                    # init_pseudo_labels = get_pseudo_labels(init_student_out['logit_all'][-1], init_student_out['boxes_all'][-1],thresholds)
                    
                #* Ours..  
                    pseudo_labels_init_student, init_k_batch_indices = get_known_pseudo_labels(init_student_out['logit_all'][-1], init_student_out['boxes_all'][-1],thresholds)
                    init_u_pseudo_labels = get_unknown_pseudo_labels(init_student_out['logit_all'][-1], init_student_out['boxes_all'][-1], init_student_out['hidden_states_last'], init_k_batch_indices, unk_threshold=unk_thresh)
                if len(init_u_pseudo_labels) > 0:
                    init_pseudo_labels = []
                    for k_anno, u_anno in zip(pseudo_labels_init_student, init_u_pseudo_labels):
                        total_scores = torch.cat([k_anno['scores'], u_anno['scores']])
                        total_labels = torch.cat([k_anno['labels'], u_anno['labels']])
                        total_boxes = torch.cat([k_anno['boxes'], u_anno['boxes']])
                        init_pseudo_labels.append({'scores': total_scores, 'labels': total_labels, 'boxes': total_boxes})
                else:
                    init_pseudo_labels = pseudo_labels_init_student
                #* ==================================
                
                # Loss from pseudo labels of init student
                init_student_loss, init_student_loss_dict = criterion_pseudo_weak(target_student_out,
                                                                                    init_pseudo_labels, use_pseudo_label_weights)
                #* masked_init_student_loss, masked_init_student_loss_dict = criterion_pseudo_weak(masked_target_student_out, init_pseudo_labels, use_pseudo_label_weights)
                #* loss_init_student = init_student_loss + coef_masked_img * masked_init_student_loss
                loss_init_student = init_student_loss
                loss += loss_init_student

        # Backward
        optimizer.zero_grad()
        loss.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), clip_max_norm)
        optimizer.step()

        # Record epoch losses
        epoch_loss += loss.detach()

        # update loss_dict
        for k, v in target_loss_dict.items():
            epoch_target_loss_dict[k] += v.detach().cpu().item()

        # Dynamic update EMA teacher : Update weight of teacher model
        if dynamic_update:
            if len(stu_buffer_cost) < max_update_iter:
                all_score = eval_stu(student_model, stu_buffer_img, stu_buffer_mask)
                compare_score = np.array(all_score) - np.array(stu_buffer_cost)
                # print(len(stu_buffer_cost), len(all_score), np.mean(compare_score<0))
                if np.mean(compare_score < 0) >= 0.5:
                    res_dict['stu_ori'].append(stu_buffer_cost)
                    res_dict['stu_now'].append(all_score)
                    res_dict['update_iter'].append(len(stu_buffer_cost))

                    df = pd.DataFrame(res_dict)
                    df.to_csv('dynamic_update.csv')

                    with torch.no_grad():
                        state_dict, student_state_dict = teacher_model.state_dict(), student_model.state_dict()
                        for key, value in state_dict.items():
                            state_dict[key] = alpha_ema * value + (1 - alpha_ema) * student_state_dict[key].detach()
                        teacher_model.load_state_dict(state_dict)

                    # Clear buffer
                    stu_buffer_cost = []
                    stu_buffer_img = []
                    stu_buffer_mask = []
            else:
                # print(len(stu_buffer_cost), 'Load previous student model weight')
                with torch.no_grad():
                    student_model = selective_reinitialize(student_model, init_student_model.state_dict(), keep_modules)

                # Clear buffer
                stu_buffer_cost = []
                stu_buffer_img = []
                stu_buffer_mask = []
        else:
            # EMA update teacher after fix iteration
            if iter % fix_update_iter == 0:
                
                with torch.no_grad():
                    state_dict, student_state_dict = teacher_model.state_dict(), student_model.state_dict()
                    for key, value in state_dict.items():
                        state_dict[key] = alpha_ema * value + (1 - alpha_ema) * student_state_dict[key].detach()
                    teacher_model.load_state_dict(state_dict)


        # Data pre-fetch
        target_images, target_masks, _ = target_fetcher.next()
        if target_images is not None:
            target_teacher_images, target_student_images = target_images[0], target_images[1]

        # Log
        if is_main_process() and (iter + 1) % print_freq == 0:
            print('Teaching epoch ' + str(epoch) + ' : [ ' + str(iter + 1) + '/' + str(total_iters) + ' ] ' +
                  'total loss: ' + str(loss.detach().cpu().numpy()), flush=flush)

    # Final process of loss dict
    epoch_loss /= total_iters
    for k, v in epoch_target_loss_dict.items():
        epoch_target_loss_dict[k] /= total_iters
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Teaching epoch ' + str(epoch) + ' finished. Time cost: ' + total_time_str +
          ' Epoch loss: ' + str(epoch_loss.detach().cpu().numpy()), flush=flush)
    return epoch_loss, epoch_target_loss_dict


# def train_one_epoch_teaching_unknown_specialist2(student_model: torch.nn.Module,
def train_one_epoch_teaching_unknown_specialist2(student_model: torch.nn.Module,
                                      teacher_model: torch.nn.Module,
                                      criterion_pseudo: torch.nn.Module,
                                      target_loader: DataLoader,
                                      optimizer: torch.optim.Optimizer,
                                      thresholds: List[float],
                                      alpha_ema: float,
                                      device: torch.device,
                                      epoch: int,
                                      clip_max_norm: float = 0.0,
                                      print_freq: int = 20,
                                      flush: bool = True,
                                      fix_update_iter: int = 1,
                                      unk_thresh: float = 0.3):
    """
    Train the student model with the teacher model, using only unlabeled training set target .
    """
    start_time = time.time()
    student_model.train()
    teacher_model.train()
    criterion_pseudo.train()
    target_fetcher = DataPreFetcher(target_loader, device=device)
    target_images, target_masks, _ = target_fetcher.next()
    target_teacher_images, target_student_images = target_images[0], target_images[1]
    # Record epoch losses
    epoch_loss = torch.zeros(1, dtype=torch.float, device=device, requires_grad=False)

    # Training data statistics
    epoch_target_loss_dict = defaultdict(float)
    total_iters = len(target_loader)

    for iter in range(total_iters):
        # Target teacher forward
        with torch.no_grad():
            #? Original
            # teacher_out = teacher_model(target_teacher_images, target_masks)
            # pseudo_labels = get_pseudo_labels(teacher_out['logit_all'][-1], teacher_out['boxes_all'][-1], thresholds)
            
            teacher_out_k = teacher_model(target_teacher_images, target_masks, bi_attn=False)
            k_pseudo_labels, k_batch_indices = get_known_pseudo_labels(teacher_out_k['logit_all'][-1], teacher_out_k['boxes_all'][-1], thresholds)
            u_pseudo_labels = get_unknown_pseudo_labels(teacher_out_k['logit_all'][-1], teacher_out_k['boxes_all'][-1], teacher_out_k['hidden_states_last'], k_batch_indices, unk_threshold=unk_thresh)
        if len(u_pseudo_labels) > 0:
            pseudo_labels = []
            for k_anno, u_anno in zip(k_pseudo_labels, u_pseudo_labels):
                total_scores = torch.cat([k_anno['scores'], u_anno['scores']])
                total_labels = torch.cat([k_anno['labels'], u_anno['labels']])
                total_boxes = torch.cat([k_anno['boxes'], u_anno['boxes']])
                pseudo_labels.append({'scores': total_scores, 'labels': total_labels, 'boxes': total_boxes})
        else:
            pseudo_labels = k_pseudo_labels
        
        # Target student forward
        target_student_out = student_model(target_student_images, target_masks)
        target_loss, target_loss_dict = criterion_pseudo(target_student_out, pseudo_labels)

        loss = target_loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), clip_max_norm)
        optimizer.step()

        # Record epoch losses
        epoch_loss += loss.detach()

        # update loss_dict
        for k, v in target_loss_dict.items():
            epoch_target_loss_dict[k] += v.detach().cpu().item()

        if iter % fix_update_iter == 0:
            with torch.no_grad():
                state_dict, student_state_dict = teacher_model.state_dict(), student_model.state_dict()
                for key, value in state_dict.items():
                    state_dict[key] = alpha_ema * value + (1 - alpha_ema) * student_state_dict[key].detach()
                teacher_model.load_state_dict(state_dict)

        # Data pre-fetch
        target_images, target_masks, _ = target_fetcher.next()
        if target_images is not None:
            target_teacher_images, target_student_images = target_images[0], target_images[1]

        # Log
        if is_main_process() and (iter + 1) % print_freq == 0:
            print('Teaching epoch ' + str(epoch) + ' : [ ' + str(iter + 1) + '/' + str(total_iters) + ' ] ' +
                  'total loss: ' + str(loss.detach().cpu().numpy()), flush=flush)

    # Final process of loss dict
    epoch_loss /= total_iters
    for k, v in epoch_target_loss_dict.items():
        epoch_target_loss_dict[k] /= total_iters
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Teaching epoch ' + str(epoch) + ' finished. Time cost: ' + total_time_str +
          ' Epoch loss: ' + str(epoch_loss.detach().cpu().numpy()), flush=flush)
    return epoch_loss, epoch_target_loss_dict


def train_one_epoch_upuk(student_model: torch.nn.Module,
                        teacher_model: torch.nn.Module,
                        criterion_pseudo: torch.nn.Module,
                        target_loader: DataLoader,
                        optimizer: torch.optim.Optimizer,
                        thresholds: List[float],
                        alpha_ema: float,
                        device: torch.device,
                        epoch: int,
                        clip_max_norm: float = 0.0,
                        print_freq: int = 20,
                        flush: bool = True,
                        fix_update_iter: int = 1):
    """
    teacher_model: NetB+NetC
    """
    import torch.nn.functional as F
    
    start_time = time.time()
    student_model.train()
    teacher_model.train()
    criterion_pseudo.train()
    target_fetcher = DataPreFetcher(target_loader, device=device)
    target_images, target_masks, _ = target_fetcher.next()
    target_teacher_images, target_student_images = target_images[0], target_images[1]
    # Record epoch losses
    epoch_loss = torch.zeros(1, dtype=torch.float, device=device, requires_grad=False)

    # Training data statistics
    epoch_target_loss_dict = defaultdict(float)
    total_iters = len(target_loader)
    
    num_sample = len(target_loader.dataset)
    
    # fea_bank = torch.randn(num_sample, 256)
    # score_bank = torch.randn(num_sample, 3).cuda()
    fea_bank = []
    score_bank = []
    
    
    
    for iter in range(total_iters):
        
                
                # fea_bank.append(output_norm.detach().clone().cpu())
                # score_bank.append(outputs.detach().clone().cpu())
                
            # fea_bank = torch.cat(fea_bank, dim=0)
            # score_bank = torch.cat(score_bank, dim=0)
        
        # Target teacher forward
        with torch.no_grad():
            teacher_out = teacher_model(target_teacher_images, target_masks)
            pseudo_labels = get_pseudo_labels(teacher_out['logit_all'][-1], teacher_out['boxes_all'][-1], thresholds)
            
            # out_dict = teacher_model(target_student_images, target_masks)
            tea_outputs = teacher_out['logit_all'][-1]
            # tea_output_f = out_dict['hidden_states_last']
            # tea_f_norm = F.normalize(tea_output_f)
            tea_prob = F.softmax(tea_outputs, dim=-1)

        
        # Target student forward
        target_student_out = student_model(target_student_images, target_masks)
        
        
        # outputs = teacher_model(inputs, image_mask)['logit_all'][-1]
        # output_f = teacher_model(inputs, image_mask)['hidden_states_last']
        # tea_f_norm = F.normalize(output_f)
        # tea_prob = F.softmax(dim=-1)(outputs)
        stu_outputs = target_student_out['logit_all'][-1]
        # stu_output_f = target_student_out['hidden_states_last'][-1]
        # stu_f_norm = F.normalize(stu_output_f)
        stu_prob = F.softmax(stu_outputs, dim=-1)
        
        # fa = a.flatten(start_dim=0, end_dim=1)
        upuk_loss_1 = torch.mean(F.kl_div(stu_prob, tea_prob, reduction='none'))
            # (stu_f_norm.flatten(start_dim=0, end_dim=1) @ stu_f_norm.flatten(start_dim=0, end_dim=1).T).diag()
        
        stu_scores = stu_prob.flatten(start_dim=0, end_dim=1)
        mask = torch.ones((stu_scores.shape[0], stu_scores.shape[0]), device=device)
        diag_num = torch.diag(mask)
        mask_diag = torch.diag_embed(diag_num)
        mask = mask - mask_diag
        copy = stu_scores.T

        dot_neg = stu_scores @ copy
        dot_neg = (dot_neg * mask).sum(-1)
        upuk_loss_2 = torch.mean(dot_neg)
        
        
        
        target_loss, target_loss_dict = criterion_pseudo(target_student_out, pseudo_labels)

        loss = target_loss + 0.2*(upuk_loss_1 + upuk_loss_2)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), clip_max_norm)
        optimizer.step()

        # Record epoch losses
        epoch_loss += loss.detach()

        # update loss_dict
        for k, v in target_loss_dict.items():
            epoch_target_loss_dict[k] += v.detach().cpu().item()

        if iter % fix_update_iter == 0:
            with torch.no_grad():
                state_dict, student_state_dict = teacher_model.state_dict(), student_model.state_dict()
                for key, value in state_dict.items():
                    state_dict[key] = alpha_ema * value + (1 - alpha_ema) * student_state_dict[key].detach()
                teacher_model.load_state_dict(state_dict)

        # Data pre-fetch
        target_images, target_masks, _ = target_fetcher.next()
        if target_images is not None:
            target_teacher_images, target_student_images = target_images[0], target_images[1]

        # Log
        if is_main_process() and (iter + 1) % print_freq == 0:
            print('Teaching epoch ' + str(epoch) + ' : [ ' + str(iter + 1) + '/' + str(total_iters) + ' ] ' +
                  'total loss: ' + str(loss.detach().cpu().numpy()), flush=flush)

    # Final process of loss dict
    epoch_loss /= total_iters
    for k, v in epoch_target_loss_dict.items():
        epoch_target_loss_dict[k] /= total_iters
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Teaching epoch ' + str(epoch) + ' finished. Time cost: ' + total_time_str +
          ' Epoch loss: ' + str(epoch_loss.detach().cpu().numpy()), flush=flush)
    return epoch_loss, epoch_target_loss_dict


def analysis_process(model: torch.nn.Module,
                             criterion: torch.nn.Module,
                             data_loader: DataLoader,
                             device: torch.device
                             ):
    """
    Train the standard detection model, using only labelled training set source.
    """
    start_time = time.time()
    model.eval()
    criterion.eval()
    fetcher = DataPreFetcher(data_loader, device=device)
    target_images, masks, annotations = fetcher.next()
    
    # target_fetcher = DataPreFetcher(target_loader, device=device)
    # target_images, target_masks, _ = target_fetcher.next()
    teacher_images, student_images = target_images[0], target_images[1]
    
    teacher_dict={}
    student_dict={}
    # Training statistics
    # epoch_loss = torch.zeros(1, dtype=torch.float, device=device, requires_grad=False)
    # epoch_loss_dict = defaultdict(float)
    def merging_dicts(agg_dict, iter_dict, domain):
        for cid in iter_dict.keys():
            if cid not in agg_dict.keys():
                agg_dict[cid] = {}
            for iter_key in iter_dict[cid].keys():
                if iter_key not in agg_dict[cid].keys():
                    agg_dict[cid][iter_key]=[iter_dict[cid][iter_key].cpu()]
                    # try:
                    #     agg_dict[cid][iter_key]=[iter_dict[cid][iter_key].cpu()]
                    # except:
                    #     print('iter_key:',iter_key)
                    #     print('Value:', iter_dict[cid][iter_key])
                else:
                    agg_dict[cid][iter_key].append(iter_dict[cid][iter_key].cpu())
        # print(f"Write Dict Info -{domain}-")
    
    for i in range(len(data_loader)):
        # if i>10:
        #     break
        
        # Forward
        with torch.no_grad():
            out = model(teacher_images, masks)
        # Loss
        #* loss, loss_dict = criterion(out, annotations)
        iter_tea_dict = criterion.analysis(teacher_images.cpu(), out, annotations)
        del out
        
        if iter_tea_dict is not None:
            # print('='*50)
            # print(f'[Iter: {i}]Image: Teacher Image')
            merging_dicts(teacher_dict, iter_tea_dict, 'Teacher')
            # print()
        
        
        # print('='*50)
        # print('Image: Student Image')
        # # Forward
        # out = model(student_images, masks)
        # iter_stu_dict = criterion.analysis(student_images.cpu(), out, annotations)
        # del out
        # if iter_stu_dict is not None:
        #     merging_dicts(student_dict, iter_stu_dict, 'Student')
        # print()
        
        
        target_images, masks, annotations = fetcher.next()
        if target_images is not None:
            teacher_images, student_images = target_images[0], target_images[1]
    
    text_detail = ['logit_candid_score', 'logit_gt_score', 'logit_candid_label_acc', 'hidden_state_intra_sim', 'hidden_state_bgd_sim', 'res_attn']
    for a_key in teacher_dict.keys():
        print(f'[Merge Result] [Categories: {a_key}]')
        for k,v in teacher_dict[a_key].items():
            # agg_value = torch.cat(v,dim=0)
            try:
                # 예외가 발생할 가능성이 있는 코드
                agg_value = torch.cat(v,dim=0)
            except:
                # 예외가 발생했을 때 실행할 코드
                # print('Potential Error: RuntimeError: zero-dimensional tensor (at position 0) cannot be concatenated')
                # print('Key:', k)
                agg_value = torch.tensor(v)
                
            if k in text_detail:
                if k == 'res_attn':
                    mask = torch.isnan(agg_value)
                    print(f'{k} (num: {len(v)}): {agg_value.shape} || Not nan: {(~mask).sum()} || AVG: {agg_value[~mask].mean()}')
                else:
                    print(f'{k} (num: {len(v)}): {agg_value.shape} || AVG: {agg_value.mean()}')
            else:
                print(f'{k} (num: {len(v)}): {agg_value.shape}')
        print()
    
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Finished. Time cost: ', total_time_str)
    
    torch.save(teacher_dict, '/data/pgh2874/SFOpen_suites/DRU/outputs/def-detr-base/SFUOD/city2foggy/teaching_standard_analysis/analysis_dict.pt')
    print('save info dicts..')

def train_one_epoch_teaching_mask(student_model: torch.nn.Module,
                                  teacher_model: torch.nn.Module,
                                  init_student_model: torch.nn.Module,
                                  criterion_pseudo: torch.nn.Module,
                                  criterion_pseudo_weak: torch.nn.Module,
                                  target_loader: DataLoader,
                                  optimizer: torch.optim.Optimizer,
                                  thresholds: List[float],
                                  coef_masked_img: float,
                                  alpha_ema: float,
                                  device: torch.device,
                                  epoch: int,
                                  keep_modules: List[str],
                                  clip_max_norm: float = 0.0,
                                  print_freq: int = 20,
                                  masking: Masking = None,
                                  flush: bool = True,
                                  fix_update_iter: int = 1,
                                  max_update_iter: int = 5,
                                  dynamic_update: bool = False,
                                  stu_buffer_cost: List[float] = None,
                                  stu_buffer_img: List[torch.Tensor] = None,
                                  stu_buffer_mask: List[torch.Tensor] = None,
                                  res_dict: dict = None,
                                  use_pseudo_label_weights: bool = False,
                                  use_loss_student: bool = False):
    """
    Train the student model with the teacher model, using only unlabeled training set target (plus masked target image)
    """
    start_time = time.time()
    student_model.train()
    teacher_model.train()
    init_student_model.train()
    criterion_pseudo.train()
    criterion_pseudo_weak.train()
    target_fetcher = DataPreFetcher(target_loader, device=device)
    target_images, target_masks, _ = target_fetcher.next()
    target_teacher_images, target_student_images = target_images[0], target_images[1]
    # Record epoch losses
    epoch_loss = torch.zeros(1, dtype=torch.float, device=device, requires_grad=False)

    # Training data statistics
    epoch_target_loss_dict = defaultdict(float)
    total_iters = len(target_loader)

    for iter in range(total_iters):
        # Target teacher forward
        with torch.no_grad():
            teacher_out = teacher_model(target_teacher_images, target_masks)
            pseudo_labels = get_pseudo_labels(teacher_out['logit_all'][-1], teacher_out['boxes_all'][-1], thresholds)

        # Target student forward
        target_student_out = student_model(target_student_images, target_masks)
        # loss from pseudo labels of current teacher
        target_loss, target_loss_dict = criterion_pseudo(target_student_out, pseudo_labels)

        # Masked target student forward
        masked_target_images = masking(target_student_images)
        masked_target_student_out = student_model(masked_target_images, target_masks)
        # loss from pseudo labels of current teacher
        masked_target_loss, masked_target_loss_dict = criterion_pseudo(masked_target_student_out, pseudo_labels)

        # Final loss
        loss = target_loss + coef_masked_img * masked_target_loss

        # Loss from pseudo labels of previous student (just testing, not used)
        # if use_loss_student:
        #     # Loss from pseudo labels of previous student
        #     with torch.no_grad():
        #         student_out = student_model(target_teacher_images, target_masks)
        #         pseudo_labels_student = get_pseudo_labels(student_out['logit_all'][-1], student_out['boxes_all'][-1],
        #                                                   thresholds)
        #     target_loss_student, target_loss_dict_student = criterion_pseudo_weak(target_student_out,
        #                                                                         pseudo_labels_student, use_pseudo_label_weights)
        #     masked_target_loss_student, masked_target_loss_dict_student = criterion_pseudo_weak(masked_target_student_out,
        #                                                                                       pseudo_labels_student, use_pseudo_label_weights)
        #
        #     # Final loss
        #     loss_student = target_loss_student + coef_masked_img * masked_target_loss_student
        #     loss += loss_student

        # Dynamic update EMA teacher : Create buffer cost and buffer image in student model
        if dynamic_update:
            with torch.no_grad():
                student_out = student_model(target_teacher_images, target_masks)
            # variance logit
            student_out_var = student_out['logit_all'].var(dim=0)
            var_total = student_out_var.mean().item()
            stu_buffer_cost.append(var_total)

            # Store batch data to buffer
            stu_buffer_img.append(target_teacher_images.clone().detach())
            stu_buffer_mask.append(target_masks.clone().detach())

            if len(stu_buffer_cost) == 1:
                with torch.no_grad():
                    init_student_model.load_state_dict(student_model.state_dict())

            if len(stu_buffer_cost) >= 1:
                with torch.no_grad():
                    init_student_out = init_student_model(target_teacher_images, target_masks)
                    pseudo_labels_init_student = get_pseudo_labels(init_student_out['logit_all'][-1], init_student_out['boxes_all'][-1],
                                                              thresholds)
                # Loss from pseudo labels of init student
                init_student_loss, init_student_loss_dict = criterion_pseudo_weak(target_student_out,
                                                                                    pseudo_labels_init_student, use_pseudo_label_weights)
                masked_init_student_loss, masked_init_student_loss_dict = criterion_pseudo_weak(masked_target_student_out,
                                                                                                  pseudo_labels_init_student, use_pseudo_label_weights)
                loss_init_student = init_student_loss + coef_masked_img * masked_init_student_loss
                loss += loss_init_student

        # Backward
        optimizer.zero_grad()
        loss.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), clip_max_norm)
        optimizer.step()

        # Record epoch losses
        epoch_loss += loss.detach()

        # update loss_dict
        for k, v in target_loss_dict.items():
            epoch_target_loss_dict[k] += v.detach().cpu().item()

        # Dynamic update EMA teacher : Update weight of teacher model
        if dynamic_update:
            if len(stu_buffer_cost) < max_update_iter:
                all_score = eval_stu(student_model, stu_buffer_img, stu_buffer_mask)
                compare_score = np.array(all_score) - np.array(stu_buffer_cost)
                # print(len(stu_buffer_cost), len(all_score), np.mean(compare_score<0))
                if np.mean(compare_score < 0) >= 0.5:
                    res_dict['stu_ori'].append(stu_buffer_cost)
                    res_dict['stu_now'].append(all_score)
                    res_dict['update_iter'].append(len(stu_buffer_cost))

                    df = pd.DataFrame(res_dict)
                    df.to_csv('dynamic_update.csv')

                    with torch.no_grad():
                        state_dict, student_state_dict = teacher_model.state_dict(), student_model.state_dict()
                        for key, value in state_dict.items():
                            state_dict[key] = alpha_ema * value + (1 - alpha_ema) * student_state_dict[key].detach()
                        teacher_model.load_state_dict(state_dict)

                    # Clear buffer
                    stu_buffer_cost = []
                    stu_buffer_img = []
                    stu_buffer_mask = []
            else:
                # print(len(stu_buffer_cost), 'Load previous student model weight')
                with torch.no_grad():
                    student_model = selective_reinitialize(student_model, init_student_model.state_dict(), keep_modules)

                # Clear buffer
                stu_buffer_cost = []
                stu_buffer_img = []
                stu_buffer_mask = []
        else:
            # EMA update teacher after fix iteration
            if iter % fix_update_iter == 0:
                with torch.no_grad():
                    state_dict, student_state_dict = teacher_model.state_dict(), student_model.state_dict()
                    for key, value in state_dict.items():
                        state_dict[key] = alpha_ema * value + (1 - alpha_ema) * student_state_dict[key].detach()
                    teacher_model.load_state_dict(state_dict)


        # Data pre-fetch
        target_images, target_masks, _ = target_fetcher.next()
        if target_images is not None:
            target_teacher_images, target_student_images = target_images[0], target_images[1]

        # Log
        if is_main_process() and (iter + 1) % print_freq == 0:
            print('Teaching epoch ' + str(epoch) + ' : [ ' + str(iter + 1) + '/' + str(total_iters) + ' ] ' +
                  'total loss: ' + str(loss.detach().cpu().numpy()), flush=flush)

    # Final process of loss dict
    epoch_loss /= total_iters
    for k, v in epoch_target_loss_dict.items():
        epoch_target_loss_dict[k] /= total_iters
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Teaching epoch ' + str(epoch) + ' finished. Time cost: ' + total_time_str +
          ' Epoch loss: ' + str(epoch_loss.detach().cpu().numpy()), flush=flush)
    return epoch_loss, epoch_target_loss_dict


@torch.no_grad()
def evaluate(model: torch.nn.Module,
             criterion: torch.nn.Module,
             data_loader_val: DataLoader,
             device: torch.device,
             print_freq: int,
             output_result_labels: bool = False,
             flush: bool = False,
             bi_attn: bool = True):
    start_time = time.time()
    model.eval()
    criterion.eval()
    if hasattr(data_loader_val.dataset, 'coco') or hasattr(data_loader_val.dataset, 'anno_file'):
        evaluator = CocoEvaluator(data_loader_val.dataset.coco)
        coco_data = json.load(open(data_loader_val.dataset.anno_file, 'r'))
        # dataset_annotations = [[] for _ in range(len(coco_data['images']))]
        dataset_annotations = defaultdict(list)
    else:
        raise ValueError('Unsupported dataset type.')
    epoch_loss = 0.0
    for i, (images, masks, annotations) in enumerate(data_loader_val):
        # To CUDA
        images = images.to(device)
        masks = masks.to(device)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        # Forward
        # out = model(images, masks, bi_attn=bi_attn)
        out = model(images, masks)
        logit_all, boxes_all = out['logit_all'], out['boxes_all']
        # Get pseudo labels
        if output_result_labels:
            #? results = get_pseudo_labels(logit_all[-1], boxes_all[-1], [0.4 for _ in range(9)])
            results = get_pseudo_labels(logit_all[-1], boxes_all[-1], [0.4 for _ in range(10)])
            for anno, res in zip(annotations, results):
                image_id = anno['image_id'].item()
                orig_image_size = anno['orig_size']
                img_h, img_w = orig_image_size.unbind(0)
                scale_fct = torch.stack([img_w, img_h, img_w, img_h])
                converted_boxes = convert_to_xywh(box_cxcywh_to_xyxy(res['boxes'] * scale_fct))
                converted_boxes = converted_boxes.detach().cpu().numpy().tolist()
                for label, box in zip(res['labels'].detach().cpu().numpy().tolist(), converted_boxes):
                    pseudo_anno = {
                        'id': 0,
                        'image_id': image_id,
                        'category_id': label,
                        'iscrowd': 0,
                        'area': box[-2] * box[-1],
                        'bbox': box
                    }
                    # dataset_annotations[image_id].append(pseudo_anno)
                    dataset_annotations[image_id].append(pseudo_anno)
        # Loss
        loss, loss_dict = criterion(out, annotations)
        epoch_loss += loss
        if is_main_process() and (i + 1) % print_freq == 0:
            print('Evaluation : [ ' + str(i + 1) + '/' + str(len(data_loader_val)) + ' ] ' +
                  'total loss: ' + str(loss.detach().cpu().numpy()), flush=flush)
        # mAP
        orig_image_sizes = torch.stack([anno['orig_size'] for anno in annotations], dim=0)
        results = post_process(logit_all[-1], boxes_all[-1], orig_image_sizes, 100)
        results = {anno['image_id'].item(): res for anno, res in zip(annotations, results)}
        evaluator.update(results)
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    aps = evaluator.summarize()
    epoch_loss /= len(data_loader_val)
    end_time = time.time()
    total_time_str = str(datetime.timedelta(seconds=int(end_time - start_time)))
    print('Evaluation finished. Time cost: ' + total_time_str, flush=flush)
    # Save results
    if output_result_labels:
        dataset_annotations_return = []
        id_cnt = 0
        # for image_anno in dataset_annotations:
        for image_anno in dataset_annotations.values():
            for box_anno in image_anno:
                box_anno['id'] = id_cnt
                id_cnt += 1
                dataset_annotations_return.append(box_anno)
        coco_data['annotations'] = dataset_annotations_return
        return aps, epoch_loss / len(data_loader_val), coco_data
    return aps, epoch_loss / len(data_loader_val)


def eval_stu(student_model: torch.nn.Module,
             stu_buffer_img: List[torch.Tensor],
             stu_buffer_mask: List[torch.Tensor]):
    """
    Evaluate student model with variance of logit
    """
    student_model.eval()
    all_score = []
    with torch.no_grad():
        for i in range(len(stu_buffer_img)):
            # student_out['logit_all']: [num_decoder_layers, batch size, num_queries, num_classes]
            student_out = student_model(stu_buffer_img[i], stu_buffer_mask[i])

            student_out_var = student_out['logit_all'].var(dim=0)
            var_total = student_out_var.mean().item()
            all_score.append(var_total)

    return all_score


#todo Visualization Code Implementation..
import matplotlib.pyplot as plt

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def plot_image(ax, img, norm):
    if norm:
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = (img * 255)
    img = img.astype('uint8')
    ax.imshow(img)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(out_bbox)
    return b

def plot_prediction(image, scores, boxes, labels, ax=None, plot_prob=True, cls_mapper=None):
    boxes = rescale_bboxes(boxes, list(image.shape[2:])[::-1])    
    # boxes = [rescale_bboxes(boxes[i], [w, h]).cpu() for i in range(len(boxes))]
    if ax is None:
        ax = plt.gca()
    plot_results(image[0].permute(1, 2, 0).detach().cpu().numpy(), scores, boxes, labels, ax, plot_prob=plot_prob, cls_mapper=cls_mapper)

def plot_results(pil_img, scores, boxes, labels, ax, plot_prob=True, norm=True, cls_mapper=None):
    from matplotlib import pyplot as plt
    h, w = pil_img.shape[:-1]
    image = plot_image(ax, pil_img, norm)
    colors = COLORS * 100
    # colors = ['darkorange', 'darkgreen', 'royalblue', 'red']
    unk_color= 'red'
    if boxes is not None:
        # boxes = [rescale_bboxes(boxes[i], [w, h]).cpu() for i in range(len(boxes))]
        for sc, cl, (xmin, ymin, xmax, ymax), c in zip(scores, labels, boxes, colors):
        # for sc, cl, box, c in zip(scores, labels, boxes.tolist(), colors):
            xmin, ymin, xmax, ymax = xmin.cpu(), ymin.cpu(), xmax.cpu(), ymax.cpu()
            # print('sc:', sc, sc.device)
            # print('cl:', cl, cl.device)
            # print('(xmin, ymin, xmax, ymax):', (xmin, ymin, xmax, ymax), (xmin.device, ymin.device, xmax.device, ymax.device))
            
            # ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
            #                            fill=False, color=c, linewidth=2))
            c_idx = cl.item()
            # if c_idx == 9:
            if c_idx == 21:
                c = 'red'
            else:
                c = 'yellow'
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                       fill=False, color=c, linewidth=5))
            
            cat = cls_mapper[cl.item()]['name']
            text = f'{cat}: {sc:0.2f}'
            # text = f'{cl}: {sc:0.2f}'
            #* ax.text(xmin, ymin, text, fontsize=10, bbox=dict(facecolor='yellow', alpha=0.5))
            ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))


@torch.no_grad()
def viz_engine(model: torch.nn.Module, criterion: torch.nn.Module, data_loader_val: DataLoader, device: torch.device):
    start_time = time.time()
    model.eval()
    criterion.eval()
    
    label_mapper = data_loader_val.dataset.coco.cats
    print('label_mapper', label_mapper)
    # out_layers_idx = -1
    # sample_list = torch.randperm(len(data_loader_val))[:150]
    
    # sample_list = torch.tensor([
    #     26,126,127,149,154,176,181,231,242,256,266,285,319,331,340,351,370,436,439,449,494,
    #     69,78,87,98,108,130,184,185,250,282,300,334,335,434,437
    # ])
    
    # sample_list = torch.tensor([3507, 5249, 6434, 7284, 236, 1920, 2034, 2072, 2697, 4350, 7460])
    # sample_list = torch.tensor([1421, 1092, 1279, 1287])
    # sample_list = torch.tensor([1624, 848, 7284, 171, 4040, 1338, 1421, 1792, 3577, 3601])
    # sample_list = torch.tensor([263, 1092, 1279, 1287, 1421, 1808, 3601, 5227, 5425])
    sample_list = torch.tensor([848,1338,1920,2637,3507])
    
    
    for i, (images, masks, annotations) in enumerate(data_loader_val):
        # if i > 50:
        # if i > 200:
            # break
        if i not in sample_list:
            continue
        print('index:', i)
        print("Annotations")
        print(annotations)
        print()
        # images = images[1]
        # target_fetcher = DataPreFetcher(target_loader, device=device)
        # target_images, target_masks, _ = target_fetcher.next()
        # target_teacher_images, target_student_images = target_images[0], target_images[1]
        
        
        # To CUDA
        images = images.to(device)
        masks = masks.to(device)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        # Forward
        out = model(images, masks)
        logit_all, boxes_all = out['logit_all'], out['boxes_all']
        
        # print('images:', images.shape)
        # print('out:', out.keys())
        # print('logits:', logit_all[-1].shape)
        # print('boxes:', boxes_all[-1].shape)
        # print('embeds:', out['resnet_1024_feat'].shape)
        
        # for anno in annotations:
            # s_var, s_cnt = anno['labels'].unique(return_counts=True)
            # for var, cnt in zip(s_var, s_cnt):
                # print('Category / Counts:', label_mapper[var.item()]['name'], cnt.item())
        
        #* indices, _ = criterion.matcher(logit_all[-1], boxes_all[-1], annotations)
        indices = criterion.matcher(logit_all[-1], boxes_all[-1], annotations)
        query_idx = criterion._get_src_permutation_idx(indices)[-1]
        # print('indices:', indices)
        # print('query_idx:', query_idx)
        
        # results = get_pseudo_labels(logit_all[-1], boxes_all[-1], [0.4 for _ in range(9)])
        # pseudo_labels.append({'scores': scores, 'labels': labels, 'boxes': boxes})
        # return pseudo_labels
        
        # pred_labels = logit_all[-1].squeeze(0)[query_idx]
        # pred_boxes = boxes_all[-1].squeeze(0)[query_idx]
        # print('pred_labels:', pred_labels.shape)
        # print('pred_boxes:', pred_boxes.shape)
        
        h,w = images.shape[2:]
        img_w = torch.tensor(w, device=device)
        img_h = torch.tensor(h, device=device)
        
        upsaple = torch.nn.Upsample(size=(img_h,img_w), mode='bilinear')
        img_c_feat = upsaple(out['resnet_1024_feat'])
        # print('img_c_feat:', img_c_feat.shape)
        # ori_feat_maps.append(img_c_feat.cpu())
        
        mean_feature_map = torch.sum(img_c_feat[0], 0) / img_c_feat[0].shape[0]  # Compute mean across channels
        # print('mean_feature_map:',mean_feature_map.shape)
        
        # img = images.squeeze(0).cpu().permute(1,2,0).numpy()
        
        # fig = plt.figure(figsize=(30, 30))
        fig = plt.figure(figsize=(30, 60))
        ax = fig.add_subplot(1, 2, 1)
        # plot_image(ax, img, True)
        # ft_size=
        
        plot_prediction(images.cpu(), torch.ones(annotations[0]['boxes'].shape[0]), annotations[0]['boxes'], annotations[0]['labels'], ax, plot_prob=False, cls_mapper=label_mapper)
        ax.axis("off")
        ax.set_title('GT', fontsize=30)
        
        
        ax = fig.add_subplot(1, 2, 2)
        # ax.imshow(mean_feature_map.cpu().numpy())
        # images.cpu()
        image = plot_image(ax, images[0].permute(1, 2, 0).detach().cpu().numpy(), True)
        # plot_results(image[0].permute(1, 2, 0).detach().cpu().numpy(), scores, boxes, labels, ax, plot_prob=plot_prob, cls_mapper=cls_mapper)
        
        
        
        pred_labels = logit_all[-1].squeeze(0)
        pred_boxes = boxes_all[-1].squeeze(0)
        pred_boxes = rescale_bboxes(pred_boxes, list(images.shape[2:])[::-1])    
        # print('pred_boxes:',pred_boxes.shape)
        # orig_image_sizes = torch.stack([anno['orig_size'] for anno in annotations], dim=0)
        # results = post_process(logit_all[-1], boxes_all[-1], orig_image_sizes, 100)
        # pred_labels = results[0]['labels']
        # pred_scores = results[0]['scores']
        # pred_boxes = results[0]['boxes']
        

        for labels, box_pos in zip(pred_labels, pred_boxes):
        # for res in results:
            # xmin, ymin, xmax, ymax = viz_boxes[0]
            # print('labels:', labels.shape)
            # print('sc:', sc.shape)
            # print('box_pos:', box_pos.shape)
            
            xmin, ymin, xmax, ymax = box_pos
            xmin, ymin, xmax, ymax = xmin.cpu(), ymin.cpu(), xmax.cpu(), ymax.cpu()
            
            probs = labels[1:].sigmoid()
            # probs = labels.sigmoid()
            # probs = labels[1:].sigmoid()
            # probs = labels.softmax(dim=0)[1:]
            cls_idx = probs.argmax()
            sc = probs[cls_idx]
            
            
            # if sc > 0.01:
            if sc > 0.6:
            # if sc > 0.1:
                if cls_idx.item()+1 == 21:
                # if cls_idx.item() == 9:
                    color='red'
                else:
                    color='yellow'
                # if cls_idx !=0:
                # ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                #                         fill=False, color=color, linewidth=3))
                ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                        fill=False, color=color, linewidth=5))
            
                cat = label_mapper[cls_idx.item()+1]['name']
                # cat = label_mapper[cls_idx.item()]['name']
                text = f'{cat}: {sc:0.2f}'
                ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))
        
        ax.axis("off")
        ax.set_title('Prediction', fontsize=30)
        
        ax = fig.add_subplot(2, 2, 1)
        image = plot_image(ax, images[0].permute(1, 2, 0).detach().cpu().numpy(), True)
        
        # ax.imshow(mean_feature_map.cpu().numpy())
        # boxes = rescale_bboxes(annotations[0]['boxes'], list(images.shape[2:])[::-1])    
        # for labels, box_pos in zip(annotations[0]['labels'], boxes):
            # xmin, ymin, xmax, ymax = box_pos
            # xmin, ymin, xmax, ymax = xmin.cpu(), ymin.cpu(), xmax.cpu(), ymax.cpu()
            # if labels.item() == 9:
                # color='red'
            # else:
                # color='yellow'
            # ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    # fill=False, color=color, linewidth=3))
        ax.axis("off")
        ax.set_title('Image', fontsize=30)
        
        
        pred_labels = logit_all[-1].squeeze(0)[query_idx]
        pred_boxes = boxes_all[-1].squeeze(0)[query_idx]
        
        ax = fig.add_subplot(2, 2, 2)
        ax.imshow(mean_feature_map.cpu().numpy())
        # pred_boxes = rescale_bboxes(pred_boxes, list(images.shape[2:])[::-1])    
        # for labels, box_pos in zip(pred_labels, pred_boxes):
        #     xmin, ymin, xmax, ymax = box_pos
        #     xmin, ymin, xmax, ymax = xmin.cpu(), ymin.cpu(), xmax.cpu(), ymax.cpu()
        #     probs = labels[1:].sigmoid()
        #     cls_idx = probs.argmax()
        #     if cls_idx.item()+1 == 9:
        #         color='red'
        #     else:
        #         color='yellow'
        #     ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
        #                             fill=False, color=color, linewidth=3))
        #     cat = label_mapper[cls_idx.item()+1]['name']
        #     sc = probs[cls_idx]
        #     text = f'{cat}: {sc:0.2f}'
        #     ax.text(xmin, ymin, text, fontsize=8, bbox=dict(facecolor='yellow', alpha=0.5))
        ax.axis("off")
        ax.set_title('Feature Map', fontsize=30)
        
        
        plt.savefig(f'/data/pgh2874/SFOpen_suites/DRU/exp_viz/MT_dior_V2/Known_viz_test_{i}.png', dpi=200)
        print(f'Done: viz_test_{i}.png')
        # plt.savefig(f'/data/pgh2874/SFOpen_suites/DRU/exp_viz/Baseline_Tch/viz_test_{i}.png', dpi=150)
                
        plt.cla()   # clear the current axes
        plt.clf()   # clear the current figure
        plt.close() # closes the current figure
        
        
        #* results = get_pseudo_labels(logit_all[-1], boxes_all[-1], [0.4 for _ in range(9)])
        
