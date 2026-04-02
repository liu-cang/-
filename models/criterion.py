import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits, l1_loss, mse_loss, kl_div
import torch.nn.functional as F

from torch.distributed import all_reduce
from torchvision.ops.boxes import nms
import math
from scipy.optimize import linear_sum_assignment

from utils.box_utils import box_cxcywh_to_xyxy, generalized_box_iou, box_iou
from utils.distributed_utils import is_dist_avail_and_initialized, get_world_size
from collections import defaultdict

import copy

class HungarianMatcher(nn.Module):

    def __init__(self,
                 coef_class: float = 2,
                 coef_bbox: float = 5,
                 coef_giou: float = 2,
                 alpha: float = 0.25,
                 gamma: float = 2.0,
                 iou_order_alpha: float = 4.0,
                 high_quality_matches: bool = False):
        """Creates the matcher

        Params:
            coef_class: This is the relative weight of the classification error in the matching cost
            coef_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            coef_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        
        self.coef_class = coef_class
        self.coef_bbox = coef_bbox
        self.coef_giou = coef_giou
        self.alpha = alpha
        self.gamma = gamma
        self.iou_order_alpha = iou_order_alpha
        self.high_quality_matches = high_quality_matches
        
        self.forward_k = 20
        
        assert coef_class != 0 or coef_bbox != 0 or coef_giou != 0, "all costs cant be 0"

    def forward(self, pred_logits, pred_boxes, annotations):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            annotations: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = pred_logits.shape[:2]

            # We flatten to compute the cost matrices in a batch
            pred_logits = pred_logits.flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
            pred_boxes = pred_boxes.flatten(0, 1)  # [batch_size * num_queries, 4]

            # Also concat the target labels and boxes
            gt_class = torch.cat([anno["labels"] for anno in annotations]).to(pred_logits.device)
            gt_boxes = torch.cat([anno["boxes"] for anno in annotations]).to(pred_logits.device)

            if self.high_quality_matches:
                class_score = pred_logits[:, gt_class]  # shape = [batch_size * num_queries, gt num within a batch]

                # # Compute iou
                bbox_iou, _ = box_iou(box_cxcywh_to_xyxy(pred_boxes), box_cxcywh_to_xyxy(gt_boxes))  # shape = [batch_size * num_queries, gt num within a batch]

                # Final cost matrix
                C = (-1) * (class_score * torch.pow(bbox_iou, self.iou_order_alpha))
            else:  # Default matching
                # Compute the classification cost.
                neg_cost_class = (1 - self.alpha) * (pred_logits ** self.gamma) * (-(1 - pred_logits + 1e-8).log())
                pos_cost_class = self.alpha * ((1 - pred_logits) ** self.gamma) * (-(pred_logits + 1e-8).log())
                cost_class = pos_cost_class[:, gt_class] - neg_cost_class[:, gt_class]

                # Compute the L1 cost between boxes
                cost_boxes = torch.cdist(pred_boxes, gt_boxes, p=1)

                # Compute the giou cost between boxes
                cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(pred_boxes), box_cxcywh_to_xyxy(gt_boxes))

                # Final cost matrix
                C = self.coef_bbox * cost_boxes + self.coef_class * cost_class + self.coef_giou * cost_giou
            
            
            # print('[Hungarian] C:', C.shape)
            C = C.view(bs, num_queries, -1).cpu()
            # print('[Hungarian] C after reshape:', C.shape)
            #todo Unmatched 찾아서?
            sizes = [len(anno["boxes"]) for anno in annotations]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            
            # pred_logits = pred_logits.reshape(bs, num_queries, -1)
            # unk_indices = []
            
            # for bidx in range(bs):
            #     forward_score = torch.sum(pred_logits[bidx], dim=1)
            #     # print(f'bidx:{bidx} / len(indices[bidx][0]: {len(indices[bidx][0])}')
            #     num_cand_unk = int(self.forward_k-len(indices[bidx][0])) if self.forward_k-len(indices[bidx][0]) >0 else 0
                
            #     _, forward_index = torch.topk(forward_score, num_cand_unk, largest=True, sorted=True)
            #     forward_index_list = forward_index.cpu().numpy().tolist()
            #     unknown_label = []
            #     for each in forward_index_list:
            #         # if each not in matched_qidx.cpu().numpy().tolist():
            #         if each not in indices[bidx][0].tolist():
            #             unknown_label.append(each)
                
            #     unknown_indices_batchi_a = torch.tensor(unknown_label)
            #     unknown_indices_batchi_b = torch.tensor([0] * len(unknown_label), dtype=torch.long)
            #     unk_indices.append((unknown_indices_batchi_a, unknown_indices_batchi_b))
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class SetCriterion(nn.Module):

    def __init__(self,
                 num_classes=9,
                 coef_class=2,
                 coef_boxes=5,
                 coef_giou=2,
                 alpha_focal=0.25,
                 alpha_dt=0.5,
                 gamma_dt=0.9,
                 max_dt=0.45,
                 device='cuda',
                 high_quality_matches=False):
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        self.matcher = HungarianMatcher(high_quality_matches=high_quality_matches)
        self.coef_class = coef_class
        self.coef_boxes = coef_boxes
        self.coef_giou = coef_giou
        self.alpha_focal = alpha_focal
        self.logits_sum = [torch.zeros(1, dtype=torch.float, device=device) for _ in range(num_classes)]
        self.logits_count = [torch.zeros(1, dtype=torch.int, device=device) for _ in range(num_classes)]
        self.alpha_dt = alpha_dt
        self.gamma_dt = gamma_dt
        self.max_dt = max_dt

    @staticmethod
    def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples. Default = -1 (no weighting).
            gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
        Returns:
            Loss tensor
        """
        prob = inputs.sigmoid()
        ce_loss = binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)
        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss.mean(1).sum() / num_boxes

    @staticmethod
    def sigmoid_quality_focal_loss(inputs, targets, scores, num_boxes, alpha: float = 0.25, gamma: float = 2):
        """
        Quality Focal Loss (QFL) is from `Generalized Focal Loss: Learning
        Qualified and Distributed Bounding Boxes for Dense Object Detection
         <https://arxiv.org/abs/2006.04388>`_.
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            scores: A float tensor with the same shape as targets: targets weighted by scores
                    (0 for the negative class and _score (0<_score<=1) for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples. Default = -1 (no weighting).
            gamma: Exponent of the modulating factor to
                balance easy vs hard examples.
        Returns:
            Loss tensor
        """
        prob = inputs.sigmoid()
        ce_loss = binary_cross_entropy_with_logits(inputs, scores, reduction="none")
        # p_t = prob * targets + (1 - prob) * (1 - targets)
        p_t = (scores - prob) * targets + prob * (1 - targets)
        loss = ce_loss * (abs(p_t) ** gamma)
        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss.mean(1).sum() / num_boxes

    # def loss_class(self, pred_logits, annotations, indices, num_boxes, use_pseudo_label_weights=False):
    def loss_class(self, pred_logits, annotations, indices, num_boxes, use_pseudo_label_weights=False):
        """Classification loss (NLL)
        annotations dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        idx = self._get_src_permutation_idx(indices)
        
        # if unk_indices is not None:
            # unk_idx = self._get_src_permutation_idx(unk_indices)
        # for i, anno in enumerate(annotations):
            # print('[loss class] idx:', i , 'anno[\'labels\']:', anno["labels"])
        
        gt_classes_o = torch.cat([anno["labels"][j] for anno, (_, j) in zip(annotations, indices)])
        gt_classes = torch.full(pred_logits.shape[:2], self.num_classes, dtype=torch.int64, device=pred_logits.device)
        gt_classes[idx] = gt_classes_o
        
        # if unk_indices is not None:
            # gt_classes[unk_idx] = self.num_classes - 1 #* 10 -1 = 9 (Unknown ID)

        gt_classes_onehot = torch.zeros([pred_logits.shape[0], pred_logits.shape[1], pred_logits.shape[2] + 1],
                                        dtype=pred_logits.dtype, layout=pred_logits.layout, device=pred_logits.device)
        gt_classes_onehot.scatter_(2, gt_classes.unsqueeze(-1), 1)
        gt_classes_onehot = gt_classes_onehot[:, :, :-1]

        if use_pseudo_label_weights:
            gt_scores_o = torch.cat([anno["scores"][j] for anno, (_, j) in zip(annotations, indices)])
            gt_scores = torch.full(pred_logits.shape[:2], 0.0, dtype=torch.float, device=pred_logits.device)
            gt_scores[idx] = gt_scores_o
            gt_scores_weight = gt_classes_onehot * gt_scores.unsqueeze(-1)
            loss_ce = self.sigmoid_quality_focal_loss(pred_logits, gt_classes_onehot, gt_scores_weight, num_boxes, alpha=self.alpha_focal, gamma=2) * pred_logits.shape[1]
        else:
            loss_ce = self.sigmoid_focal_loss(pred_logits, gt_classes_onehot, num_boxes, alpha=self.alpha_focal, gamma=2) * pred_logits.shape[1]

        return loss_ce

    def loss_boxes(self, pred_boxes, annotations, indices, num_boxes, use_pseudo_label_weights=False):
        """Compute the losses related to the bounding boxes: the L1 regression loss
           annotations dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The annotations boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        idx = self._get_src_permutation_idx(indices)
        src_boxes = pred_boxes[idx]
        gt_boxes = torch.cat([anno['boxes'][i] for anno, (_, i) in zip(annotations, indices)], dim=0)
        if use_pseudo_label_weights:
            gt_weights = torch.cat([anno['scores'][i] for anno, (_, i) in zip(annotations, indices)], dim=0)
            loss_bbox = l1_loss(src_boxes, gt_boxes, reduction='none') * gt_weights.unsqueeze(-1)
        else:
            loss_bbox = l1_loss(src_boxes, gt_boxes, reduction='none')
        return loss_bbox.sum() / num_boxes

    def loss_giou(self, pred_boxes, annotations, indices, num_boxes, use_pseudo_label_weights=False):
        """Compute the losses related to the bounding boxes: the gIoU loss
           annotations dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The annotations boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        idx = self._get_src_permutation_idx(indices)
        src_boxes = pred_boxes[idx]
        gt_boxes = torch.cat([anno['boxes'][i] for anno, (_, i) in zip(annotations, indices)], dim=0)
        if use_pseudo_label_weights:
            gt_weights = torch.cat([anno['scores'][i] for anno, (_, i) in zip(annotations, indices)], dim=0)
            loss_giou = 1 - torch.diag(generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(gt_boxes)))
            loss_giou = loss_giou * gt_weights
        else:
            loss_giou = 1 - torch.diag(generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(gt_boxes)))
        return loss_giou.sum() / num_boxes

    def record_positive_logits(self, logits, indices):
        idx = self._get_src_permutation_idx(indices)
        labels = logits[idx].argmax(dim=1)
        pos_logits = logits[idx].max(dim=1).values
        for label, logit in zip(labels, pos_logits):
            self.logits_sum[label] += logit
            self.logits_count[label] += 1

    def dynamic_threshold(self, thresholds):
        for s in self.logits_sum:
            all_reduce(s)
        for n in self.logits_count:
            all_reduce(n)
        logits_means = [s.item() / n.item() if n > 0 else 0.0 for s, n in zip(self.logits_sum, self.logits_count)]
        assert len(logits_means) == len(thresholds)
        new_thresholds = [self.gamma_dt * threshold + (1 - self.gamma_dt) * self.alpha_dt * math.sqrt(mean)
                          for threshold, mean in zip(thresholds, logits_means)]
        new_thresholds = [max(min(threshold, self.max_dt), 0.25) for threshold in new_thresholds]
        print('New Dynamic Thresholds: ', new_thresholds)
        return new_thresholds

    def clear_positive_logits(self):
        self.logits_sum = [torch.zeros(1, dtype=torch.float, device=self.device) for _ in range(self.num_classes)]
        self.logits_count = [torch.zeros(1, dtype=torch.int, device=self.device) for _ in range(self.num_classes)]

    @staticmethod
    def _get_src_permutation_idx(indices):
        # Permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    @staticmethod
    def _get_tgt_permutation_idx(indices):
        # Permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    @staticmethod
    def _discard_empty_labels(out, annotations):
        reserve_index = []
        for anno_idx in range(len(annotations)):
            if torch.numel(annotations[anno_idx]["boxes"]) != 0:
                reserve_index.append(anno_idx)
        for key, value in out.items():
            if key in ['logit_all', 'boxes_all']:
                out[key] = value[:, reserve_index, ...]
            elif key in ['features']:
                continue
            else:
                out[key] = value[reserve_index, ...]
        annotations = [annotations[idx] for idx in reserve_index]
        return out, annotations

    # def forward(self, samples, outputs, targets, epoch)
    def forward(self, out, annotations=None, use_pseudo_label_weights=False):
        logit_all = out['logit_all']
        boxes_all = out['boxes_all']
        
        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_boxes = sum(len(anno["labels"]) for anno in annotations) if annotations is not None else 0
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=logit_all.device)
        if is_dist_avail_and_initialized():
            all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        
        # Compute all the requested losses
        loss = torch.zeros(1).to(logit_all.device)
        loss_dict = defaultdict(float)
        num_decoder_layers = logit_all.shape[0]
        for i in range(num_decoder_layers):
            # Compute DETR losses
            if annotations is not None:
                indices = self.matcher(logit_all[i], boxes_all[i], annotations)
                # Compute the DETR losses
                loss_class = self.loss_class(logit_all[i], annotations, indices, num_boxes, use_pseudo_label_weights)
                loss_boxes = self.loss_boxes(boxes_all[i], annotations, indices, num_boxes, use_pseudo_label_weights)
                loss_giou = self.loss_giou(boxes_all[i], annotations, indices, num_boxes, use_pseudo_label_weights)
                loss_dict["loss_class"] += loss_class
                loss_dict["loss_boxes"] += loss_boxes
                loss_dict["loss_giou"] += loss_giou
                loss += self.coef_class * loss_class + self.coef_boxes * loss_boxes + self.coef_giou * loss_giou

        # Calculate average for all decoder layers
        loss /= num_decoder_layers
        for k, v in loss_dict.items():
            loss_dict[k] /= num_decoder_layers
        return loss, loss_dict



@torch.no_grad()
def post_process(pred_logits, pred_boxes, target_sizes, topk=100):
    """ Perform the computation
        Parameters:
            outputs -> pred_logits, pred_boxes: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
    """
    assert len(pred_logits) == len(target_sizes)
    assert target_sizes.shape[1] == 2

    prob = pred_logits.sigmoid()
    topk_values, topk_indexes = torch.topk(prob.view(pred_logits.shape[0], -1), topk, dim=1)
    scores = topk_values
    topk_boxes = torch.div(topk_indexes, pred_logits.shape[2], rounding_mode='trunc')
    labels = topk_indexes % pred_logits.shape[2]
    boxes = box_cxcywh_to_xyxy(pred_boxes)
    boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

    # From relative [0, 1] to absolute [0, height] coordinates
    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    boxes = boxes * scale_fct[:, None, :]
    results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
    return results

def get_topk_outputs(pred_logits, pred_boxes, topk=50):
    """
    Get top_k outputs from pred_logits and pred_boxes
    """
    prob = pred_logits.sigmoid()
    topk_values, topk_indexes = torch.topk(prob.view(pred_logits.shape[0], -1), topk, dim=1)
    topk_boxes = torch.div(topk_indexes, pred_logits.shape[2], rounding_mode='trunc')
    labels = topk_indexes % pred_logits.shape[2]

    # took_pred_boxes :[batch_size, topk, 4]
    topk_pred_boxes = torch.gather(pred_boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

    # topk_pred_logits : [batch_size, topk, num_classes]
    topk_pred_logits = torch.gather(pred_logits, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, pred_logits.shape[-1]))

    topk_outputs = {'labels_topk': labels, 'boxes_topk': topk_pred_boxes, 'logits_topk': topk_pred_logits}
    return topk_outputs

def get_pseudo_labels(pred_logits, pred_boxes, thresholds, is_nms=False, nms_threshold=0.7, unk_threshold=0.5):
    probs = pred_logits.sigmoid()
    # print('[Get_pseudo_labels] probs:', probs.shape)
    scores_batch, labels_batch = torch.max(probs, dim=-1)
    pseudo_labels = []
    thresholds_tensor = torch.tensor(thresholds, device=pred_logits.device)
    
    for scores, labels, pred_box in zip(scores_batch, labels_batch, pred_boxes):
        larger_idx = torch.gt(scores, thresholds_tensor[labels]).nonzero()[:, 0]
        scores, labels, boxes = scores[larger_idx], labels[larger_idx], pred_box[larger_idx, :]
        #todo pseudo unknown labeling
        unk_idx = scores<unk_threshold
        labels[unk_idx] = int(probs.shape[-1]-1)
        # unk_cnt += unk_idx.sum()
        # known_cnt += (~unk_idx).sum()
        #todo =======================================
        if is_nms:
            nms_idx = nms(box_cxcywh_to_xyxy(boxes), scores, iou_threshold=nms_threshold)
            scores, labels, boxes = scores[nms_idx], labels[nms_idx], boxes[nms_idx, :]
        pseudo_labels.append({'scores': scores, 'labels': labels, 'boxes': boxes})
        
    
    return pseudo_labels


def get_known_pseudo_labels(pred_logits, pred_boxes, thresholds, is_nms=False, nms_threshold=0.7, unk_threshold=0.3, unknown_warmup=False):
    #todo 1. Assign Known Pseudo Labels
    #todo 2. Set Known Confidence to 0.1 
    #todo 3. Assign Pseudo Labels to prediction having max confidence
    #todo 4. Return 1: Assigned Pseudo Label (Class, Score, Box)
    #todo 5. Return 2: Assigned pseudo index (Rest of them are used in Unknown Pseudo Labeling Process)
    # if unknown_warmup:
    #     known_cls_idx = torch.tensor([2,5,8,9], device=pred_logits.device)
    # else:
    #     known_cls_idx = torch.tensor([2,5,8], device=pred_logits.device)
    
    #* Original
    # known_cls_idx = torch.tensor([2,5,8], device=pred_logits.device)
    #* Reverse
    # known_cls_idx = torch.tensor([1,3,4,6,7], device=pred_logits.device)
    #* Unk - Overlapping Vehicle
    # known_cls_idx = torch.tensor([2,3,4,6,7], device=pred_logits.device)
    #* Unk - Night
    # known_cls_idx = torch.tensor([2,], device=pred_logits.device)
    
    #* Dior (Remote Sensing V1 [Objects: Airplane, Ship, Vehicle])
    # known_cls_idx = torch.tensor([i for i in range(1,21) if i not in [2, 4, 13]], device=pred_logits.device)
    
    #* Dior (Remote Sensing V2 [Small Architecture: Windmill, Chimney, StorageTank])
    known_cls_idx = torch.tensor([i for i in range(1,21) if i not in [3, 11, 17]], device=pred_logits.device)
    
    
    #* Original
    probs = pred_logits[:,:,known_cls_idx].sigmoid()
    scores_batch, labels_batch = torch.max(probs, dim=-1)
    
    # probs = pred_logits.sigmoid()
    # scores_batch, labels_batch = torch.max(probs, dim=-1)
    
    pseudo_labels = []
    known_batch_indices = []
    thresholds_tensor = torch.tensor(thresholds, device=pred_logits.device)
    for scores, labels, pred_box in zip(scores_batch, labels_batch, pred_boxes):
        # larger_idx = torch.gt(scores, thresholds_tensor[labels]).nonzero()[:, 0]
        larger_mask = torch.gt(scores, thresholds_tensor[labels])
        larger_idx = larger_mask.nonzero()[:, 0]
        
        #! Indexing Error Occured --> Larger Idx is not a Boolian Matrix!!
        known_batch_indices.append(larger_mask)
        
        #* Original
        scores, labels, boxes = scores[larger_idx], known_cls_idx[labels[larger_idx]], pred_box[larger_idx, :]
        # scores, labels, boxes = scores[larger_idx], labels[larger_idx], pred_box[larger_idx, :]
        
        # print('Known_larger_idx:', larger_idx)
        # print('Known_scores:', scores.shape)
        # print('Known_labels:', labels)
        # print('Known_boxes:', boxes.shape)
        #todo pseudo unknown labeling
        # unk_idx = scores<unk_threshold
        # labels[unk_idx] = int(probs.shape[-1]-1)
        #todo =======================================
        # if is_nms:
        #     nms_idx = nms(box_cxcywh_to_xyxy(boxes), scores, iou_threshold=nms_threshold)
        #     scores, labels, boxes = scores[nms_idx], labels[nms_idx], boxes[nms_idx, :]
        pseudo_labels.append({'scores': scores, 'labels': labels, 'boxes': boxes})
    return pseudo_labels, known_batch_indices

#* Ours-Best= 0.1
def get_unknown_pseudo_labels(pred_logits, pred_boxes, pred_embeds, k_batch_idx, is_nms=False, nms_threshold=0.7, unk_threshold=0.1):
    
    #* Ours
    # known_cls_idx = torch.tensor([2,5,8,9], device=pred_logits.device)
    #* Ours-Reverse
    # known_cls_idx = torch.tensor([1,3,4,6,7,9], device=pred_logits.device)
    #* Ours-Overlapping Vehicle
    # known_cls_idx = torch.tensor([2,3,4,6,7,9], device=pred_logits.device)
    #* Unk - Night
    # known_cls_idx = torch.tensor([2,9], device=pred_logits.device)
    
    #* Dior (Remote Sensing V1 [Objects: Airplane, Ship, Vehicle])
    # known_cls_idx = torch.tensor([i for i in range(1,22) if i not in [2, 4, 13]], device=pred_logits.device)
    
    #* Dior (Remote Sensing V2 [Small Architecture: Windmill, Chimney, StorageTank])
    known_cls_idx = torch.tensor([i for i in range(1,22) if i not in [3, 11, 17]], device=pred_logits.device)
    
    
    
    #todo Negative Energy
    # known_cls_idx = torch.tensor([2,5,8], device=pred_logits.device)
    # energies = torch.logsumexp(pred_logits[:, :, known_cls_idx], dim=-1) 
    #todo =====================================================
    
    #* Original
    probs = pred_logits[:, :, known_cls_idx].sigmoid()
    scores_batch, labels_batch = torch.max(probs, dim=-1)
    
    # probs = pred_logits.sigmoid()
    # scores_batch, labels_batch = torch.max(probs, dim=-1)
    
    known_embeds = []
    for batch_idx, k_obj_idx in enumerate(k_batch_idx):
        known_embeds.append(pred_embeds[batch_idx, k_obj_idx, :])
    
    known_embeds = torch.cat(known_embeds, dim=0)
    # print('known_embeds:', known_embeds.shape)
    pseudo_labels = []
    
    if len(known_embeds)>0:
        Uk, Sk, Vkh = torch.linalg.svd(known_embeds.T, full_matrices=True)
        # known_proj = Uk[:5]
        # known_proj = Uk[:50]
        sep = int(Uk.shape[0]/2)
        known_proj = Uk[:sep]
        
        # print('Sk:', Sk.shape)
        # softmax_Sk = torch.softmax(Sk, dim=0)
        # cumsum_Sk = torch.cumsum(softmax_Sk, dim=0)
        # print('cumsum_Sk')
        # print(cumsum_Sk)
        # 50% 이상이 되는 첫 번째 index
        # half_index = (cumsum_Sk >= 0.5).nonzero(as_tuple=True)[0][0].item()
        # known_proj = Uk[:5]
        # print('known_proj:',known_proj.shape)
        
        proj_known = known_embeds @ known_proj.T
        
        for scores, labels, pred_box, pred_embed, k_idx, prob in zip(scores_batch, labels_batch, pred_boxes, pred_embeds, k_batch_idx, probs):
        # for scores, labels, pred_box, pred_embed, k_idx, energy in zip(scores_batch, labels_batch, pred_boxes, pred_embeds, k_batch_idx, energies):
            #* larger_idx = torch.gt(scores, thresholds_tensor[labels]).nonzero()[:, 0]
            #* scores, labels, boxes = scores[larger_idx], labels[larger_idx], pred_box[larger_idx, :]
            #todo k_idx에 해당하지 않는 인덱스들 선별 
            scores, labels, boxes, embeds = scores[~k_idx], labels[~k_idx], pred_box[~k_idx, :], pred_embed[~k_idx,:]
            #todo =====================================
            #todo Objectness from the DETR embeddings
            prob = prob[~k_idx]
            
            proj_out = embeds @ known_proj.T

            obj_anchor = (F.normalize(proj_known, dim=-1) @ F.normalize(proj_known,dim=-1).T).mean()
            candid_score = (F.normalize(proj_out, dim=-1) @ F.normalize(proj_known,dim=-1).T).mean(dim=-1)   #* not_Known x Known -> not_Known
            unk_mask1 = candid_score >= obj_anchor
            unk_mask2 = prob[:,-1] >= unk_threshold
            # unk_mask2 = prob[:,-1] >= prob[:,-1].mean()
            # print('[Unknown Pseudo Labels]:', prob[:,-1].min(), prob[:,-1].max(), prob[:,-1].mean())
            unk_mask = unk_mask1 + unk_mask2
            #todo =====================================
            #todo Energy-based
            # rest_energy = energy[~k_idx]
            # known_energy_anc = energy[k_idx].mean()
            # unk_mask = rest_energy >= known_energy_anc
            #todo =====================================
            
            unk_scores, unk_labels, unk_boxes = scores[unk_mask], labels[unk_mask], boxes[unk_mask, :]
            
            unk_labels[:] = int(pred_logits.shape[-1]-1)
            #todo =======================================
            pseudo_labels.append({'scores': unk_scores, 'labels': unk_labels, 'boxes': unk_boxes})
                
        # print('unk_cnt:', unk_cnt)
        # print('known_cnt:', known_cnt)
    return pseudo_labels


def get_unknown_pseudo_labels_attn(img, pred_logits, pred_boxes, res_feats, k_batch_idx, is_nms=False, nms_threshold=0.7, unk_threshold=0.1):
    h,w = img.shape[-2:]
    img_w = torch.tensor(w, device=pred_logits.device)
    img_h = torch.tensor(h, device=pred_logits.device)
    upsaple = nn.Upsample(size=(img_h,img_w), mode='bilinear')
    
    res_feat = torch.mean(res_feats, 1)
    
    known_cls_idx = torch.tensor([2,5,8,9], device=pred_logits.device)
    probs = pred_logits[:, :, known_cls_idx].sigmoid()
    scores_batch, labels_batch = torch.max(probs, dim=-1)
    
    # known_embeds = []
    # for batch_idx, k_obj_idx in enumerate(k_batch_idx):
    #     known_embeds.append(pred_embeds[batch_idx, k_obj_idx, :])
    # known_embeds = torch.cat(known_embeds, dim=0)
    
    
    
    # if len(known_embeds)>0:
        # Uk, Sk, Vkh = torch.linalg.svd(known_embeds.T, full_matrices=True)
        # known_proj = Uk[:5]
        # proj_known = known_embeds @ known_proj.T
    attn_scores=[]
    for b_idx in range(img.shape[0]):
        boxes = box_cxcywh_to_xyxy(pred_boxes[b_idx])
        bb = boxes.cpu() * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        
        img_feat = upsaple(res_feat[b_idx].unsqueeze(0).unsqueeze(0))
        img_feat = img_feat.squeeze(0).squeeze(0)
        means_bb = torch.zeros(boxes.shape[0]).to(pred_logits)
        for j in range(bb.shape[0]):
            xmin = bb[j,:][0].long()
            ymin = bb[j,:][1].long()
            xmax = bb[j,:][2].long()
            ymax = bb[j,:][3].long()
            
            means_bb[j] = torch.mean(img_feat[ymin:ymax,xmin:xmax])
            if torch.isnan(means_bb[j]):
                means_bb[j] = -10e10
        
        attn_scores.append(means_bb)
    attn_scores = torch.stack(attn_scores)
    pseudo_labels = []
    for scores, labels, pred_box, attn_score, k_idx, prob in zip(scores_batch, labels_batch, pred_boxes, attn_scores, k_batch_idx, probs):
        scores, labels, boxes, attn_sc = scores[~k_idx], labels[~k_idx], pred_box[~k_idx, :], attn_score[~k_idx]
        prob = prob[~k_idx]
        
        # proj_out = embeds @ known_proj.T
        # obj_anchor = (F.normalize(proj_known, dim=-1) @ F.normalize(proj_known,dim=-1).T).mean()
        # candid_score = (F.normalize(proj_out, dim=-1) @ F.normalize(proj_known,dim=-1).T).mean(dim=-1)   #* not_Known x Known -> not_Known
        # unk_mask1 = candid_score >= obj_anchor
        # unk_mask2 = prob[:,-1] >= unk_threshold
        # unk_mask = unk_mask1 + unk_mask2
        
        _, unk_idx =  torch.topk(attn_sc, k_idx.sum())
        
        unk_scores, unk_labels, unk_boxes = scores[unk_idx], labels[unk_idx], boxes[unk_idx, :]
        
        unk_labels[:] = int(pred_logits.shape[-1]-1)
        #todo =======================================
        pseudo_labels.append({'scores': unk_scores, 'labels': unk_labels, 'boxes': unk_boxes})
            
    return pseudo_labels



class Set_UNK_Criterion(nn.Module):
    def __init__(self,
                 num_classes=9,
                 coef_class=2,
                 coef_boxes=5,
                 coef_giou=2,
                 alpha_focal=0.25,
                 alpha_dt=0.5,
                 gamma_dt=0.9,
                 max_dt=0.45,
                 device='cuda',
                 high_quality_matches=False):
        super().__init__()
        self.num_classes = num_classes
        self.device = device
        self.matcher = HungarianMatcher(high_quality_matches=high_quality_matches)
        self.coef_class = coef_class
        self.coef_boxes = coef_boxes
        self.coef_giou = coef_giou
        self.alpha_focal = alpha_focal
        self.logits_sum = [torch.zeros(1, dtype=torch.float, device=device) for _ in range(num_classes)]
        self.logits_count = [torch.zeros(1, dtype=torch.int, device=device) for _ in range(num_classes)]
        self.alpha_dt = alpha_dt
        self.gamma_dt = gamma_dt
        self.max_dt = max_dt

    @staticmethod
    def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples. Default = -1 (no weighting).
            gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
        Returns:
            Loss tensor
        """
        prob = inputs.sigmoid()
        ce_loss = binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)
        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss.mean(1).sum() / num_boxes

    @staticmethod
    def sigmoid_quality_focal_loss(inputs, targets, scores, num_boxes, alpha: float = 0.25, gamma: float = 2):
        """
        Quality Focal Loss (QFL) is from `Generalized Focal Loss: Learning
        Qualified and Distributed Bounding Boxes for Dense Object Detection
         <https://arxiv.org/abs/2006.04388>`_.
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            scores: A float tensor with the same shape as targets: targets weighted by scores
                    (0 for the negative class and _score (0<_score<=1) for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples. Default = -1 (no weighting).
            gamma: Exponent of the modulating factor to
                balance easy vs hard examples.
        Returns:
            Loss tensor
        """
        prob = inputs.sigmoid()
        ce_loss = binary_cross_entropy_with_logits(inputs, scores, reduction="none")
        # p_t = prob * targets + (1 - prob) * (1 - targets)
        p_t = (scores - prob) * targets + prob * (1 - targets)
        loss = ce_loss * (abs(p_t) ** gamma)
        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss.mean(1).sum() / num_boxes

    # def loss_class(self, pred_logits, annotations, indices, num_boxes, use_pseudo_label_weights=False):
    def loss_class(self, pred_logits, annotations, indices, num_boxes, use_pseudo_label_weights=False):
        """Classification loss (NLL)
        annotations dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        idx = self._get_src_permutation_idx(indices)
        
        gt_classes_o = torch.cat([anno["labels"][j] for anno, (_, j) in zip(annotations, indices)])
        gt_classes = torch.full(pred_logits.shape[:2], self.num_classes, dtype=torch.int64, device=pred_logits.device)
        gt_classes[idx] = gt_classes_o
        
        gt_classes_onehot = torch.zeros([pred_logits.shape[0], pred_logits.shape[1], pred_logits.shape[2] + 1],
                                        dtype=pred_logits.dtype, layout=pred_logits.layout, device=pred_logits.device)
        gt_classes_onehot.scatter_(2, gt_classes.unsqueeze(-1), 1)
        gt_classes_onehot = gt_classes_onehot[:, :, :-1]
        
        #? ============================================================================================================
        gt_classes_score = torch.cat([anno["scores"][j].float() for anno, (_, j) in zip(annotations, indices)])
        gt_scores = torch.full(pred_logits.shape[:2], 0., dtype=torch.float, device=pred_logits.device)
        gt_scores[idx] = gt_classes_score
        
        fir, sec, idx = gt_classes_onehot.nonzero(as_tuple=True)
        mask = idx!=9
        
        gt_classes_onehot[fir[mask], sec[mask], idx[mask]] = gt_scores[fir[mask], sec[mask]]
        gt_classes_onehot[fir[mask], sec[mask], -1] = 1. - gt_scores[fir[mask], sec[mask]]
        #? ============================================================================================================
        
        if use_pseudo_label_weights:
            gt_scores_o = torch.cat([anno["scores"][j] for anno, (_, j) in zip(annotations, indices)])
            gt_scores = torch.full(pred_logits.shape[:2], 0.0, dtype=torch.float, device=pred_logits.device)
            gt_scores[idx] = gt_scores_o
            gt_scores_weight = gt_classes_onehot * gt_scores.unsqueeze(-1)
            loss_ce = self.sigmoid_quality_focal_loss(pred_logits, gt_classes_onehot, gt_scores_weight, num_boxes, alpha=self.alpha_focal, gamma=2) * pred_logits.shape[1]
        else:
            loss_ce = self.sigmoid_focal_loss(pred_logits, gt_classes_onehot, num_boxes, alpha=self.alpha_focal, gamma=2) * pred_logits.shape[1]

        return loss_ce
    
    def class_refinements(self, pred_logits, pred_hs, pred_wts, pred_boxes, res_feats, annotations, indices, num_boxes):
        #* class idx: 2 (car), 5 (truck), 8 (bus), 9 (unknown)
        cls_idx = torch.tensor([2,5,8,9], device=pred_logits.device)
        # print('res_feats:',res_feats.shape)
        idx = self._get_src_permutation_idx(indices)
        # print('idx:', idx)
        
        # gt_classes_o = torch.cat([anno["labels"][j] for anno, (_, j) in zip(annotations, indices)])
        
        src_logits = pred_logits[idx][:,cls_idx]
        feat_mag = torch.norm(pred_hs[idx], dim=1)
        wts_mag = torch.norm(pred_wts[cls_idx],dim=1)
        
        # src_boxes = pred_boxes[idx]
        # print('pred_boxes:', src_boxes.shape, src_boxes)
        
        # obj_mask = torch.full(pred_logits.shape[:2], 0, dtype=torch.int64, device=pred_logits.device).bool()
        # obj_mask[idx] = True
        
        # print('gt:',gt_classes_o)
        # print('src_logits:',src_logits.shape)
        # print('feat_mag:',feat_mag.shape, feat_mag)
        # print('wts_mag:',wts_mag.shape, wts_mag)
        # obj_mag = pred_hs[obj_mask]
        # bgd_mag = pred_hs[obj_mask]
        
        
        #! Layer Norm 영향으로 전부 비슷함
        # obj_mag = torch.norm(pred_hs[obj_mask],dim=1).mean()
        # print('obj_mag:',obj_mag)
        # bgd_mag = torch.norm(pred_hs[~obj_mask], dim=1)
        # top_val, _ = bgd_mag.topk(k=10)
        # bot_val, _ = bgd_mag.topk(k=10,largest=False)
        # print('top_val:', top_val.mean())
        # print('bot_val:', bot_val.mean())
        
        denom= feat_mag.unsqueeze(1) @ wts_mag.unsqueeze(0)
        
        wts_logits = src_logits / feat_mag.unsqueeze(1)
        
        cos_logits = src_logits / denom
        
        loss_debias = torch.dist(wts_logits,cos_logits.detach())
        # loss_debias = kl_div(wts_logits.log_softmax(dim=1), cos_logits.detach().softmax(dim=1), reduction='batchmean')
        
        return loss_debias
    
    def objectness(self, samples, pred_logits, pred_hs, pred_wts, pred_boxes, res_feats, annotations, indices, num_boxes):
        #* class idx: 2 (car), 5 (truck), 8 (bus), 9 (unknown)
        cls_idx = torch.tensor([2,5,8,9], device=pred_logits.device)
        
        iter_dict = {}
        
        # print('samples:',samples.shape) #* Batch, C, H, W
        # img = samples.cpu().numpy()
        
        h, w = samples.shape[-2:]
        img_w = torch.tensor(w, device=pred_logits.device)
        img_h = torch.tensor(h, device=pred_logits.device)
        # img_w = torch.tensor(w)
        # img_h = torch.tensor(h)
        
        upsaple = nn.Upsample(size=(img_h,img_w), mode='bilinear')
        
        # print('res_feats:', res_feats.shape)
        res_feat = torch.mean(res_feats, 1)
        # print('res_feat:', res_feat.shape)
        
        
        # print('indices:', indices)
        idx = self._get_src_permutation_idx(indices)
        # print('idx:', idx)
        
        # src_logits = pred_logits[idx][:,cls_idx]
        # feat_mag = torch.norm(pred_hs[idx], dim=1)
        wts_mag = torch.norm(pred_wts[cls_idx],dim=1)
        wts_sim = (pred_wts[cls_idx]/wts_mag.unsqueeze(1)) @ (pred_wts[cls_idx]/wts_mag.unsqueeze(1)).T
        
        # print('Classifier Magnitude:', wts_mag)
        # print('Classifier Similarity:', wts_sim)
        # print()
        
        gt_classes_o = torch.cat([anno["labels"][j] for anno, (_, j) in zip(annotations, indices)])
        unique_class_id = gt_classes_o.unique()
        for cid in unique_class_id:
            cid_mask = torch.nonzero(gt_classes_o == cid).squeeze(1)
            iter_dict[cid.item()] = {'class_mask': cid_mask}
            # print('category:',cid.item(), iter_dict[cid.item()]['class_mask'].shape)
        
        # print('GT:', gt_classes_o.shape, gt_classes_o)
        gt_known = gt_classes_o<9
        # print('GT-Known:', gt_classes_o[gt_known].shape)
        # print('GT-Unknown:', gt_classes_o[~gt_known].shape)
        # print()
        gt_classes = torch.full(pred_logits.shape[:2], -1, dtype=torch.int64, device=pred_logits.device)
        gt_classes[idx] = gt_classes_o
        
        probs = pred_logits[idx].sigmoid()
        
        
        
        k_confidence=[]
        u_confidence=[]
        max_confidence=[]
        pseudo_label = []
        for prob, gt in zip(probs, gt_classes_o):
            score = prob[gt]
            # scores_batch, labels_batch = torch.max(probs, dim=-1)
            max_score, max_label = torch.max(prob, dim=0)
            
            if max_score < 0.5:
                candid_label = 9
            else:
                candid_label = max_label.long().item()
            
            if 'logit_gt_score' not in  iter_dict[gt.item()].keys():
                iter_dict[gt.item()]['logit_gt_score'] = [score]
            else:
                iter_dict[gt.item()]['logit_gt_score'].append(score)
            
            if 'logit_candid_score' not in  iter_dict[gt.item()].keys():
                iter_dict[gt.item()]['logit_candid_score'] = [max_score]
            else:
                iter_dict[gt.item()]['logit_candid_score'].append(max_score)
                
            if 'logit_candid_label' not in  iter_dict[gt.item()].keys():
                iter_dict[gt.item()]['logit_candid_label'] = [candid_label]
            else:
                iter_dict[gt.item()]['logit_candid_label'].append(candid_label)
            
                
            # if gt < 9:
            #     # iter_dict[cid.item()] = {'class_mask': cid_mask}
            #     k_confidence.append(score)
            # else:
            #     u_confidence.append(score)
            # max_confidence.append(max_score)
            # pseudo_label.append(candid_label)
        
        #* Naive Pseudo Labeling Analysis
        for key in iter_dict.keys():
            iter_dict[key]['logit_gt_score'] = torch.stack(iter_dict[key]['logit_gt_score'], dim=0).cpu()
            iter_dict[key]['logit_candid_score'] = torch.stack(iter_dict[key]['logit_candid_score'], dim=0).cpu()
            iter_dict[key]['logit_candid_label'] = torch.tensor(iter_dict[key]['logit_candid_label']).cpu()
            
            # c_candid = torch.tensor(iter_dict[key]['logit_candid_label'], device=gt_classes.device)
            
            c_mask = iter_dict[key]['class_mask']
            correct = gt_classes_o[c_mask] == iter_dict[key]['logit_candid_label'].to(gt_classes.device)
            label_acc = correct.sum()/c_mask.shape[0]
            # print(f'[Category:{key}] Label ACC:', (c_acc.sum()/c_mask.shape[0]))
            
            # if 'logit_candid_label_acc' not in  iter_dict[key].keys():
            #     iter_dict[key]['logit_candid_label_acc'] = [label_acc]
            # else:
            #     iter_dict[key]['logit_candid_label_acc'].append(label_acc)
            
            iter_dict[key]['logit_candid_label_acc'] = label_acc
            
        # k_confidence = torch.stack(k_confidence,dim=0)
        # u_confidence = torch.stack(u_confidence,dim=0)
        # max_confidence = torch.stack(max_confidence,dim=0)
        # pseudo_label = torch.tensor(pseudo_label, device=gt_classes.device)
        
        # print('k_confidence:', k_confidence.mean())
        # print('u_confidence:', u_confidence.mean())
        # print('real_label', gt_classes_o)
        # print('pseudo_label:', pseudo_label)
        # print('pseudo_confidence:', max_confidence.mean())
        # print()
        
        #*for key in iter_dict.keys():
            #*c_mask = iter_dict[key]['class_mask']
            #*c_acc = gt_classes_o[c_mask] == iter_dict[key]['logit_candid_label']
            #*print(f'[Category:{key}] Label ACC:', (c_acc.sum()/c_mask.shape[0]))
        
        # k_acc_mask = gt_classes_o<9
        # print('k_acc_mask:',k_acc_mask.sum())
        # print('u_acc_mask:',(~k_acc_mask).sum())
        # k_acc = gt_classes_o[k_acc_mask] == pseudo_label[k_acc_mask]
        # u_acc = gt_classes_o[~k_acc_mask] == pseudo_label[~k_acc_mask]
        # #todo Class 별 Acc 추가 
        # print('ACC-Known Label:', (k_acc.sum()/k_acc_mask.sum()))
        # print('ACC-Unknown Label:', (u_acc.sum()/(~k_acc_mask).sum()))
        # print()
        
        # print('='*50)
        # print()
        
        # known_mask = torch.logical_and(gt_classes>0, gt_classes<9) 
        # print('known_mask:', known_mask.shape)
        # print('known_mask[0]:', known_mask[0].sum())
        # print('known_mask[1]:', known_mask[1].sum())
        # print()
        # unknown_mask = gt_classes==9
        # print('unknown_mask:',unknown_mask.shape)
        # print('unknown_mask[0]:',unknown_mask[0].sum())
        # print('unknown_mask[1]:',unknown_mask[1].sum())
        # print()
        
        bgd_mask = gt_classes==-1
        
        
        obj_mask = torch.full(pred_logits.shape[:2], 0, dtype=torch.int64, device=pred_logits.device).bool()
        obj_mask[idx] = True
        
        
        # print('obj_mask:',obj_mask.shape)
        # print('obj_mask[0]:',obj_mask[0].sum())
        # print('obj_mask[1]:',obj_mask[1].sum())
        # print()
        
        
        
        #todo Analysis Hidden State 
        #todo pred_hs (크기는 Normalize 되어 있어서 의미가 없음)
        #* cid_mask = torch.nonzero(gt_classes_o == cid).squeeze(1)
        #* iter_dict[cid.item()] = {'class_mask': cid_mask}
        src_hs = pred_hs[idx]
        bgd_hs = pred_hs[bgd_mask]
        ridx = torch.randperm(bgd_hs.shape[0])[:10]
        
        
        src_known_logits = pred_logits[idx][:, cls_idx]
        # print('src_known_logits:',src_known_logits.shape)
        src_bgd_logits = pred_logits[bgd_mask][ridx][:,cls_idx]
        
        for logits, gt in zip(src_known_logits, gt_classes_o):
            if 'src_logits' not in  iter_dict[gt.item()].keys():
                iter_dict[gt.item()]['src_logits'] = [logits]
            else:
                iter_dict[gt.item()]['src_logits'].append(logits)
                
        for key in iter_dict.keys():
            iter_dict[key]['src_logits'] = torch.stack(iter_dict[key]['src_logits'], dim=0).cpu()
            if key ==9:
                iter_dict[key]['src_bgd_logits'] = src_bgd_logits.cpu()
            
        
        for key in iter_dict.keys():
            c_mask = iter_dict[key]['class_mask']
            iter_dict[key]['hidden_state'] = src_hs[c_mask].cpu()
            # print(f'[Category:{key}] Hidden_State:', iter_dict[key]['hidden_state'].shape)
            
            hs_intra_known_sim = F.normalize(src_hs[c_mask],dim=-1) @ F.normalize(src_hs[c_mask],dim=-1).T
            iter_dict[key]['hidden_state_intra_sim'] = hs_intra_known_sim.mean().cpu()
            # print(f'[Category:{key}] HS-Intra Similarity:', iter_dict[key]['hidden_state_intra_sim'])
            
            hs_obj_bgd_sim = F.normalize(src_hs[c_mask],dim=-1) @ F.normalize(pred_hs[~obj_mask],dim=-1).T
            iter_dict[key]['hidden_state_bgd_sim'] = hs_obj_bgd_sim.mean().cpu()
            
            if key ==9:
                iter_dict[key]['bgd_hidden_state'] = bgd_hs[ridx].cpu()
        
        # known_hs = pred_hs[known_mask]
        # unknown_hs = pred_hs[unknown_mask]
        # print('known_hs:', known_hs.shape)
        # print('unknown_hs:', unknown_hs.shape)
        
        # hs_intra_known_sim = F.normalize(known_hs,dim=-1) @ F.normalize(known_hs,dim=-1).T
        # print('HS: Known intra similarity:', hs_intra_known_sim.mean())
        # hs_intra_unknown_sim = F.normalize(unknown_hs,dim=-1) @ F.normalize(unknown_hs,dim=-1).T
        # print('HS: Unknown intra similarity:', hs_intra_unknown_sim.mean())
        
        # hs_sim = F.normalize(known_hs,dim=-1) @ F.normalize(unknown_hs,dim=-1).T
        # print('HS: K-U similarity:', hs_sim.mean())
        # hs_sim_obj = F.normalize(pred_hs[obj_mask],dim=-1) @ F.normalize(pred_hs[~obj_mask],dim=-1).T
        # print('HS: OBJ-BGD similarity:', hs_sim_obj.mean())
        
        
        # bgd_mask = ~obj_mask
        
        res_embeds = []
        res_attn_scores = []
        bgd_res_embeds = []
        bgd_res_attn_scores = []
        for b_idx in range(samples.shape[0]):
            # print('obj_mask[0]:',obj_mask[0].sum())
            obj_box_mask = obj_mask[b_idx]
            bgd_box_mask = (~obj_mask)[b_idx]
            #* bgd_mask = ~obj_mask
            #todo Background 영역에 대한 Res_Embedding 출력필요
            #todo Unknown과 Background를 구별할 수 있을지 확인 필요
            
            # unmatched_boxes = box_cxcywh_to_xyxy(src_boxes)
            # boxes = pred_boxes[obj_mask[0]]
            #* boxes = box_cxcywh_to_xyxy(pred_boxes[b_idx])
            boxes = box_cxcywh_to_xyxy(pred_boxes[b_idx][obj_box_mask])
            bgd_boxes = pred_boxes[b_idx][bgd_box_mask]
            ridx = torch.randperm(bgd_boxes.shape[0])[:5]
            bgd_boxes = box_cxcywh_to_xyxy(bgd_boxes[ridx])
            
            # boxes = rescale_boxes
            # print(f'Batch_idx:{b_idx} / boxes:', boxes.shape, boxes[:3])
            # bb = boxes * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(pred_logits.device)
            bb = boxes.cpu() * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
            bgd_bb = bgd_boxes.cpu() * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
            # print(f'Batch_idx:{b_idx} / Scaling boxes:', bb.shape, bb[:3])
            if bb.shape[0] ==0:
                continue
            
            img_feat = upsaple(res_feat[b_idx].unsqueeze(0).unsqueeze(0))
            img_feat = img_feat.squeeze(0).squeeze(0)
            # print(f'Batch_idx:{b_idx} / img_feat:', img_feat.shape)
            
            # ch_img_feat = upsaple(res_feats[b_idx].unsqueeze(0).cpu())
            ch_img_feat = upsaple(res_feats[b_idx].unsqueeze(0))
            ch_img_feat = ch_img_feat.squeeze(0)
            # print(f'Batch_idx:{b_idx} / ch_img_feat:', ch_img_feat.shape)
            
            # means_bb = torch.zeros(queries.shape[0]).to(unmatched_boxes)
            
            obj_res_vecs = []
            attn_scores = []
            for j in range(bb.shape[0]):
                # if j>5:
                    # break
                xmin = bb[j,:][0].long()
                ymin = bb[j,:][1].long()
                xmax = bb[j,:][2].long()
                ymax = bb[j,:][3].long()
                
                attn_score = torch.mean(img_feat[ymin:ymax,xmin:xmax])
                attn_scores.append(attn_score)
                ch_crop_feat = ch_img_feat[:, ymin:ymax,xmin:xmax]
                C,H,W = ch_crop_feat.shape
                ch_crop_feat = ch_crop_feat.permute(1,2,0).reshape(H*W,C).mean(dim=0)
                obj_res_vecs.append(ch_crop_feat)
            attn_scores = torch.stack(attn_scores)
            obj_res_vecs = torch.stack(obj_res_vecs)    #* Query, C (=1024)
            res_attn_scores.append(attn_scores)
            res_embeds.append(obj_res_vecs)
            
            
            bgd_res_vecs = []
            bgd_attn_scores = []
            for j in range(bgd_bb.shape[0]):
                # if j>5:
                    # break
                xmin = bgd_bb[j,:][0].long()
                ymin = bgd_bb[j,:][1].long()
                xmax = bgd_bb[j,:][2].long()
                ymax = bgd_bb[j,:][3].long()
                
                bgd_attn_score = torch.mean(img_feat[ymin:ymax,xmin:xmax])
                bgd_attn_scores.append(bgd_attn_score)
                ch_crop_feat = ch_img_feat[:, ymin:ymax,xmin:xmax]
                C,H,W = ch_crop_feat.shape
                ch_crop_feat = ch_crop_feat.permute(1,2,0).reshape(H*W,C).mean(dim=0)
                bgd_res_vecs.append(ch_crop_feat)
            
            bgd_attn_scores = torch.stack(bgd_attn_scores)
            bgd_res_vecs = torch.stack(bgd_res_vecs)    #* Query, C (=1024)
            
            bgd_res_attn_scores.append(bgd_attn_scores)
            bgd_res_embeds.append(bgd_res_vecs)
            
            
            
            del ch_img_feat
        #* res_embeds = torch.stack(res_embeds)
        res_attn_scores = torch.cat(res_attn_scores)
        res_embeds = torch.cat(res_embeds, dim=0)
        
        bgd_res_attn_scores = torch.cat(bgd_res_attn_scores)
        bgd_res_embeds = torch.cat(bgd_res_embeds, dim=0)
        
        #todo resnet embedding
        # print('res_attn_scores:', res_attn_scores.shape)
        # print('res_embeds:', res_embeds.shape)
        # print('bgd_res_embeds:', bgd_res_embeds.shape)
        
        # print('res_embeds[idx]:', res_embeds[idx].shape)
        #* cid_mask = torch.nonzero(gt_classes_o == cid).squeeze(1)
        #* iter_dict[cid.item()] = {'class_mask': cid_mask}
        # src_hs = pred_hs[idx]
        #*src_embeds = res_embeds[idx]
        # src_embeds = res_embeds
        for key in iter_dict.keys():
            # iter_dict[key]['hidden_state'] = src_hs[c_mask]
            # print(f'[Category:{key}] Hidden_State:', iter_dict[key]['hidden_state'].shape)
            c_mask = iter_dict[key]['class_mask']
            iter_dict[key]['res_attn'] = res_attn_scores[c_mask].cpu()
            iter_dict[key]['res_embeds'] = res_embeds[c_mask].cpu()
            
            if key ==9:
                # bgd_res_attn_scores = torch.cat(bgd_res_attn_scores)
                # bgd_res_embeds = torch.cat(bgd_res_embeds, dim=0)
                iter_dict[key]['bgd_res_attn'] = bgd_res_attn_scores
                iter_dict[key]['bgd_res_embeds'] = bgd_res_embeds.cpu()
            # print(f'[Category:{key}] Res_Attn:', iter_dict[key]['res_attn'].shape)
            # print(f'[Category:{key}] Res_embeds:', iter_dict[key]['res_embeds'].shape)
        
        # print('res_embeds[obj_mask]:', res_embeds[obj_mask].shape)
        # print('res_embeds[known_mask]:', res_embeds[known_mask].shape)
        # print('res_embeds[unknown_mask]:', res_embeds[unknown_mask].shape)
        
        return iter_dict

        
    def loss_boxes(self, pred_boxes, annotations, indices, num_boxes, use_pseudo_label_weights=False):
        """Compute the losses related to the bounding boxes: the L1 regression loss
           annotations dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The annotations boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        idx = self._get_src_permutation_idx(indices)
        src_boxes = pred_boxes[idx]
        gt_boxes = torch.cat([anno['boxes'][i] for anno, (_, i) in zip(annotations, indices)], dim=0)
        if use_pseudo_label_weights:
            gt_weights = torch.cat([anno['scores'][i] for anno, (_, i) in zip(annotations, indices)], dim=0)
            loss_bbox = l1_loss(src_boxes, gt_boxes, reduction='none') * gt_weights.unsqueeze(-1)
        else:
            loss_bbox = l1_loss(src_boxes, gt_boxes, reduction='none')
        return loss_bbox.sum() / num_boxes

    def loss_giou(self, pred_boxes, annotations, indices, num_boxes, use_pseudo_label_weights=False):
        """Compute the losses related to the bounding boxes: the gIoU loss
           annotations dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The annotations boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        idx = self._get_src_permutation_idx(indices)
        src_boxes = pred_boxes[idx]
        gt_boxes = torch.cat([anno['boxes'][i] for anno, (_, i) in zip(annotations, indices)], dim=0)
        if use_pseudo_label_weights:
            gt_weights = torch.cat([anno['scores'][i] for anno, (_, i) in zip(annotations, indices)], dim=0)
            loss_giou = 1 - torch.diag(generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(gt_boxes)))
            loss_giou = loss_giou * gt_weights
        else:
            loss_giou = 1 - torch.diag(generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(gt_boxes)))
        return loss_giou.sum() / num_boxes

    def record_positive_logits(self, logits, indices):
        idx = self._get_src_permutation_idx(indices)
        labels = logits[idx].argmax(dim=1)
        pos_logits = logits[idx].max(dim=1).values
        for label, logit in zip(labels, pos_logits):
            self.logits_sum[label] += logit
            self.logits_count[label] += 1

    def dynamic_threshold(self, thresholds):
        for s in self.logits_sum:
            all_reduce(s)
        for n in self.logits_count:
            all_reduce(n)
        logits_means = [s.item() / n.item() if n > 0 else 0.0 for s, n in zip(self.logits_sum, self.logits_count)]
        assert len(logits_means) == len(thresholds)
        new_thresholds = [self.gamma_dt * threshold + (1 - self.gamma_dt) * self.alpha_dt * math.sqrt(mean)
                          for threshold, mean in zip(thresholds, logits_means)]
        new_thresholds = [max(min(threshold, self.max_dt), 0.25) for threshold in new_thresholds]
        print('New Dynamic Thresholds: ', new_thresholds)
        return new_thresholds

    def clear_positive_logits(self):
        self.logits_sum = [torch.zeros(1, dtype=torch.float, device=self.device) for _ in range(self.num_classes)]
        self.logits_count = [torch.zeros(1, dtype=torch.int, device=self.device) for _ in range(self.num_classes)]

    @staticmethod
    def _get_src_permutation_idx(indices):
        # Permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    @staticmethod
    def _get_tgt_permutation_idx(indices):
        # Permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    @staticmethod
    def _discard_empty_labels(out, annotations):
        reserve_index = []
        for anno_idx in range(len(annotations)):
            if torch.numel(annotations[anno_idx]["boxes"]) != 0:
                reserve_index.append(anno_idx)
        for key, value in out.items():
            if key in ['logit_all', 'boxes_all']:
                out[key] = value[:, reserve_index, ...]
            elif key in ['features']:
                continue
            else:
                out[key] = value[reserve_index, ...]
        annotations = [annotations[idx] for idx in reserve_index]
        return out, annotations


    def analysis(self, samples, out, annotations=None, ):
        #* unk_loss, unk_loss_dict = criterion_pseudo_unk(teacher_out_u, u_pseudo_labels)
        
        logit_all = out['logit_all']
        boxes_all = out['boxes_all']
        
        # hs_all = out['hidden_states_all']
        hs_last = out['hidden_states_last']
        
        # cls_wts_all = out['classifier_wts_all']
        cls_wts_last = out['classifier_wts_last']
        res_feats = out['resnet_1024_feat']
        
        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_boxes = sum(len(anno["labels"]) for anno in annotations) if annotations is not None else 0
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=logit_all.device)
        if is_dist_avail_and_initialized():
            all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        
        # Compute all the requested losses
        loss = torch.zeros(1).to(logit_all.device)
        loss_dict = defaultdict(float)
        num_decoder_layers = logit_all.shape[0]
        
        
        if annotations is not None:
            indices = self.matcher(logit_all[-1], boxes_all[-1], annotations)
            # gt_classes = torch.cat([anno["labels"][j] for anno, (_, j) in zip(annotations, indices)]).unique().cpu()
            # anno = [anno["labels"][j] for anno, (_, j) in zip(annotations, indices)]
            anno = torch.cat([anno["labels"][j] for anno, (_, j) in zip(annotations, indices)]).unique().cpu()
            mask_truck = torch.nonzero(anno==5).squeeze(1)
            mask_bux = torch.nonzero(anno==8).squeeze(1)
            if (len(mask_truck) + len(mask_bux)) > 0:
                #* class idx: 2 (car), 5 (truck), 8 (bus), 9 (unknown)
                # iter_dict = self.objectness(samples, logit_all[-1], hs_all[-1], cls_wts_all[-1], boxes_all[-1], res_feats, annotations, indices, num_boxes)
                iter_dict = self.objectness(samples, logit_all[-1], hs_last, cls_wts_last, boxes_all[-1], res_feats, annotations, indices, num_boxes)
            else:
                iter_dict = None
            return iter_dict
        else:
            return None
            
        # for i in range(num_decoder_layers):
        #     # Compute DETR losses
        #     if annotations is not None:
        #         indices = self.matcher(logit_all[i], boxes_all[i], annotations)

        #         # hs_all = out['hidden_states_all']
        #         # cls_wts_all = out['classifier_wts_all']
        #         self.objectness(samples, logit_all[i], hs_all[i], cls_wts_all[i], boxes_all[i], res_feats, annotations, indices, num_boxes)
    
    
    
    def forward(self, samples, out, annotations=None, use_pseudo_label_weights=False):
        #* unk_loss, unk_loss_dict = criterion_pseudo_unk(teacher_out_u, u_pseudo_labels)
        
        logit_all = out['logit_all']
        boxes_all = out['boxes_all']
        
        hs_all = out['hidden_states_all']
        cls_wts_all = out['classifier_wts_all']
        res_feats = out['resnet_1024_feat']
        
        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_boxes = sum(len(anno["labels"]) for anno in annotations) if annotations is not None else 0
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=logit_all.device)
        if is_dist_avail_and_initialized():
            all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        
        # Compute all the requested losses
        loss = torch.zeros(1).to(logit_all.device)
        loss_dict = defaultdict(float)
        num_decoder_layers = logit_all.shape[0]
        for i in range(num_decoder_layers):
            # Compute DETR losses
            if annotations is not None:
                indices = self.matcher(logit_all[i], boxes_all[i], annotations)
                # k_indices, _ = self.matcher(logit_all[i], boxes_all[i], k_annotations)
                # u_indices, _ = self.matcher(logit_all[i], boxes_all[i], u_annotations)
                
                #todo Pseudo-Label Unknown 할당은 모델 예측 값 통해 모든 Annotation 지정
                #todo Mathcing Score 방식은 Unmatched 중 Score가 높은 Proposal에 대해 Unknown ID 생성

                # Compute the DETR losses
                #* loss_class = self.loss_class(logit_all[i], annotations, indices, num_boxes, use_pseudo_label_weights)
                loss_class = self.loss_class(logit_all[i], annotations, indices, num_boxes, use_pseudo_label_weights)
                loss_boxes = self.loss_boxes(boxes_all[i], annotations, indices, num_boxes, use_pseudo_label_weights)
                loss_giou = self.loss_giou(boxes_all[i], annotations, indices, num_boxes, use_pseudo_label_weights)
                
                # hs_all = out['hidden_states_all']
                # cls_wts_all = out['classifier_wts_all']
                loss_debias = self.class_refinements(logit_all[i], hs_all[i], cls_wts_all[i], boxes_all[i], res_feats, annotations, indices, num_boxes)
                self.objectness(samples, logit_all[i], hs_all[i], cls_wts_all[i], boxes_all[i], res_feats, annotations, indices, num_boxes)
                
                loss_dict["loss_class"] += loss_class
                loss_dict["loss_boxes"] += loss_boxes
                loss_dict["loss_giou"] += loss_giou
                
                loss_dict["loss_debias"] += loss_debias
                loss += self.coef_class * loss_class + self.coef_boxes * loss_boxes + self.coef_giou * loss_giou + loss_debias

        # Calculate average for all decoder layers
        loss /= num_decoder_layers
        for k, v in loss_dict.items():
            loss_dict[k] /= num_decoder_layers
        return loss, loss_dict
    
    