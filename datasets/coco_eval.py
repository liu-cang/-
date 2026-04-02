import os
import contextlib
import copy
import numpy as np
import time
import datetime

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

from utils.box_utils import convert_to_xywh
from utils.distributed_utils import all_gather
from collections import defaultdict

class CocoEval(COCOeval):

    def __init__(self, coco_gt=None, coco_dt=None, iou_type='bbox'):
        super(CocoEval, self).__init__(coco_gt, coco_dt, iou_type)
        
        if coco_gt is not None:
            self.base_classes = coco_gt.base_classes
            self.novel_classes = coco_gt.novel_classes
            self.cats_dict = coco_gt.cats
            self.unk_id = len(self.base_classes) + len(self.novel_classes) + 1

            # print('[CocoEval] base_classes:', self.base_classes)
            # print('[CocoEval] novel_classes:', self.novel_classes)
            # print('[CocoEval] cats_dict:', self.cats_dict)
            # print('[CocoEval] unk_id:', self.unk_id)
    
    def _prepare(self):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''
        def _toMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann['segmentation'] = rle
        p = self.params
        if p.useCats:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds, catIds=p.catIds))
        else:
            gts=self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=p.imgIds))
            dts=self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=p.imgIds))

        # convert ground truth to mask if iouType == 'segm'
        if p.iouType == 'segm':
            _toMask(gts, self.cocoGt)
            _toMask(dts, self.cocoDt)
        # set ignore flag
        for gt in gts:
            gt['ignore'] = gt['ignore'] if 'ignore' in gt else 0
            gt['ignore'] = 'iscrowd' in gt and gt['iscrowd']
            if p.iouType == 'keypoints':
                gt['ignore'] = (gt['num_keypoints'] == 0) or gt['ignore']
        self._gts = defaultdict(list)       # gt for evaluation
        self._dts = defaultdict(list)       # dt for evaluation
        for gt in gts:
            # print('[Coco_Eval-prepare]')
            # print('gt:', gt)
            cat_name = self.cats_dict[gt['category_id']]['name']
            if cat_name in self.novel_classes:
                gt['category_id'] = self.unk_id
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)
        self.evalImgs = defaultdict(list)   # per-image per-category evaluation results
        self.eval     = {}                  # accumulated evaluation results
    
    def evaluate(self):
        p = self.params
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params = p
        
        self._prepare()
        
        _gts = self._gts
        cat_gt = []
        # print('_gts.keys():',_gts.keys())
        for _gt in _gts.keys():
            for gt_dict in _gts[_gt[0],_gt[1]]:
                cat_gt.append(gt_dict['category_id'])
        # print('[coco_eval] _gts_category:', np.unique(cat_gt, return_counts=True))
        
        _dts = self._dts
        cat_dt = []
        for _dt in _dts.keys():
            for gt_dict in _dts[_dt[0],_dt[1]]:
                cat_dt.append(gt_dict['category_id'])
        # print('[coco_eval] _dts_category:', np.unique(cat_dt, return_counts=True))
        
        cat_ids = p.catIds if p.useCats else [-1]
        self.ious = {
            (imgId, catId): self.computeIoU(imgId, catId)
            for imgId in p.imgIds
            for catId in cat_ids
        }
        eval_imgs = [
            self.evaluateImg(imgId, catId, areaRng, p.maxDets[-1])
            for catId in cat_ids
            for areaRng in p.areaRng
            for imgId in p.imgIds
        ]
        eval_imgs = np.asarray(eval_imgs).reshape(len(cat_ids), len(p.areaRng), len(p.imgIds))
        self._paramsEval = copy.deepcopy(self.params)
        
        # print('[coco_eval] self.params.catIds:', self.params.catIds)
        
        return p.imgIds, eval_imgs

    def accumulate(self, p = None):
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T           = len(p.iouThrs)
        R           = len(p.recThrs)
        K           = len(p.catIds) if p.useCats else 1
        A           = len(p.areaRng)
        M           = len(p.maxDets)
        precision   = -np.ones((T,R,K,A,M)) # -1 for the precision of absent categories
        recall      = -np.ones((T,K,A,M))
        scores      = -np.ones((T,R,K,A,M))

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds)  if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng)) if a in setA]
        i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0*A0*I0
            for a, a0 in enumerate(a_list):
                Na = a0*I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if not e is None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate([e['dtScores'][0:maxDet] for e in E])

                    # different sorting method generates slightly different results.
                    # mergesort is used to be consistent as Matlab implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]

                    dtm  = np.concatenate([e['dtMatches'][:,0:maxDet] for e in E], axis=1)[:,inds]
                    dtIg = np.concatenate([e['dtIgnore'][:,0:maxDet]  for e in E], axis=1)[:,inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg==0 )
                    if npig == 0:
                        continue
                    tps = np.logical_and(               dtm,  np.logical_not(dtIg) )
                    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg) )

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=float)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp+tp+np.spacing(1))
                        q  = np.zeros((R,))
                        ss = np.zeros((R,))

                        if nd:
                            recall[t,k,a,m] = rc[-1]
                        else:
                            recall[t,k,a,m] = 0

                        # numpy is slow without cython optimization for accessing elements
                        # use python array gets significant speed improvement
                        pr = pr.tolist(); q = q.tolist()

                        for i in range(nd-1, 0, -1):
                            if pr[i] > pr[i-1]:
                                pr[i-1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs, side='left')
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except:
                            pass
                        precision[t,:,k,a,m] = np.array(q)
                        scores[t,:,k,a,m] = np.array(ss)
        self.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall':   recall,
            'scores': scores,
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format( toc-tic))
    
    def summarize_ap(self, if_print=True):

        def _summarize(iou_thr=None, area_rng='all', max_dets=100):
            p = self.params
            iou_str = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iou_thr is None else '{:0.2f}'.format(iou_thr)
            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == area_rng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == max_dets]
            # dimension of precision: [TxRxKxAxM]
            s = self.eval['precision']
            if iou_thr is not None:
                t = np.where(iou_thr == p.iouThrs)[0]
                s = s[t]
            s = s[:, :, :, aind, mind]
            # print('[EVAL] aps-1:', s.shape)
            aps = np.asarray([np.mean(s[:, :, i, :]) for i in range(s.shape[2])])
            # print('[EVAL] aps-2:', aps.shape)
            
            recs = self.eval['recall']
            if iou_thr is not None:
                t = np.where(iou_thr == p.iouThrs)[0]
                recs = recs[t]
            recs = recs[:,:,aind,mind].squeeze()
            
            # print('[EVAL] recs:', recs.shape, recs.squeeze())
            # recs = np.asarray([np.mean(recs[:, :, i, :]) for i in range(recs.shape[1])])
            # print('[EVAL] recs:', recs.shape)
            
            # print('[EVAL] self.params.cat_ids:', p.catIds)
            
            #* Original
            aps_clean = [ap for ap in aps[:-1] if ap > -0.001]  #* Exclude Last Class (unknown) Precision
            # print('[Summarize] aps_clean', len(aps_clean), aps_clean)
            #* base_map_idx = np.where(aps_clean > -1.)[0]
            #* print('[Summarize] aps_clean[mask]', len(aps_clean), aps_clean)
            
            mean_ap = np.mean(aps_clean)
            # map_idx = np.array(list(self.cocoGt.cats.keys()))
            # aps_clean = [ap for i, ap in enumerate(aps) if ap > -0.001 and (i+1) in map_idx]
            # mean_ap = np.mean(aps_clean)
            if if_print:
                print('Mean Average Precision of Base Classes @ [ IoU='
                      + iou_str + ' | area=' + area_rng + ' | max_dets=' + str(max_dets) + ' ] = ' + str(mean_ap))
                
                # print('\tAP of category [' + self.cocoGt.cats[0]['name'] + ']:\t\t' + str(aps[0]))
                for i, ap in enumerate(aps):
                    #* Original
                    # print('\tAP of category [' + self.cocoGt.cats[i + 1]['name'] + ']:\t\t' + str(ap))
                    if ap > -1.:
                        print('\tAP of category [' + self.cocoGt.cats[i + 1]['name'] + ']:\t\t' + str(ap))
                        print('\tRC of category [' + self.cocoGt.cats[i + 1]['name'] + ']:\t\t' + str(recs[i]))
            # return aps
            return aps_clean

        if not self.eval:
            raise Exception('Please run accumulate() first')
        return _summarize(iou_thr=0.5, max_dets=self.params.maxDets[2])


class CocoEvaluator:

    def __init__(self, coco_gt):
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt
        self.coco_eval = CocoEval(coco_gt)
        self.img_ids = []
        self.eval_imgs = []
        
        # self.coco_eval.params.catIds = sorted(self.coco_eval.params.catIds.append(0))
        catIds = list(np.unique(self.coco_eval.params.catIds)) + [len(self.coco_eval.params.catIds)+1]
        self.coco_eval.params.catIds = sorted(list(catIds))
        # print("[Coco Evaluator] params.catIds", self.coco_eval.params.catIds)

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)
        results = self.prepare_for_coco_detection(predictions)
        # suppress pycocotools prints
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()
        self.coco_eval.cocoDt = coco_dt
        # print('self.coco_eval.cocoDt')
        # print(self.coco_eval.cocoDt)
        self.coco_eval.params.imgIds = list(img_ids)
        img_ids, eval_imgs = self.coco_eval.evaluate()
        self.eval_imgs.append(eval_imgs)

    def synchronize_between_processes(self):
        self.eval_imgs = np.concatenate(self.eval_imgs, 2)
        img_ids, eval_imgs = self.merge(self.img_ids, self.eval_imgs)
        img_ids, eval_imgs = list(img_ids), list(eval_imgs.flatten())
        self.coco_eval.evalImgs = eval_imgs
        self.coco_eval.params.imgIds = img_ids
        self.coco_eval._paramsEval = copy.deepcopy(self.coco_eval.params)

    def accumulate(self):
        self.coco_eval.accumulate()

    def summarize(self, if_print=True):
        return self.coco_eval.summarize_ap(if_print)

    @staticmethod
    def prepare_for_coco_detection(predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            #* Original
            if len(prediction) == 0:
                continue
            
            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            coco_results.extend([
                {"image_id": original_id, "category_id": labels[k], "bbox": box, "score": scores[k]}
                for k, box in enumerate(boxes)
            ])
        return coco_results

    @staticmethod
    def merge(img_ids, eval_imgs):
        all_img_ids = all_gather(img_ids)
        all_eval_imgs = all_gather(eval_imgs)
        merged_img_ids = []
        for p in all_img_ids:
            merged_img_ids.extend(p)
        merged_eval_imgs = [p for p in all_eval_imgs]
        merged_img_ids = np.array(merged_img_ids)
        merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)
        # keep only unique (and in sorted order) images
        merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
        merged_eval_imgs = merged_eval_imgs[..., idx]
        return merged_img_ids, merged_eval_imgs
