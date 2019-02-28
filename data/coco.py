"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""

import os
import pickle
import os.path
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
import json
import uuid
from utils.pycocotools.coco import COCO
from utils.pycocotools.cocoeval import COCOeval
from utils.pycocotools import mask as COCOmask


class COCODetection(data.Dataset):

    def __init__(self, root, image_sets, preproc=None, target_transform=None,
                 dataset_name='COCO'):
        self.root = root
        self.data_path = os.path.join(os.path.expanduser("~"),'data')
        self.cache_path = os.path.join(self.data_path, 'coco_cache')
        self.image_set = image_sets
        self.preproc = preproc
        self.target_transform = target_transform
        self.name = dataset_name
        self.ids = list()
        self.annotations = list()
        self._view_map = {
            'minival2014' : 'val2014',          # 5k val2014 subset
            'valminusminival2014' : 'val2014',  # val2014 \setminus minival2014
            'test-dev2015' : 'test2015',
        }

        for (year, image_set) in image_sets:
            coco_name = image_set+year
            data_name = (self._view_map[coco_name]
                        if coco_name in self._view_map
                        else coco_name)
            annofile = self._get_ann_file(coco_name)
            _COCO = COCO(annofile)
            self._COCO = _COCO
            self.coco_name = coco_name
            cats = _COCO.loadCats(_COCO.getCatIds())
            self._classes = tuple(['__background__'] + [c['name'] for c in cats])
            self.num_classes = len(self._classes)
            self._class_to_ind = dict(zip(self._classes, range(self.num_classes)))
            self._class_to_coco_cat_id = dict(zip([c['name'] for c in cats],
                                                  _COCO.getCatIds()))
            indexes = _COCO.getImgIds()
            self.image_indexes = indexes
            self.ids.extend([self.image_path_from_index(data_name, index) for index in indexes ])
            if image_set.find('test') != -1:
                print('test set will not load annotations!')
            else:
                self.annotations.extend(self._load_coco_annotations(coco_name, indexes,_COCO))



    def image_path_from_index(self, name, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        # Example image path for index=119993:
        #   images/train2014/COCO_train2014_000000119993.jpg
        file_name = ('COCO_' + name + '_' +
                     str(index).zfill(12) + '.jpg')
        image_path = os.path.join(self.root, 'images',
                              name, file_name)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path


    def _get_ann_file(self, name):
        prefix = 'instances' if name.find('test') == -1 \
                else 'image_info'
        return os.path.join(self.root, 'annotations',
                        prefix + '_' + name + '.json')


    def _load_coco_annotations(self, coco_name, indexes, _COCO):
        cache_file=os.path.join(self.cache_path,coco_name+'_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(coco_name,cache_file))
            return roidb

        gt_roidb = [self._annotation_from_index(index, _COCO)
                    for index in indexes]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb,fid,pickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))
        return gt_roidb


    def _annotation_from_index(self, index, _COCO):
        """
        Loads COCO bounding-box instance annotations. Crowd instances are
        handled by marking their overlaps (with all categories) to -1. This
        overlap value means that crowd "instances" are excluded from training.
        """
        im_ann = _COCO.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']

        annIds = _COCO.getAnnIds(imgIds=index, iscrowd=None)
        objs = _COCO.loadAnns(annIds)
        # Sanitize bboxes -- some are invalid
        valid_objs = []
        for obj in objs:
            x1 = np.max((0, obj['bbox'][0]))
            y1 = np.max((0, obj['bbox'][1]))
            x2 = np.min((width - 1, x1 + np.max((0, obj['bbox'][2] - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, obj['bbox'][3] - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)
        objs = valid_objs
        num_objs = len(objs)

        res = np.zeros((num_objs, 5))

        # Lookup table to map from COCO category ids to our internal class
        # indices
        coco_cat_id_to_class_ind = dict([(self._class_to_coco_cat_id[cls],
                                          self._class_to_ind[cls])
                                         for cls in self._classes[1:]])

        for ix, obj in enumerate(objs):
            cls = coco_cat_id_to_class_ind[obj['category_id']]
            res[ix, 0:4] = obj['clean_bbox']
            res[ix, 4] = cls

        return res



    def __getitem__(self, index):
        img_id = self.ids[index]
        target = self.annotations[index]
        img = cv2.imread(img_id, cv2.IMREAD_COLOR)
        height, width, _ = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target)


        if self.preproc is not None:
            img, target = self.preproc(img, target)

                    # target = self.target_transform(target, width, height)
        #print(target.shape)

        return img, target

    def __len__(self):
        return len(self.ids)

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(img_id, cv2.IMREAD_COLOR)


    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        to_tensor = transforms.ToTensor()
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)

    def _print_detection_eval_metrics(self, coco_eval):
        IoU_lo_thresh = 0.5
        IoU_hi_thresh = 0.95
        def _get_thr_ind(coco_eval, thr):
            ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                           (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
            iou_thr = coco_eval.params.iouThrs[ind]
            assert np.isclose(iou_thr, thr)
            return ind

        ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
        ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)
        # precision has dims (iou, recall, cls, area range, max dets)
        # area range index 0: all area ranges
        # max dets index 2: 100 per image
        precision = \
            coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
        ap_default = np.mean(precision[precision > -1])
        print('~~~~ Mean and per-category AP @ IoU=[{:.2f},{:.2f}] '
               '~~~~'.format(IoU_lo_thresh, IoU_hi_thresh))
        print('{:.1f}'.format(100 * ap_default))
        aps = list()
        for cls_ind, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            # minus 1 because of __background__
            precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, cls_ind - 1, 0, 2]
            ap = np.mean(precision[precision > -1])
            # print('{:.1f}'.format(100 * ap))
            aps.append(100 * ap)
        # print(aps)
        print('~~~~ Summary metrics ~~~~')
        coco_eval.summarize()

    def _do_detection_eval(self, res_file, output_dir):
        ann_type = 'bbox'
        coco_dt = self._COCO.loadRes(res_file)
        coco_eval = COCOeval(self._COCO, coco_dt)
        coco_eval.params.useSegm = (ann_type == 'segm')
        coco_eval.evaluate()
        coco_eval.accumulate()
        self._print_detection_eval_metrics(coco_eval)
        eval_file = os.path.join(output_dir, 'detection_results.pkl')
        with open(eval_file, 'wb') as fid:
            pickle.dump(coco_eval, fid, pickle.HIGHEST_PROTOCOL)
        print('Wrote COCO eval results to: {}'.format(eval_file))

    def _coco_results_one_category(self, boxes, cat_id):
        results = []
        for im_ind, index in enumerate(self.image_indexes):
            dets = boxes[im_ind].astype(np.float)
            if dets == []:
                continue
            scores = dets[:, -1]
            xs = dets[:, 0]
            ys = dets[:, 1]
            ws = dets[:, 2] - xs + 1
            hs = dets[:, 3] - ys + 1
            results.extend(
              [{'image_id' : index,
                'category_id' : cat_id,
                'bbox' : [xs[k], ys[k], ws[k], hs[k]],
                'score' : scores[k]} for k in range(dets.shape[0])])
        return results

    def _write_coco_results_file(self, all_boxes, res_file):
        results = []
        print('Collecting Results......')
        for cls_ind, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            # print('Collecting {} results ({:d}/{:d})'.format(cls, cls_ind,
            #                                              self.num_classes ))
            coco_cat_id = self._class_to_coco_cat_id[cls]
            results.extend(self._coco_results_one_category(all_boxes[cls_ind],
                                                           coco_cat_id))

        print('Writing results json to {}'.format(res_file))
        with open(res_file, 'w') as fid:
            json.dump(results, fid)

    def evaluate_detections(self, all_boxes, output_dir):
        res_file = os.path.join(output_dir, ('detections_' +
                                         self.coco_name +
                                         '_results'))
        res_file += '.json'
        self._write_coco_results_file(all_boxes, res_file)

        if self.coco_name.find('test') == -1:
            self._do_detection_eval(res_file, output_dir)
