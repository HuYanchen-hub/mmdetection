# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, Union


import numpy as np
from mmdet.datasets.transforms.loading import LoadAnnotations

from mmdet.registry import TRANSFORMS
from mmdet.structures.mask import BitmapMasks




@TRANSFORMS.register_module()
class PreprocessAnnotationsOneFormer(LoadAnnotations):
    """Load multiple types of panoptic annotations.

    The annotation format is as the following:

    .. code-block:: python

        {
            # In (x1, y1, x2, y2) order, float type. N is the number of bboxes
            # in an image
            'gt_bboxes': BaseBoxes(N, 4)
             # In int type.
            'gt_bboxes_labels': np.ndarray(N, )
             # In built-in class
            'gt_masks': PolygonMasks (H, W) or BitmapMasks (H, W)
             # In uint8 type.
            'gt_seg_map': np.ndarray (H, W)
             # in (x, y, v) order, float type.
        }

    After this module, the annotation has been changed to the format below:

    .. code-block:: python

        {
            'task': str
             # In str type.
            'task_text': List[str]
             # in List[str] type
        }

    Required Keys:
    - gt_bboxes (BaseBoxes[torch.float32])
    - gt_bboxes_labels (np.int64)
    - gt_masks (BitmapMasks | PolygonMasks)
    - gt_seg_map (np.uint8)
    - gt_ignore_flags (bool)
    - text (List[str])   (thing classes)
    - stuff_text (List[str])  (stuff classes)
    

    Added Keys:

    - masks
    - labels
    - task (str)
    - text_task(List[str])

    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
            Defaults to True.
        with_label (bool): Whether to parse and load the label annotation.
            Defaults to True.
        with_mask (bool): Whether to parse and load the mask annotation.
             Defaults to True.
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Defaults to False.
        box_type (str): The box mode used to wrap the bboxes.
        imdecode_backend (str): The image decoding backend type. The backend
            argument for :func:``mmcv.imfrombytes``.
            See :fun:``mmcv.imfrombytes`` for details.
            Defaults to 'cv2'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend in mmdet >= 3.0.0rc7. Defaults to None.
    """

    def __init__(self,
                 semantic_prob = 0.33,
                 instance_prob = 0.67,
                 num_queries = 134,
                 num_things = 80,
                 num_stuff = 53,
                 backend_args: dict = None) -> None:
        self.semantic_prob = semantic_prob
        self.instance_prob = instance_prob
        self.num_queries = num_queries
        self.num_things = num_things
        self.num_stuff = num_stuff
        self.num_classes = num_things + num_stuff

        super(PreprocessAnnotationsOneFormer, self).__init__(
            backend_args=backend_args)

    def transform(self, results: dict) -> dict:
        """Function to load multiple types panoptic annotations.

        Args:
            results (dict): Result dict from :obj:``mmdet.CustomDataset``.

        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """

        # prob_task = np.random.uniform(0,1.)
        prob_task = 0.5
        
        if prob_task < self.semantic_prob:
            results['task'] = "The task is semantic"
            results = self._get_semantic_dict(results)
        elif prob_task < self.instance_prob:
            results['task'] = "The task is instance"
            results = self._get_instance_dict(results)
        else:
            results['task'] = "The task is panoptic"
            results = self._get_panoptic_dict(results)

        return results

    def _get_semantic_dict(self, results: dict) -> dict:
        ignore_flags = []
        
        num_class_obj = {}
        class_names = results['text'] + results['stuff_text']
        for name in class_names:
            num_class_obj[name] = 0
        
        texts = ["a semantic photo"] * self.num_queries
        h, w = results['gt_seg_map'].shape[-2], results['gt_seg_map'].shape[-1]
        masks = []
        labels = []

        
        gt_semantic_seg = results['gt_seg_map']
        semantic_labels = np.unique(gt_semantic_seg)
        for label in semantic_labels:
            if label < self.num_things or label >= self.num_classes:
                continue
            stuff_mask = gt_semantic_seg == label
            masks.append(stuff_mask.astype(np.uint8))
            labels.append(label)
            cls_name = class_names[label]
            num_class_obj[cls_name] += 1
            ignore_flags.append(0)
        
        num = 0
        for i, cls_name in enumerate(class_names):
            if num_class_obj[cls_name] > 0:
                for _ in range(num_class_obj[cls_name]):
                    if num >= len(texts):
                        break
                    texts[num] = f"a photo with a {cls_name}"
                    num += 1

        if len(masks) == 0:
            # Some image does not have annotation (all ignored)
            masks = np.zeros((0, h, w))
            masks = BitmapMasks(masks, h, w)
        else:
            labels = np.array(labels)
            masks = BitmapMasks(masks, h, w)
        # results['oneformer_labels'] = labels
        # results['oneformer_masks'] = masks
        results['gt_bboxes_labels'] = labels
        results['gt_masks'] = masks
        results['oneformer_texts'] = texts
        if 'gt_ignore_flags' in  results:
            results['gt_ignore_flags'] = np.array(ignore_flags, dtype=bool)

        return results
    
    
    def _get_instance_dict(self, results):
        num_class_obj = {}
        class_names = results['text'] + results['stuff_text']
        for name in class_names:
            num_class_obj[name] = 0

        texts = ["an instance photo"] * self.num_queries
        labels = results['gt_bboxes_labels']
        masks = results['gt_masks']

        for i in range(labels.shape[0]):
            cls_name = class_names[labels[i]]
            num_class_obj[cls_name] += 1

        num = 0
        for i, cls_name in enumerate(class_names):
            if num_class_obj[cls_name] > 0:
                for _ in range(num_class_obj[cls_name]):
                    if num >= len(texts):
                        break
                    texts[num] = f"a photo with a {cls_name}"
                    num += 1
        results['gt_bboxes_labels'] = labels
        results['gt_masks'] = masks
        results['oneformer_texts'] = texts

        return results

    
    def _get_panoptic_dict(self, results):
        classes = []
        texts = ["a panoptic photo"] * self.num_queries
        num_class_obj = {}
        class_names = results['text'] + results['stuff_text']
        for name in class_names:
            num_class_obj[name] = 0

        gt_semantic_seg = results['gt_seg_map']
        semantic_labels = np.unique(gt_semantic_seg)
        stuff_masks_list = []
        stuff_labels_list = []
        ignore_flags = []
        for label in semantic_labels:
            if label < self.num_things or label >= self.num_classes:
                continue
            stuff_mask = gt_semantic_seg == label
            stuff_masks_list.append(stuff_mask)
            stuff_labels_list.append(label)
            ignore_flags.append(0)

        if len(stuff_masks_list) > 0:
            stuff_masks = np.stack(stuff_masks_list, axis=0)
            stuff_labels = np.stack(stuff_labels_list, axis=0)
            labels = np.concatenate([results['gt_bboxes_labels'], stuff_labels], axis=0)
            masks = np.concatenate([results['gt_masks'].to_ndarray(), stuff_masks], axis=0)
            masks = BitmapMasks(masks, masks.shape[-2], masks.shape[-1])
            if 'gt_ignore_flags' in  results:
                ignore_flags = np.concatenate([results['gt_ignore_flags'], np.array(ignore_flags, dtype=bool)], axis=0) 
                results['gt_ignore_flags'] = ignore_flags
        else:
            labels = results['gt_bboxes_labels']
            masks = results['gt_masks']

        for i in range(labels.shape[0]):
            cls_name = class_names[labels[i]]
            num_class_obj[cls_name] += 1

        num = 0
        for i, cls_name in enumerate(class_names):
            if num_class_obj[cls_name] > 0:
                for _ in range(num_class_obj[cls_name]):
                    if num >= len(texts):
                        break
                    texts[num] = f"a photo with a {cls_name}"
                    num += 1
        results['gt_bboxes_labels'] = labels
        results['gt_masks'] = masks
        results['oneformer_texts'] = texts

        return results