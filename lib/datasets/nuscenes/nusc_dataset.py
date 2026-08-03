import os
import numpy as np
import torch.utils.data as data
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
ROOT_DIR = os.path.dirname(ROOT_DIR)
ROOT_DIR = os.path.dirname(ROOT_DIR)
sys.path.append(ROOT_DIR)

from lib.datasets.kitti.pd import PhotometricDistort

from lib.datasets.utils import angle2class
from lib.datasets.kitti.kitti_utils import get_objects_from_label
from lib.datasets.kitti.kitti_utils import Calibration
from lib.datasets.kitti.kitti_utils import get_affine_transform
from lib.datasets.kitti.kitti_utils import affine_transform
from lib.datasets.kitti.kitti_eval_python.eval import get_official_eval_result
from lib.datasets.nuscenes.nusc_eval import get_official_eval_result_nusc
import lib.datasets.kitti.kitti_eval_python.kitti_common as kitti


class NUSC_Dataset(data.Dataset):
    """nuScenes (front camera) exported to the KITTI label format.

    Expected layout under `root_dir` (all of it may be symlinked):
        ImageSets/{train,val}.txt
        training/{image,calib,label}     <- train split
        validation/{image,calib,label}   <- val split

    `label/` holds the lower-case nuScenes class names, `label_2/` the
    capitalised ones; pick one with the `label_dir` config key.

    Two evaluation taxonomies are supported through `eval_mode`:
        'kitti'    -> pedestrian / car / cyclist, scored by the KITTI evaluator.
                      This is the cross-dataset setting (train on KITTI, test
                      here, or vice versa). Note that nuScenes has no `cyclist`
                      class, so that row stays empty unless the labels were
                      remapped beforehand.
        'nuscenes' -> the 10 native nuScenes classes, scored by nusc_eval.
    """

    # h, w, l averaged over the nuScenes training split
    NUSC_CLASSES = ['barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
                    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck']
    NUSC_MEAN_SIZE = np.array([[0.9856928278, 2.5350272733, 0.4964593291],
                               [1.3212777838, 0.6161487411, 1.7345529633],
                               [3.5310034111, 2.9322881261, 11.2886576022],
                               [1.7303239450, 1.9504693996, 4.6300396845],
                               [3.2509390814, 2.8786857152, 7.0579567879],
                               [1.5086667691, 0.7789657637, 2.1047384053],
                               [1.7830991759, 0.6741489141, 0.7322679154],
                               [1.0864875408, 0.4343957855, 0.4397158911],
                               [3.8569623772, 2.8859517869, 12.2184129639],
                               [2.9017936138, 2.5306154230, 7.0539758051]])

    KITTI_CLASSES = ['pedestrian', 'car', 'cyclist']
    # pedestrian / car / bicycle rows of NUSC_MEAN_SIZE
    KITTI_MEAN_SIZE = NUSC_MEAN_SIZE[[6, 3, 1]]
    # index of each class in the KITTI evaluator's own class list
    KITTI_EVAL_ID = {'car': 0, 'pedestrian': 1, 'cyclist': 2}

    def __init__(self, split, cfg):

        # basic configuration
        self.root_dir = cfg.get('root_dir')
        self.split = split
        self.max_objs = 50

        self.eval_mode = cfg.get('eval_mode', 'kitti')
        assert self.eval_mode in ['kitti', 'nuscenes']
        if self.eval_mode == 'nuscenes':
            self.class_name = list(self.NUSC_CLASSES)
            self.cls_mean_size = self.NUSC_MEAN_SIZE.copy()
        else:
            self.class_name = list(self.KITTI_CLASSES)
            self.cls_mean_size = self.KITTI_MEAN_SIZE.copy()
        self.num_classes = len(self.class_name)
        self.cls2id = {name: i for i, name in enumerate(self.class_name)}

        # nuScenes front images are 1600 x 900, so a wider crop than KITTI's
        self.resolution = np.array(cfg.get('resolution', [896, 512]))  # W * H
        self.use_3d_center = cfg.get('use_3d_center', True)
        self.writelist = [name.lower() for name in cfg.get('writelist', ['car'])]
        unknown = [name for name in self.writelist if name not in self.cls2id]
        assert not unknown, \
            "writelist entries %s are not in the '%s' taxonomy %s" % (unknown, self.eval_mode, self.class_name)
        self.meanshape = cfg.get('meanshape', False)
        if not self.meanshape:
            self.cls_mean_size = np.zeros_like(self.cls_mean_size, dtype=np.float32)

        # data split loading
        self.split_file = os.path.join(self.root_dir, 'ImageSets', self.split + '.txt')
        self.idx_list = [x.strip() for x in open(self.split_file).readlines()]

        # path configuration
        self.data_dir = os.path.join(self.root_dir, 'validation' if 'val' in split else 'training')
        self.image_dir = os.path.join(self.data_dir, 'image')
        self.calib_dir = os.path.join(self.data_dir, 'calib')
        self.label_dir = os.path.join(self.data_dir, cfg.get('label_dir', 'label'))

        # data augmentation configuration
        self.data_augmentation = True if 'train' in split else False
        self.istrain = self.data_augmentation

        self.aug_pd = cfg.get('aug_pd', False)
        self.aug_crop = cfg.get('aug_crop', False)
        self.aug_calib = cfg.get('aug_calib', False)

        self.random_mixup3d = cfg.get('random_mixup3d', 0.5)
        self.random_flip = cfg.get('random_flip', 0.5)
        self.random_crop = cfg.get('random_crop', 0.5)
        self.scale = cfg.get('scale', 0.4)
        self.shift = cfg.get('shift', 0.1)

        self.depth_scale = cfg.get('depth_scale', 'normal')

        # letterbox the image instead of squeezing it into self.resolution, and
        # optionally rescale it so that every sample shares one focal length
        self.maintain_image_ratio = cfg.get('maintain_image_ratio', False)
        self.target_focal = cfg.get('target_focal', None)

        # statistics
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # others
        self.downsample = 32
        self.depth_downsample_factor = 16
        self.pd = PhotometricDistort()
        self.clip_2d = cfg.get('clip_2d', False)

    def get_image(self, idx):
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        return Image.open(img_file)    # (H, W, 3) RGB mode

    def get_image_size(self, idx):
        # PIL only reads the header until the pixels are touched
        return np.array(self.get_image(idx).size)

    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, '%06d.txt' % idx)
        assert os.path.exists(label_file)
        return get_objects_from_label(label_file)

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '%06d.txt' % idx)
        assert os.path.exists(calib_file)
        return Calibration(calib_file)

    def calculate_new_image_size(self, img_size, resolution):
        """Largest size that fits in `resolution` while keeping the aspect ratio."""
        if resolution[0] / img_size[0] < resolution[1] / img_size[1]:
            ratio = resolution[0] / img_size[0]
        else:
            ratio = resolution[1] / img_size[1]
        return np.array([int(ratio * img_size[0]), int(ratio * img_size[1])]), ratio

    def get_image_with_padding(self, img, target_size, pad_color=(0, 0, 0)):
        new_img = Image.new(img.mode, tuple(int(v) for v in target_size), pad_color)
        new_img.paste(img, (0, 0))
        return new_img

    def build_transform(self, img_size, center, crop_size, calib):
        """Affine that maps the source image onto the network input.

        Returns (trans, trans_inv, out_size, scale_ratio); `out_size` is the size
        the image is warped to before padding, `scale_ratio` is None unless
        maintain_image_ratio is on.
        """
        if not self.maintain_image_ratio:
            trans, trans_inv = get_affine_transform(center, crop_size, 0, self.resolution, inv=1)
            return trans, trans_inv, self.resolution, None

        out_size, scale_ratio = self.calculate_new_image_size(img_size, self.resolution)
        if self.target_focal is not None:
            # zoom so that the effective focal length becomes self.target_focal
            if self.resolution[1] / img_size[1] > self.resolution[0] / img_size[0]:
                focal_crop = calib.P2[0, 0] * self.resolution[0] / self.target_focal / img_size[0]
            else:
                focal_crop = calib.P2[0, 0] * self.resolution[1] / self.target_focal / img_size[1]
            crop_size = (crop_size / img_size - 1 + focal_crop) * img_size
        trans, trans_inv = get_affine_transform(center, crop_size, 0, out_size, inv=1)
        return trans, trans_inv, out_size, scale_ratio

    def get_eval_calib(self, idx):
        """Calibration matching the coordinate frame the detections live in.

        Without maintain_image_ratio the decoder works in original image
        coordinates (as for KITTI), so the raw calibration applies. Otherwise
        the image was warped by `trans`, and P2 has to be warped along with it:
        a point projects to trans @ P2 @ X.
        """
        calib = self.get_calib(idx)
        if not self.maintain_image_ratio:
            return calib
        img_size = self.get_image_size(idx)
        trans, _, _, _ = self.build_transform(img_size, np.array(img_size) / 2, img_size, calib)
        P2 = np.vstack([trans, [0, 0, 1]]).astype(np.float32) @ calib.P2
        return Calibration({'P2': P2, 'R0': calib.R0, 'Tr_velo2cam': calib.V2C})

    def eval(self, results_dir, logger):
        logger.info("==> Loading detections and GTs...")
        img_ids = [int(id) for id in self.idx_list]
        dt_annos = kitti.get_label_annos(results_dir)
        gt_annos = kitti.get_label_annos(self.label_dir, img_ids)

        logger.info('==> Evaluating (official) ...')
        car_moderate = 0
        for category in self.writelist:
            if self.eval_mode == 'nuscenes':
                results_str, results_dict, mAP3d_R40 = get_official_eval_result_nusc(
                    gt_annos, dt_annos, self.cls2id[category])
            else:
                results_str, results_dict, mAP3d_R40 = get_official_eval_result(
                    gt_annos, dt_annos, self.KITTI_EVAL_ID[category])
            if category == 'car':
                car_moderate = mAP3d_R40
            logger.info(results_str)
        return car_moderate

    def __len__(self):
        return self.idx_list.__len__()

    def __getitem__(self, item):
        #  ============================   get inputs   ===========================
        index = int(self.idx_list[item])  # index mapping, get real data id
        # image loading
        img = self.get_image(index)
        img_size = np.array(img.size)
        features_size = self.resolution // self.downsample    # W * H

        dst_W, dst_H = img_size

        # data augmentation for image
        center = np.array(img_size) / 2
        crop_size, crop_scale = img_size, 1
        random_flip_flag, random_crop_flag = False, False
        random_mix_flag = False
        calib = self.get_calib(index)

        if self.data_augmentation:

            if np.random.random() < self.random_mixup3d:
                random_mix_flag = True

            if self.aug_pd:
                img = np.array(img).astype(np.float32)
                img = self.pd(img).astype(np.uint8)
                img = Image.fromarray(img)

            if np.random.random() < self.random_flip:
                random_flip_flag = True
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            if self.aug_crop:
                if np.random.random() < self.random_crop:
                    random_crop_flag = True
                    crop_scale = np.clip(np.random.randn() * self.scale + 1, 1 - self.scale, 1 + self.scale)
                    crop_size = img_size * crop_scale
                    center[0] += img_size[0] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)
                    center[1] += img_size[1] * np.clip(np.random.randn() * self.shift, -2 * self.shift, 2 * self.shift)

        if random_mix_flag == True:
            count_num = 0
            random_mix_flag = False
            while count_num < 50:
                count_num += 1
                random_index = int(np.random.choice(self.idx_list))
                calib_temp = self.get_calib(random_index)

                if calib_temp.cu == calib.cu and calib_temp.cv == calib.cv and calib_temp.fu == calib.fu and calib_temp.fv == calib.fv:
                    img_temp = self.get_image(random_index)
                    img_size_temp = np.array(img_temp.size)
                    dst_W_temp, dst_H_temp = img_size_temp
                    if dst_W_temp == dst_W and dst_H_temp == dst_H:
                        objects_1 = self.get_label(index)
                        objects_2 = self.get_label(random_index)
                        if len(objects_1) + len(objects_2) < self.max_objs:
                            random_mix_flag = True
                            if random_flip_flag == True:
                                img_temp = img_temp.transpose(Image.FLIP_LEFT_RIGHT)
                            img_blend = Image.blend(img, img_temp, alpha=0.5)
                            img = img_blend
                            break

        # add affine transformation for 2d images.
        trans, trans_inv, out_size, scale_ratio = self.build_transform(img_size, center, crop_size, calib)
        img = img.transform(tuple(int(v) for v in out_size),
                            method=Image.AFFINE,
                            data=tuple(trans_inv.reshape(-1).tolist()),
                            resample=Image.BILINEAR)
        if self.maintain_image_ratio:
            img = self.get_image_with_padding(img, self.resolution)

        # `calib` keeps projecting into the *original* image below, but the boxes
        # are normalised by self.resolution, so the calibration handed to the
        # network has to follow the resize for depth_geo to stay consistent.
        p2_out = calib.P2.copy()
        if self.maintain_image_ratio:
            p2_out[0, 0] = self.target_focal if self.target_focal is not None else p2_out[0, 0] * scale_ratio

        # image encoding
        img = np.array(img).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = img.transpose(2, 0, 1)  # C * H * W

        #  ============================   get labels   ==============================
        objects = self.get_label(index)

        # data augmentation for labels
        if random_flip_flag:
            if self.aug_calib:
                calib.flip(img_size)
            for object in objects:
                [x1, _, x2, _] = object.box2d
                object.box2d[0],  object.box2d[2] = img_size[0] - x2, img_size[0] - x1
                object.alpha = np.pi - object.alpha
                object.ry = np.pi - object.ry
                if self.aug_calib:
                    object.pos[0] *= -1
                if object.alpha > np.pi:  object.alpha -= 2 * np.pi  # check range
                if object.alpha < -np.pi: object.alpha += 2 * np.pi
                if object.ry > np.pi:  object.ry -= 2 * np.pi
                if object.ry < -np.pi: object.ry += 2 * np.pi

        # labels encoding
        calibs = np.zeros((self.max_objs, 3, 4), dtype=np.float32)
        indices = np.zeros((self.max_objs), dtype=np.int64)
        mask_2d = np.zeros((self.max_objs), dtype=bool)
        labels = np.zeros((self.max_objs), dtype=np.int8)
        depth = np.zeros((self.max_objs, 1), dtype=np.float32)
        heading_bin = np.zeros((self.max_objs, 1), dtype=np.int64)
        heading_res = np.zeros((self.max_objs, 1), dtype=np.float32)
        size_2d = np.zeros((self.max_objs, 2), dtype=np.float32)
        size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
        src_size_3d = np.zeros((self.max_objs, 3), dtype=np.float32)
        boxes = np.zeros((self.max_objs, 4), dtype=np.float32)
        boxes_3d = np.zeros((self.max_objs, 6), dtype=np.float32)

        obj_region = np.zeros((img.shape[1], img.shape[2]), dtype=bool) # (H, W)

        object_num = len(objects) if len(objects) < self.max_objs else self.max_objs

        for i in range(object_num):
            # filter objects by writelist
            if objects[i].cls_type not in self.writelist:
                continue

            # filter inappropriate samples
            if objects[i].level_str == 'UnKnown' or objects[i].pos[-1] < 2:
                continue

            # ignore the samples beyond the threshold [hard encoding]
            threshold = 65
            if objects[i].pos[-1] > threshold:
                continue

            # process 2d bbox & get 2d center
            bbox_2d = objects[i].box2d.copy()

            # add affine transformation for 2d boxes.
            bbox_2d[:2] = affine_transform(bbox_2d[:2], trans)
            bbox_2d[2:] = affine_transform(bbox_2d[2:], trans)

            # process 3d center
            center_2d = np.array([(bbox_2d[0] + bbox_2d[2]) / 2, (bbox_2d[1] + bbox_2d[3]) / 2], dtype=np.float32)  # W * H

            # create object region
            ymin, ymax = int(max(bbox_2d[1], 0)), int(min(bbox_2d[3], img.shape[1]))
            xmin, xmax = int(max(bbox_2d[0], 0)), int(min(bbox_2d[2], img.shape[2]))
            obj_region[ymin:ymax, xmin:xmax] = 1

            corner_2d = bbox_2d.copy()

            center_3d = objects[i].pos + [0, -objects[i].h / 2, 0]  # real 3D center in 3D space
            center_3d = center_3d.reshape(-1, 3)  # shape adjustment (N, 3)

            center_3d, rect_depth = calib.rect_to_img(center_3d)  # project 3D center to image plane
            center_3d = center_3d[0]  # shape adjustment

            if random_flip_flag and not self.aug_calib:  # random flip for center3d
                center_3d[0] = img_size[0] - center_3d[0]
            center_3d = affine_transform(center_3d.reshape(-1), trans)

            # filter 3d center out of img
            proj_inside_img = True

            if center_3d[0] < 0 or center_3d[0] >= self.resolution[0]:
                proj_inside_img = False
            if center_3d[1] < 0 or center_3d[1] >= self.resolution[1]:
                proj_inside_img = False

            if proj_inside_img == False:
                continue

            # class
            cls_id = self.cls2id[objects[i].cls_type]
            labels[i] = cls_id

            # encoding 2d/3d boxes
            w, h = bbox_2d[2] - bbox_2d[0], bbox_2d[3] - bbox_2d[1]
            size_2d[i] = 1. * w, 1. * h

            center_2d_norm = center_2d / self.resolution
            size_2d_norm = size_2d[i] / self.resolution

            corner_2d_norm = corner_2d
            corner_2d_norm[0: 2] = corner_2d[0: 2] / self.resolution
            corner_2d_norm[2: 4] = corner_2d[2: 4] / self.resolution
            center_3d_norm = center_3d / self.resolution

            l, r = center_3d_norm[0] - corner_2d_norm[0], corner_2d_norm[2] - center_3d_norm[0]
            t, b = center_3d_norm[1] - corner_2d_norm[1], corner_2d_norm[3] - center_3d_norm[1]

            if l < 0 or r < 0 or t < 0 or b < 0:
                if self.clip_2d:
                    l = np.clip(l, 0, 1)
                    r = np.clip(r, 0, 1)
                    t = np.clip(t, 0, 1)
                    b = np.clip(b, 0, 1)
                else:
                    continue

            boxes[i] = center_2d_norm[0], center_2d_norm[1], size_2d_norm[0], size_2d_norm[1]
            boxes_3d[i] = center_3d_norm[0], center_3d_norm[1], l, r, t, b

            # encoding depth
            if self.depth_scale == 'normal':
                depth[i] = objects[i].pos[-1] * crop_scale

            elif self.depth_scale == 'inverse':
                depth[i] = objects[i].pos[-1] / crop_scale

            elif self.depth_scale == 'none':
                depth[i] = objects[i].pos[-1]

            # encoding heading angle
            heading_angle = calib.ry2alpha(objects[i].ry, (objects[i].box2d[0] + objects[i].box2d[2]) / 2)
            if heading_angle > np.pi:  heading_angle -= 2 * np.pi  # check range
            if heading_angle < -np.pi: heading_angle += 2 * np.pi
            heading_bin[i], heading_res[i] = angle2class(heading_angle)

            # encoding size_3d
            src_size_3d[i] = np.array([objects[i].h, objects[i].w, objects[i].l], dtype=np.float32)
            mean_size = self.cls_mean_size[cls_id]
            size_3d[i] = src_size_3d[i] - mean_size

            if objects[i].trucation <= 0.5 and objects[i].occlusion <= 2:
                mask_2d[i] = 1

            calibs[i] = p2_out

        if random_mix_flag == True:
                objects = self.get_label(random_index)
                # data augmentation for labels
                if random_flip_flag:
                    for object in objects:
                        [x1, _, x2, _] = object.box2d
                        object.box2d[0],  object.box2d[2] = img_size[0] - x2, img_size[0] - x1
                        object.ry = np.pi - object.ry
                        if self.aug_calib:
                            object.pos[0] *= -1
                        if object.ry > np.pi:  object.ry -= 2 * np.pi
                        if object.ry < -np.pi: object.ry += 2 * np.pi
                object_num_temp = len(objects) if len(objects) < (self.max_objs - object_num) else (self.max_objs - object_num)
                for i in range(object_num_temp):
                    if objects[i].cls_type not in self.writelist:
                        continue

                    if objects[i].level_str == 'UnKnown' or objects[i].pos[-1] < 2:
                        continue
                    # process 2d bbox & get 2d center
                    bbox_2d = objects[i].box2d.copy()
                    # add affine transformation for 2d boxes.
                    bbox_2d[:2] = affine_transform(bbox_2d[:2], trans)
                    bbox_2d[2:] = affine_transform(bbox_2d[2:], trans)

                    # process 3d center
                    center_2d = np.array([(bbox_2d[0] + bbox_2d[2]) / 2, (bbox_2d[1] + bbox_2d[3]) / 2], dtype=np.float32)  # W * H

                    # create object region
                    ymin, ymax = int(max(bbox_2d[1], 0)), int(min(bbox_2d[3], img.shape[1]))
                    xmin, xmax = int(max(bbox_2d[0], 0)), int(min(bbox_2d[2], img.shape[2]))
                    obj_region[ymin:ymax, xmin:xmax] = 1

                    corner_2d = bbox_2d.copy()

                    center_3d = objects[i].pos + [0, -objects[i].h / 2, 0]  # real 3D center in 3D space
                    center_3d = center_3d.reshape(-1, 3)  # shape adjustment (N, 3)
                    center_3d, _ = calib.rect_to_img(center_3d)  # project 3D center to image plane
                    center_3d = center_3d[0]  # shape adjustment
                    if random_flip_flag and not self.aug_calib:  # random flip for center3d
                        center_3d[0] = img_size[0] - center_3d[0]
                    center_3d = affine_transform(center_3d.reshape(-1), trans)

                    # filter 3d center out of img
                    proj_inside_img = True

                    if center_3d[0] < 0 or center_3d[0] >= self.resolution[0]:
                        proj_inside_img = False
                    if center_3d[1] < 0 or center_3d[1] >= self.resolution[1]:
                        proj_inside_img = False

                    if proj_inside_img == False:
                            continue

                    # class
                    cls_id = self.cls2id[objects[i].cls_type]
                    labels[i + object_num] = cls_id

                    # encoding 2d/3d boxes
                    w, h = bbox_2d[2] - bbox_2d[0], bbox_2d[3] - bbox_2d[1]
                    size_2d[i + object_num] = 1. * w, 1. * h

                    center_2d_norm = center_2d / self.resolution
                    size_2d_norm = size_2d[i + object_num] / self.resolution

                    corner_2d_norm = corner_2d
                    corner_2d_norm[0: 2] = corner_2d[0: 2] / self.resolution
                    corner_2d_norm[2: 4] = corner_2d[2: 4] / self.resolution
                    center_3d_norm = center_3d / self.resolution

                    l, r = center_3d_norm[0] - corner_2d_norm[0], corner_2d_norm[2] - center_3d_norm[0]
                    t, b = center_3d_norm[1] - corner_2d_norm[1], corner_2d_norm[3] - center_3d_norm[1]

                    if l < 0 or r < 0 or t < 0 or b < 0:
                        if self.clip_2d:
                            l = np.clip(l, 0, 1)
                            r = np.clip(r, 0, 1)
                            t = np.clip(t, 0, 1)
                            b = np.clip(b, 0, 1)
                        else:
                            continue

                    boxes[i + object_num] = center_2d_norm[0], center_2d_norm[1], size_2d_norm[0], size_2d_norm[1]
                    boxes_3d[i + object_num] = center_3d_norm[0], center_3d_norm[1], l, r, t, b

                    # encoding depth
                    if self.depth_scale == 'normal':
                        depth[i + object_num] = objects[i].pos[-1] * crop_scale

                    elif self.depth_scale == 'inverse':
                        depth[i + object_num] = objects[i].pos[-1] / crop_scale

                    elif self.depth_scale == 'none':
                        depth[i + object_num] = objects[i].pos[-1]

                    # encoding heading angle
                    heading_angle = calib.ry2alpha(objects[i].ry, (objects[i].box2d[0]+objects[i].box2d[2])/2)
                    if heading_angle > np.pi:  heading_angle -= 2 * np.pi  # check range
                    if heading_angle < -np.pi: heading_angle += 2 * np.pi
                    heading_bin[i + object_num], heading_res[i + object_num] = angle2class(heading_angle)

                    src_size_3d[i + object_num] = np.array([objects[i].h, objects[i].w, objects[i].l], dtype=np.float32)
                    mean_size = self.cls_mean_size[cls_id]
                    size_3d[i + object_num] = src_size_3d[i + object_num] - mean_size

                    if objects[i].trucation <=0.5 and objects[i].occlusion<=2:
                        mask_2d[i + object_num] = 1

                    calibs[i + object_num] = p2_out

        # collect return data
        inputs = img

        if self.maintain_image_ratio:
            # the network input now *is* the padded canvas, not the source image
            img_size = self.resolution

        targets = {
                   'calibs': calibs,
                   'indices': indices,
                   'img_size': img_size,
                   'labels': labels,
                   'boxes': boxes,
                   'boxes_3d': boxes_3d,
                   'depth': depth,
                   'size_2d': size_2d,
                   'size_3d': size_3d,
                   'src_size_3d': src_size_3d,
                   'heading_bin': heading_bin,
                   'heading_res': heading_res,
                   'mask_2d': mask_2d,
                   'obj_region': obj_region}

        info = {'img_id': index,
                'img_size': img_size,
                'bbox_downsample_ratio': img_size / features_size}
        return inputs, p2_out, targets, info
