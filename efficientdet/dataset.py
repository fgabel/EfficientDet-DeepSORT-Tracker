import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader

import cv2

from pathlib import Path
import sys
import tensorflow as tf
import tqdm
import uuid

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2
from waymo_open_dataset.utils import frame_utils

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import array_to_img
import torch
import torchvision.transforms.functional as F

class TUMuchTrafficDataset(Dataset):
    """
        Dataset working on image data only for the Waymo Open Data challenge.
        Uses tfrecords directly, and reading data frame by frame. This way, each call to an index
        reads the next frame of the underlying tfrecord. Once the end of a tfrecord is reached, the
        next call will return the first element again.
        In pseudocode:
        frame_0 = dataset[x]
        frame_1 = dataset[x]
        ...
        frame_n = dataset[x]
        frame_0 = dataset[x]
        ...
    """

    def __init__(self, tfrecord_paths, transform=None):
        self.tfrecord_paths = tfrecord_paths
        self.transform = transform
        self.datasets = [iter(tf.data.TFRecordDataset(x))
                         for x in self.tfrecord_paths]
        self.cache = {k: [] for k in range(len(self.tfrecord_paths))}
    def _class(self, x):
        if x == 1:
            return 0
        elif x == 2:
            return 1
        elif x == 4:
            return 2

    def _load_frame(self, index):
        frame = dataset_pb2.Frame()
        try:
            arr = self.datasets[12].next().numpy()
        except:
            self.datasets[12] = iter(
                tf.data.TFRecordDataset(self.tfrecord_paths[12]))
            arr = self.datasets[12].next().numpy()
        frame.ParseFromString(bytearray(arr))
        return frame

    def _load_annot(self, frame, camera_id):
        annotations = np.zeros((0, 5))
        for box_cord in frame.camera_labels:
            if box_cord.name != camera_id:
                continue
            for label in box_cord.labels:
                ll, ww = label.box.length, label.box.width
                rect = [(label.box.center_x - ll // 2), (label.box.center_y - ww // 2),
                        (label.box.center_x + ll // 2), (label.box.center_y + ww // 2)]
                tmp_ann = np.zeros([1, 5])
                tmp_ann[0, :4] = rect
                tmp_ann[0, 4] = self._class(label.type)
                annotations = np.concatenate((annotations, tmp_ann))
        return annotations

    def _load_image(self, frame, camera_id):
        for image in frame.images:
            if image.name == camera_id:
                return array_to_img(tf.io.decode_jpeg(image.image))

    def __getitem__(self, index):
        if len(self.cache[index]) == 0:
            frame = self._load_frame(index)

            x = int(torch.randint(0, 5, size=(1,))) # camera index
            x=0
            frame_img = self._load_image(frame, camera_id=x + 1)
            
            frame_annot = self._load_annot(frame, camera_id=x + 1)
            self.cache[index].append([frame_img, frame_annot])

        image, annotations = self.cache[index][0]
        self.cache[index] = self.cache[index][1:]

        if self.transform:
            transformed = self.transform({
                "img": image,
                "annot": annotations
            })
            image, annotations = transformed["img"], transformed["annot"]
        return {
            "img": image,
            "annot": annotations
        }

    def __len__(self):

        return len(self.datasets)

    @staticmethod
    def collater(data):
        imgs = [s['img'] for s in data]
        annots = [s['annot'][0] for s in data]

        imgs = torch.stack(imgs, axis=0)
        max_num_annots = max(annot.shape[0] for annot in annots)

        if max_num_annots == 0:
            return {
                'img': imgs,
                'annot': torch.FloatTensor(torch.ones((len(annots), 1, 5)) * -1)
            }
        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot

        return {'img': imgs, 'annot': torch.FloatTensor(annot_padded)}

class TUMuchTrackingDataset(Dataset):
    """

    """

    def __init__(self, tfrecord_path, transform=None, cam_id = 1):
        self.tfrecord_path = tfrecord_path
        self.transform = transform
        self.dataset = iter(tf.data.TFRecordDataset(tfrecord_path))
        self.cam_id = cam_id
    def _class(self, x):
        if x == 1:
            return 0
        elif x == 2:
            return 1
        elif x == 4:
            return 2

    def _load_frame(self, index):
        frame = dataset_pb2.Frame()
        try:
            arr = self.dataset.next().numpy()
        except:
            self.datasets = iter(
                tf.data.TFRecordDataset(self.tfrecord_path))
            arr = self.datasets.next().numpy()
        frame.ParseFromString(bytearray(arr))
        return frame

    def _load_annot(self, frame, camera_id):
        annotations = np.zeros((0, 5))
        for box_cord in frame.camera_labels:
            import pdb;pdb.set_trace
            if box_cord.name != camera_id:
                continue
            for label in box_cord.labels:

                ll, ww = label.box.length, label.box.width
                rect = [(label.box.center_x - ll // 2), (label.box.center_y - ww // 2),
                        (label.box.center_x + ll // 2), (label.box.center_y + ww // 2)]
                tmp_ann = np.zeros([1, 5])
                tmp_ann[0, :4] = rect
                tmp_ann[0, 4] = self._class(label.type)
                annotations = np.concatenate((annotations, tmp_ann))
        return annotations

    def _load_image(self, frame, camera_id):
        for image in frame.images:
            if image.name == camera_id:
                
                return array_to_img(tf.io.decode_jpeg(image.image))
            
            
    def _load_meta(self, frame, camera_id):

        
        o = metrics_pb2.Object()
        o.camera_name = camera_id
        o.context_name = frame.context.name
        o.frame_timestamp_micros = frame.timestamp_micros
        return o
    
    def __getitem__(self, index):

        frame = self._load_frame(index)


        x = self.cam_id
        image = self._load_image(frame, camera_id=x)

        annotations = self._load_annot(frame, camera_id=x)
        meta = self._load_meta(frame, camera_id=x)

        if self.transform:
            transformed = self.transform({
                "img": image,
                "annot": annotations
            })
            image, annotations = transformed["img"], transformed["annot"]

        return {
            "img": image,
            "annot": annotations,
            "meta": meta
        }

    def __len__(self):
        return 150#len(tf.data.TFRecordDataset(self.tfrecord_path))

    @staticmethod
    def collater(data):
        imgs = [s['img'] for s in data]
        annots = [s['annot'][0] for s in data]
        meta = [s['meta'] for s in data]
        imgs = torch.stack(imgs, axis=0)
        max_num_annots = max(annot.shape[0] for annot in annots)

        if max_num_annots == 0:
            return {
                'img': imgs,
                'annot': torch.FloatTensor(torch.ones((len(annots), 1, 5)) * -1),
                'meta': meta
            }
        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot

        return {'img': imgs, 'annot': torch.FloatTensor(annot_padded), 'meta': meta}

class ToNumpy(object):
    def __call__(self, sample):
        img, annot = sample["img"], sample["annot"]

        img = np.array(img)
        return {
            "img": img,
            "annot": annot
        }


class TopCutter(object):
    def __init__(self, goal):
        self.goal = goal

    def __call__(self, sample):
        img, annot = sample["img"], sample["annot"]
        _, height = img.size
        delta = height - self.goal
        if delta < 0:
            raise ValueError(
                "Invalid goal specified (goal = {})".format(self.goal))
        img = np.array(img)[delta:, :]
        annot[:, 1] -= delta
        annot[:, 3] -= delta
        return {
            "img": img,
            "annot": annot
        }


class Rescale(object):
    """Supports isotropic scaling only."""

    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        img, annot = sample["img"], sample["annot"]
        img = cv2.resize(img, dsize=(img.shape[1] // self.factor,
                                     img.shape[0] // self.factor), interpolation=cv2.INTER_CUBIC)  # / + int cast more precise than //
        annot[:, :4] /= self.factor
        return {
            "img": img,
            "annot": annot
        }

class Rescale(object):
    """Supports isotropic scaling only."""

    def __init__(self, boolean):
        self.boolean = boolean

    def __call__(self, sample):
        img, annot = sample["img"], sample["annot"]


        img = cv2.resize(img, dsize=(img.shape[1] // self.factor,
                                     img.shape[0] // self.factor), interpolation=cv2.INTER_CUBIC) # / + int cast more precise than //
        annot[:, :4] /= self.factor
        return {
            "img": img,
            "annot": annot
        }

class Resize(object):
    """Smaller of side lengths is *goal* after this operation. Isotropic scaling only!"""

    def __init__(self, goal):
        self.goal = goal

    def __call__(self, sample):
        img, annot = sample["img"], sample["annot"]
        goal_frac = img.shape[0] / self.goal
        img = cv2.resize(img, dsize=(int(img.shape[1] / goal_frac),
                                     int(img.shape[0] / goal_frac)), interpolation=cv2.INTER_CUBIC)
        annot[:, :4] /= goal_frac

        return {
            "img": img,
            "annot": annot
        }

class Padder(object):
    """zeropadding"""

    def __init__(self, num_pix):
        self.num_pix = num_pix

    def __call__(self, sample):
        img, annot = sample["img"], sample["annot"]

        img = np.pad(img, ((0, self.num_pix), (0, 0), (0, 0)), 'constant')

        return {
            "img": img,
            "annot": annot
        }

class RandomCrop(object):
    def __init__(self, w, h):
        self.w = w
        self.h = h

    def __call__(self, sample):
        img, annot = sample["img"], sample["annot"]
        rx = 0 if img.shape[0] == self.w else int(torch.randint(
            low=0, high=int(img.shape[0] - self.w), size=(1,))[0])
        ry = 0 if img.shape[1] == self.h else int(torch.randint(
            low=0, high=int(img.shape[1] - self.h), size=(1,))[0])
        img = img[rx:rx+self.w, ry:ry+self.h, :]
        annot[:, 0] -= ry
        annot[:, 2] -= ry
        annot[:, 1] -= rx
        annot[:, 3] -= rx
        return {
            "img": img,
            "annot": annot
        }


class HorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, sample):
        img, annot = sample["img"], sample["annot"]
        rnd = float(np.random.uniform(size=(1,)))
        if(rnd < self.prob):
            img = np.flip(img, axis=1)
            annot[:, 0] = img.shape[1] - annot[:, 0]
            annot[:, 2] = img.shape[1] - annot[:, 2]
        return {
            "img": img,
            "annot": annot
        }


class ToTensor(object):
    def __call__(self, sample):
        img = F.to_tensor(sample["img"].copy())
        img = img
        annot = F.to_tensor(sample["annot"].copy())
        annot = annot
        return {
            "img": img,
            "annot": annot
        }


class Negate(object):
    def __call__(self, sample):
        img = 255 - sample["img"].copy()
        return {
            "img": img,
            "annot": sample["annot"]
        }


class Normalizer(object):
    def __init__(self, mean, std):
        self.mean = np.array([[mean]], dtype=np.float32)
        self.std = np.array([[std]], dtype=np.float32)

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        image = ((image.astype(np.float32) - self.mean) / self.std)
        return {'img': image, 'annot': annots}


class ContrastEnhancementWithNoiseReduction(object):
    def __call__(self, sample):
        img, annot = sample["img"], sample["annot"]
        # note opencv loads image as BGR, not RGB but matplot plots in RGB by default
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # The h channel is responsible for the contrast, so we do an histogram
        # equalisation on this channel
        vChannel = hsv[:, :, 2]
        # this is a histogram specification from opencv, where the target histogram is
        # normal distributed over the full bandwitdh (0 to 255 in gray images)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        hsv[:, :, 2] = clahe.apply(vChannel)
        # Convert back to RGB, since matplotlib is our default plotting method
        RGB = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        # CLAHE introduce noise that we reduce now
        brighted_img = cv2.fastNlMeansDenoisingColored(
            RGB, None, 10, 10, 7, 15)
        return {
            "img": brighted_img,
            "annot": annot
        }

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        image = image + np.random.randn(*image.shape) * self.std + self.mean
        return {'img': image.astype(np.float32), 'annot': annots}


class AddSaltAndPepperNoise(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        rnd = np.random.uniform(size=image.shape)
        noisy = image[:]
        noisy[rnd < self.prob/2] = 0.
        noisy[rnd > 1 - self.prob/2] = 1.

        return {'img': noisy.astype(np.float32), 'annot': annots}

