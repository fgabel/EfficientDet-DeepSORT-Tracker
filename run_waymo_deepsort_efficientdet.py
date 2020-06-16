import os
import cv2
import time
import argparse
import numpy as np
from distutils.util import strtobool
from torchvision import transforms
import torch
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2
from waymo_open_dataset.utils import frame_utils
from backbone import EfficientDetBackbone
from deep_sort import DeepSort
from deepsort_util import COLORS_10, draw_bboxes
from pathlib import Path
from efficientdet.dataset import TUMuchTrackingDataset, ToTensor, TopCutter, Normalizer, Resize, RandomCrop, \
    AddGaussianNoise, AddSaltAndPepperNoise, HorizontalFlip, Negate, ContrastEnhancementWithNoiseReduction, ToNumpy, \
    Padder
import yaml
from utils.utils import preprocess, invert_affine, postprocess, preprocess_video
from torch.utils.data import DataLoader
from efficientdet.utils import BBoxTransform, ClipBoxes
from tqdm import tqdm
import copy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf_thresh", type=float, default=0.5)
    parser.add_argument("--nms_thresh", type=float, default=0.4)

    parser.add_argument('-p', '--project', type=str, default='waymo',
                        help='project file that contains parameters')
    parser.add_argument('-c', '--compound_coef', type=int,
                        default=2, help='coefficients of efficientdet')
    parser.add_argument('-n', '--num_workers', type=int,
                        default=0, help='num_workers of dataloader')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='The number of images per batch among all devices')
    parser.add_argument('--head_only', type=bool, default=False,
                        help='whether finetunes only the regressor and the classifier, '
                             'useful in early stage convergence or small/easy dataset')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optim', type=str, default='adamw', help='select optimizer for training, '
                                                                   'suggest using \'admaw\' until the'
                                                                   ' very final stage then switch to \'sgd\'')
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--val_interval', type=int, default=1,
                        help='Number of epoches between valing phases')
    parser.add_argument('--save_interval', type=int,
                        default=500, help='Number of steps between saving')
    parser.add_argument('--es_min_delta', type=float, default=0.0,
                        help='Early stopping\'s parameter: minimum change loss to qualify as an improvement')
    parser.add_argument('--es_patience', type=int, default=0,
                        help='Early stopping\'s parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.')
    parser.add_argument('--data_path', type=str,
                        default='datasets/', help='the root folder of dataset')
    parser.add_argument('--val_path', type=str,
                        default='datasets/', help='the root folder of dataset')

    parser.add_argument('--log_path', type=str, default='logs/')
    parser.add_argument('-w', '--load_weights', type=str, default=None,
                        help='whether to load weights from a checkpoint, set None to initialize, set \'last\' to load last checkpoint')
    parser.add_argument('--saved_path', type=str, default='logs/')

    # above this line, all arguments from efficientdet, below this line from DEEPSORT
    parser.add_argument("--deepsort_checkpoint", type=str, default="deep_sort/deep/checkpoint/ckpt.t7")
    parser.add_argument("--max_dist", type=float, default=0.2)
    parser.add_argument("--ignore_display", dest="display", action="store_false")
    parser.add_argument("--display_width", type=int, default=800)
    parser.add_argument("--display_height", type=int, default=600)
    parser.add_argument("--save_path", type=str, default="d2_00.avi")  # "demo.avi")
    parser.add_argument("--use_cuda", type=str, default="True")
    parser.add_argument("--detector_weights_path", type=str, default="d2_Loss021.pth")
    args = parser.parse_args()
    return args


data_path = "../val_data"
tfrecord_paths = [data_path] if data_path.endswith(".tfrecord") else [str(x.absolute()) for x in
                                                                      Path(data_path).rglob('*.tfrecord')]
use_cuda = True
use_float16 = False

threshold = 0.2
iou_threshold = 0.2
training_params = {'batch_size': 1,
                   'shuffle': True,
                   'drop_last': True,
                   'collate_fn': TUMuchTrackingDataset.collater,
                   'num_workers': 0}

tfs = transforms.Compose([
    ToNumpy(),
    # transforms.RandomApply([Negate()], p=0.1),
    # transforms.RandomApply([ContrastEnhancementWithNoiseReduction()], p=0.1),
    Resize(1280),  # 895
    # Padder(10), # only for cam_id 3 and 4
    # RandomCrop(256, 512),
    # Normalizer(params.mean, params.std),
    ToTensor()
])

to_waymo_classes = {0: 1, 1: 2, 2: 4}

regressBoxes = BBoxTransform()
clipBoxes = ClipBoxes()


class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


class Detector(object):
    def __init__(self, args):
        self.args = args
        use_cuda = bool(strtobool(self.args.use_cuda))
        params = Params(f'projects/{self.args.project}.yml')
        self.submit = True
        self.cam_id = 1
        self.object_list = []
        self.object_list_tracks = []
        if args.display:
            pass
            # cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            # cv2.resizeWindow("test", args.display_width, args.display_height)

        self.vdo = cv2.VideoCapture()
        self.efficientdet = EfficientDetBackbone(num_classes=len(params.obj_list),
                                                 compound_coef=self.args.compound_coef,
                                                 ratios=eval(params.anchors_ratios),
                                                 scales=eval(params.anchors_scales)).cuda()
        # self.yolo3 = YOLOv3(args.yolo_cfg, args.yolo_weights, args.yolo_names, is_xywh=True, conf_thresh=args.conf_thresh, nms_thresh=args.nms_thresh, use_cuda=use_cuda)

        self.deepsort = DeepSort(args.deepsort_checkpoint, use_cuda=True)
        # self.class_names = self.yolo3.class_names
        self.efficientdet.load_state_dict(torch.load(args.detector_weights_path), strict=False)

    def __enter__(self):
        self.im_width = 1920
        self.im_height = 1280

        if self.args.save_path:
            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            self.output = cv2.VideoWriter(self.args.save_path, fourcc, 10, (self.im_width, self.im_height))

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def detect(self):
        for tf_idx, tfrecord in enumerate(tqdm(tfrecord_paths[2:])):
            self.object_list = []
            self.object_list_tracks = []
            training_set = TUMuchTrackingDataset(
                tfrecord_path=tfrecord, transform=tfs, cam_id=self.cam_id)
            training_generator = DataLoader(training_set, **training_params)
            for it, data in enumerate(training_generator):

                imgs = data['img'].to(torch.device("cuda:0"))

                if self.submit:
                    meta = data['meta']
                with torch.no_grad():
                    features, regression, classification, anchors = self.efficientdet(imgs)

                out = postprocess(imgs,
                                  anchors, regression, classification,
                                  regressBoxes, clipBoxes,
                                  threshold, iou_threshold)
                # boxes is cx, cy, cw, ch

                boxes = out[0]["rois"]
                for idx in range(out[0]["rois"].shape[0]):
                    cx, cy, lx, ly = out[0]["rois"][idx]
                    cw, ch = lx - cx, ly - cy

                    boxes[idx][0] = cx + cw / 2
                    boxes[idx][1] = cy + ch / 2
                    boxes[idx][2] = cw
                    boxes[idx][3] = ch
                bbox_xcycwh, cls_conf, cls_ids = boxes, out[0]["scores"], out[0]["class_ids"]

                if bbox_xcycwh is not None:

                    mask = cls_ids <= 4

                    bbox_xcycwh = bbox_xcycwh[mask]
                    try:
                        bbox_xcycwh[:, 3:] *= 1
                    except:
                        continue

                    cls_conf = cls_conf[mask]

                    im = imgs.cpu().numpy()
                    im = im[0, :, :, :]

                    im = np.swapaxes(im, 0, 2)
                    im = np.swapaxes(im, 0, 1)
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    im = im * 255
                    im = im.astype(np.uint8)
                    outputs = self.deepsort.update(bbox_xcycwh, cls_conf, out[0]["class_ids"], im)
                    if len(outputs) > 0:

                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -2]
                        track_class = outputs[:, -1]

                        if self.submit:
                            for box_idx in range(bbox_xyxy.shape[0]):
                                o = meta[:][0]

                                box = label_pb2.Label.Box()

                                box.center_x = (bbox_xyxy[box_idx, 0] + bbox_xyxy[box_idx, 2]) / 2
                                box.center_y = (bbox_xyxy[box_idx, 1] + bbox_xyxy[box_idx, 3]) / 2
                                box.length = (bbox_xyxy[box_idx, 2] - bbox_xyxy[box_idx, 0])
                                box.width = (bbox_xyxy[box_idx, 3] - bbox_xyxy[box_idx, 1])

                                o.object.box.CopyFrom(box)
                                o.score = 0.9  # CHECK THIS
                                # Use correct type.

                                o.object.type = to_waymo_classes[track_class[box_idx]]  # MAP THIS TO CORRECT CLASSES

                                self.object_list.append(copy.deepcopy(o))

                                o.object.id = str(identities[box_idx])
                                self.object_list_tracks.append(copy.deepcopy(o))
                                # import pdb; pdb.set_trace()
                        if self.args.save_path:
                            draw_bboxes(im, bbox_xyxy, identities)

                if self.args.display:
                    pass

                self.args.save_path = "cam_{}.avi".format(self.cam_id)
                if self.args.save_path:
                    self.output.write(im)
            objects = metrics_pb2.Objects()
            # write object detection stuff
            for o in self.object_list:
                objects.objects.append(o)
            f = open("./output/detection/sub_camid_{}.bin".format(self.cam_id), 'ab')
            f.write(objects.SerializeToString())
            f.close()
            objects = metrics_pb2.Objects()
            # write object detection stuff
            for o in self.object_list_tracks:
                objects.objects.append(o)
            f = open("./output/tracking/sub_camid_{}.bin".format(self.cam_id), 'ab')
            f.write(objects.SerializeToString())
            f.close()


if __name__ == "__main__":
    args = parse_args()
    with Detector(args) as det:
        det.detect()
