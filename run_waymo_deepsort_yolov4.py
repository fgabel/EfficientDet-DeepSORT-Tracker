import os
import cv2
import time
import argparse
import numpy as np
from distutils.util import strtobool
from torchvision import transforms
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2
from waymo_open_dataset.utils import frame_utils
from backbone import EfficientDetBackbone
from deep_sort import DeepSort
from deepsort_util import COLORS_10, draw_bboxes

from pathlib import Path
from efficientdet.dataset import TUMuchTrafficDataset, ToTensor, TopCutter, Normalizer, Resize, RandomCrop, \
    AddGaussianNoise, AddSaltAndPepperNoise, HorizontalFlip, Negate, ContrastEnhancementWithNoiseReduction, ToNumpy, \
    TUMuchTrackingDataset
import yaml
from utils.utils import preprocess, invert_affine, postprocess, preprocess_video
from torch.utils.data import DataLoader
from tqdm import tqdm
from tool.utils import *
from tool.darknet2pytorch import Darknet
import torch
import copy


def detect(cfgfile, weightfile, imgfile):
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    num_classes = 80
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'data/names'

    use_cuda = 0
    if use_cuda:
        m.cuda()

    # img = Image.open(imgfile).convert('RGB')

    # sized = img.resize((m.width, m.height))
    w_im = imgfile.shape[1]
    h_im = imgfile.shape[0]

    boxes = np.array(do_detect(m, imgfile, 0.5, 0.4, use_cuda))
    boxes[:, 0] *= w_im
    boxes[:, 1] *= h_im
    boxes[:, 2] *= w_im
    boxes[:, 3] *= h_im
    return boxes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo_cfg", type=str, default="YOLOv3/cfg/yolo_v3.cfg")
    parser.add_argument("--yolo_weights", type=str, default="YOLOv3/yolov3.weights")
    parser.add_argument("--yolo_names", type=str, default="YOLOv3/cfg/coco.names")
    parser.add_argument("--conf_thresh", type=float, default=0.5)
    parser.add_argument("--nms_thresh", type=float, default=0.4)

    parser.add_argument('-p', '--project', type=str, default='waymo',
                        help='project file that contains parameters')
    parser.add_argument('-c', '--compound_coef', type=int,
                        default=0, help='coefficients of efficientdet')
    parser.add_argument('-n', '--num_workers', type=int,
                        default=0, help='num_workers of dataloader')
    parser.add_argument('--batch_size', type=int, default=12,
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
                        default='E:/data/waymo/', help='the root folder of dataset')
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
    parser.add_argument("--save_path", type=str, default="ped.avi")  # )
    parser.add_argument("--use_cuda", type=str, default="True")
    args = parser.parse_args()
    return args


data_path = "../val_data"
tfrecord_paths = [data_path] if data_path.endswith(".tfrecord") else [str(x.absolute()) for x in
                                                                      Path(data_path).rglob('*.tfrecord')]
use_cuda = True
use_float16 = False

compound_coef = 4
force_input_size = None  # set None to use default size

training_params = {'batch_size': 1,
                   'shuffle': True,
                   'drop_last': True,
                   'collate_fn': TUMuchTrackingDataset.collater,
                   'num_workers': 0}

tfs = transforms.Compose([
    ToNumpy(),
    # transforms.RandomApply([Negate()], p=0.1),
    # transforms.RandomApply([ContrastEnhancementWithNoiseReduction()], p=0.1),
    Resize(1024),
    # RandomCrop(256, 512),
    # Normalizer(params.mean, params.std),
    ToTensor()
])

yolo_to_waymo_classes = {0: 2, 2: 1, 1: 4, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1}


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
        self.cam_id = 1
        if args.display:
            pass
            # cv2.namedWindow("test", cv2.WINDOW_NORMAL)
            # cv2.resizeWindow("test", args.display_width, args.display_height)

        self.vdo = cv2.VideoCapture()

        self.deepsort = DeepSort(args.deepsort_checkpoint, use_cuda=use_cuda)
        self.class_names = load_class_names('data/coco.names')
        self.submit = True
        self.object_list = []

    def __enter__(self):
        self.im_width = 1536
        self.im_height = 1024

        if self.args.save_path:
            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            self.output = cv2.VideoWriter(self.args.save_path, fourcc, 10, (self.im_width, self.im_height))

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def detect(self):
        m = Darknet("./cfg/yolov4.cfg").cuda()

        m.print_network()
        m.load_weights("yolov4.weights")
        for i, tfrecord in enumerate(tqdm(tfrecord_paths)):
            self.object_list = []
            self.object_list_tracks = []
            training_set = TUMuchTrackingDataset(
                tfrecord_path=tfrecord, transform=tfs, cam_id=self.cam_id)
            training_generator = DataLoader(training_set, **training_params)

            for it, data in enumerate(training_generator):
                o_list = []

                ori_im = data['img'].cpu().numpy()

                if self.submit:
                    meta = data['meta']

                im = ori_im[0, :, :, :]

                im = np.swapaxes(im, 0, 2)
                im = np.swapaxes(im, 0, 1)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im = im * 255
                im = im.astype(np.uint8)

                # img = Image.open(imgfile).convert('RGB')

                # sized = img.resize((m.width, m.height))
                w_im = im.shape[1]
                h_im = im.shape[0]

                boxes = np.array(do_detect(m, im, 0.5, 0.4, use_cuda=True))
                if len(boxes) == 0:
                    continue
                boxes[:, 0] *= w_im
                boxes[:, 1] *= h_im
                boxes[:, 2] *= w_im
                boxes[:, 3] *= h_im

                bbox_xcycwh, cls_conf, cls_ids = boxes[:, :4], boxes[:, 5], boxes[:, 6]

                if bbox_xcycwh is not None:
                    # select class car and truck
                    mask = cls_ids <= 7  # == (2 or 1 or 0)
                    cls_ids = cls_ids[mask]
                    bbox_xcycwh = bbox_xcycwh[mask]
                    bbox_xcycwh[:, 3:] *= 1.2

                    cls_conf = cls_conf[mask]
                    outputs = self.deepsort.update(bbox_xcycwh, cls_conf, cls_ids, im)

                    if len(outputs) > 0:

                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -2]
                        track_class = outputs[:, -1]

                        if self.submit:
                            for box_idx in range(bbox_xyxy.shape[0]):
                                o = meta[:][0]
                                box = label_pb2.Label.Box()

                                box.center_x = (bbox_xyxy[box_idx, 0] + bbox_xyxy[box_idx, 2]) / 1.6
                                box.center_y = (bbox_xyxy[box_idx, 1] + bbox_xyxy[box_idx, 3]) / 1.6
                                box.length = (bbox_xyxy[box_idx, 2] - bbox_xyxy[box_idx, 0]) * 1.25
                                box.width = (bbox_xyxy[box_idx, 3] - bbox_xyxy[box_idx, 1]) * 1.25
                                o.object.box.CopyFrom(box)
                                o.score = 0.9  # CHECK THIS
                                # Use correct type.

                                o.object.type = yolo_to_waymo_classes[
                                    track_class[box_idx]]  # MAP THIS TO CORRECT CLASSES

                                o.object.id = str(identities[box_idx])
                                self.object_list_tracks.append(copy.deepcopy(o))
                                import pdb;
                                pdb.set_trace()
                        ori_im = draw_bboxes(im, bbox_xyxy, track_class)

                if self.args.display:
                    pass
                    # cv2.imshow("test", ori_im)
                    # cv2.waitKey(1)

                if self.args.save_path:
                    self.output.write(im)
            objects = metrics_pb2.Objects()
            # write object detection stuff
            for o in self.object_list_tracks:
                objects.objects.append(o)
            f = open("./output/tracking/sub_camid_{}_yolo.bin".format(self.cam_id), 'ab')
            f.write(objects.SerializeToString())
            f.close()


if __name__ == "__main__":
    args = parse_args()
    with Detector(args) as det:
        det.detect()
