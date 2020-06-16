import pdb
from mean_average_precision.utils.show_frame import show_frame
from mean_average_precision.detection_map import DetectionMAP
from functools import reduce
from apex import amp
from utils.utils import replace_w_sync_bn, CustomDataParallel, get_last_weights, init_weights
from utils.sync_batchnorm import patch_replication_callback
from efficientdet.loss import FocalLoss
from tqdm.autonotebook import tqdm
import numpy as np
from tensorboardX import SummaryWriter
from backbone import EfficientDetBackbone
from efficientdet.dataset import TUMuchTrafficDataset, ToTensor, TopCutter, Normalizer, Resize, RandomCrop, \
    AddGaussianNoise, AddSaltAndPepperNoise, HorizontalFlip, Negate, ContrastEnhancementWithNoiseReduction
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
import yaml
import torch
from pathlib import Path
import traceback
import argparse
import datetime
from utils.utils import preprocess, invert_affine, postprocess, preprocess_video
from efficientdet.utils import BBoxTransform, ClipBoxes
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


def get_args():
    parser = argparse.ArgumentParser(
        'Yet Another EfficientDet Pytorch: SOTA object detection network - Zylo117')
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
                        default='datasets/', help='the root folder of dataset')
    parser.add_argument('--val_path', type=str,
                        default='datasets/', help='the root folder of dataset')

    parser.add_argument('--log_path', type=str, default='logs/')
    parser.add_argument('-w', '--load_weights', type=str, default=None,
                        help='whether to load weights from a checkpoint, set None to initialize, set \'last\' to load last checkpoint')
    parser.add_argument('--load_backbone', type=bool, default=False,
                        help='whether to load backbone weights. checkpoints not supported')
    parser.add_argument('--no_effnet', type=bool, default=False,
                        help='only train non-backbone network parts, i.e. bifpn and classifier/regressor')
    parser.add_argument('--advprop', type=bool, default=False,
                        help='use advprop-trained backbone')
    parser.add_argument('--saved_path', type=str, default='logs/')
    parser.add_argument('--debug', type=bool, default=False, help='whether visualize the predicted boxes of trainging, '
                                                                  'the output images will be in test/')

    args = parser.parse_args()
    return args


class ModelWithLoss(nn.Module):
    def __init__(self, model, debug=False):
        super().__init__()
        self.criterion = FocalLoss()
        self.model = model
        self.debug = debug

    def forward(self, imgs, annotations, obj_list=None):
        _, regression, classification, anchors = self.model(imgs)
        if self.debug:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations,
                                                imgs=imgs, obj_list=obj_list)
        else:
            cls_loss, reg_loss = self.criterion(
                classification, regression, anchors, annotations)
        return cls_loss, reg_loss


def calc_mAP(imgs, annot, model, writer, it, local_it):
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()
    orig_train = model.training
    model.eval()
    threshold = 0.2
    iou_threshold = 0.2
    with torch.no_grad():
        features, regression, classification, anchors = model.model(imgs)
        # get max. confidence
        # there are batch_size out dicts
        out = postprocess(imgs,
                          anchors, regression, classification,
                          regressBoxes, clipBoxes,
                          threshold, iou_threshold)
    mAP_list = []
    for im_idx in range(len(out)):  # iterate through images
        # (1) check ground truth
        curr_annot = annot[im_idx, :, :]  # e.g. (8,5)
        curr_annot = curr_annot[curr_annot[..., 0] != -1.]  # e.g. (6,5)
        gt_boxes = curr_annot[:, :4]
        gt_cls = curr_annot[:, 4]
        if gt_cls.shape[0] == 0:
            continue
        # (2) check prediction
        out_ = out[im_idx]
        pred_boxes = out_['rois']
        pred_classes = out_['class_ids']
        pred_scores = out_['scores']
        curr_img = imgs[im_idx]
        # (3) build map tuple
        map_tuple = tuple()
        map_tuple += (pred_boxes / curr_img.shape[1],)
        map_tuple += (pred_classes,)
        map_tuple += (pred_scores,)
        map_tuple += (gt_boxes.cpu() / curr_img.shape[2],)
        map_tuple += (gt_cls.cpu(),)
        if map_tuple is not None:
            mAP_list.append(map_tuple)
    mAP = DetectionMAP(3)
    overall_mAP = []
    classwise_mAP = []
    for mAP_item in mAP_list:
        mAP.evaluate(*mAP_item)
        ov_, cls_ = mAP.map(class_names=["vehicle", "pedestrian", "cyclist"])
        overall_mAP.append(ov_)
        classwise_mAP.append(cls_)
    key_arrays = {}
    for item in classwise_mAP:
        for k, v in item.items():
            key_arrays.setdefault(k, []).append(v)
    ave = {k: reduce(lambda x, y: x + y, v) / len(v)
           for k, v in key_arrays.items()}

    if len(overall_mAP) > 0:
        writer.add_scalars('mAP', {'val': np.mean(overall_mAP)}, it)
    for k in ave.keys():
        writer.add_scalars('val mAP {}'.format(k), {'val': ave[k]}, it)
    if orig_train:
        model.train()
    return np.mean(overall_mAP)


def plot_tensorboard(imgs, annot, model, writer, it, local_it, mAP_value):
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()
    orig_train = model.training
    model.eval()
    with torch.no_grad():
        features, regression, classification, anchors = model.model(imgs)
        # get max. confidence
        max_pred = 1e9
        out = postprocess(imgs,
                          anchors, regression, classification,
                          regressBoxes, clipBoxes,
                          0, 0, max_pred)[0]
        try:
            max_confidence = np.array(out['scores']).max()
        except:
            max_confidence = 0
        # filter out trash predictions
        threshold = 0.2
        iou_threshold = 0.2
        out = postprocess(imgs,
                          anchors, regression, classification,
                          regressBoxes, clipBoxes,
                          threshold, iou_threshold, max_pred)[0]
        boxes = out['rois']
        img = imgs[0].detach().cpu().numpy()
        img = (img - img.min())
        img = img / img.max()
        fig, ax = plt.subplots(1, 1)
        ax.axis("off")
        colors = ["red", "green", "blue", "black"]
        ax.imshow(torch.from_numpy(img).permute(1, 2, 0))
        for idx in range(boxes.shape[0]):
            cx, cy, lx, ly = boxes[idx]
            cw, ch = lx - cx, ly - cy
            class_idx = out["class_ids"][idx]
            if class_idx < 3:
                color = colors[class_idx]
            else:
                color = colors[-1]
            rect = patches.Rectangle(
                (cx, cy), cw, ch, linewidth=1, edgecolor=color, facecolor='none')
            ax.add_patch(rect)

        for idx in range(annot.shape[1]):
            curr_annot = annot[0, idx]
            cx, cy, lx, ly = curr_annot[:4]
            cw, ch = lx - cx, ly - cy
            rect = patches.Rectangle(
                (cx, cy), cw, ch, linewidth=1, edgecolor='none', facecolor='white', alpha=0.2)
            ax.add_patch(rect)
        ax.set_title("Max. Confidence: {:.3f}.".format(
            max_confidence))
        fig.canvas.draw()
        img = torch.from_numpy(np.array(fig.canvas.renderer._renderer)[
                               :, :, :3]).permute(2, 0, 1)
        plt.close()
        writer.add_image('Prediction/{}'.format(local_it), img, it)
    if orig_train:
        model.train()


def train(opt):
    params = Params(f'projects/{opt.project}.yml')
    global_validation_it = 0

    if params.num_gpus == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    opt.saved_path = opt.saved_path + f'/{params.project_name}/'
    opt.log_path = opt.log_path + f'/{params.project_name}/tensorboard/'
    os.makedirs(opt.log_path, exist_ok=True)
    os.makedirs(opt.saved_path, exist_ok=True)

    training_params = {'batch_size': opt.batch_size,
                       'shuffle': True,
                       'drop_last': True,
                       'collate_fn': TUMuchTrafficDataset.collater,
                       'num_workers': opt.num_workers}

    val_params = {'batch_size': opt.batch_size,
                  'shuffle': False,
                  'drop_last': True,
                  'collate_fn': TUMuchTrafficDataset.collater,
                  'num_workers': opt.num_workers}

    advprop = opt.advprop
    if advprop:  # for models using advprop pretrained weights
        normalize = transforms.Lambda(lambda mem: {"img": (mem["img"] * 2.0 - 1.0).astype(np.float32),
                                                   "annot": mem["annot"]})
    else:  # for other models
        normalize = Normalizer(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])

    tfs = transforms.Compose([
        TopCutter(886),
        transforms.RandomApply([Negate()], p=0.1),
        transforms.RandomApply([ContrastEnhancementWithNoiseReduction()], p=0.1),
        Resize(384),
        RandomCrop(384, 768),
        normalize,
        HorizontalFlip(prob=0.5),
        transforms.RandomApply([AddGaussianNoise(0, 2.55)], p=0.5),
        transforms.RandomApply([AddSaltAndPepperNoise(prob=0.0017)], p=0.5),
        ToTensor()
    ])
    tfrecord_paths = [opt.data_path] if opt.data_path.endswith(".tfrecord") else [str(x.absolute()) for x in
                                                                                  Path(opt.data_path).rglob(
                                                                                      '*.tfrecord')]
    training_set = TUMuchTrafficDataset(
        tfrecord_paths=tfrecord_paths, transform=tfs)
    training_generator = DataLoader(training_set, **training_params)

    tfrecord_paths = [opt.data_path] if opt.data_path.endswith(".tfrecord") else [str(x.absolute()) for x in
                                                                                  Path(opt.val_path).rglob(
                                                                                      '*.tfrecord')]
    val_set = TUMuchTrafficDataset(
        tfrecord_paths=tfrecord_paths, transform=tfs)
    val_generator = DataLoader(val_set, **val_params)

    if not opt.load_backbone:
        load_weights = False
    else:
        load_weights = True
    model = EfficientDetBackbone(num_classes=len(params.obj_list), compound_coef=opt.compound_coef,
                                 ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales),
                                 load_weights=load_weights)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("# Params: {:08d}".format(pytorch_total_params))

    # load last weights
    if opt.load_weights is not None:
        if opt.load_weights.endswith('.pth'):
            weights_path = opt.load_weights
        else:
            weights_path = get_last_weights(opt.saved_path)
        try:
            last_step = int(os.path.basename(
                weights_path).split('_')[-1].split('.')[0])
        except:
            last_step = 0

        try:
            ret = model.load_state_dict(torch.load(weights_path), strict=False)
        except RuntimeError as e:
            print(f'[Warning] Ignoring {e}')
            print(
                '[Warning] Don\'t panic if you see this, this might be because you load a pretrained weights with different number of classes. The rest of the weights should be loaded already.')

        print(
            f'[Info] loaded weights: {os.path.basename(weights_path)}, resuming checkpoint from step: {last_step}')
    else:
        last_step = 0
        print('[Info] initializing weights...')
        init_weights(model)

    # freeze backbone if train head_only
    if opt.head_only:
        def freeze_backbone(m):
            classname = m.__class__.__name__
            for ntl in ['EfficientNet', 'BiFPN']:
                if ntl in classname:
                    for param in m.parameters():
                        param.requires_grad = False

        model.apply(freeze_backbone)
        print('[Info] freezed backbone')

    # freeze backbone (only efficientnet) if train no_effnet
    if opt.no_effnet:
        def freeze_backbone(m):
            classname = m.__class__.__name__
            for ntl in ['EfficientNet']:
                if ntl in classname:
                    for param in m.parameters():
                        param.requires_grad = False

        model.apply(freeze_backbone)
        print('[Info] freezed backbone')
    pytorch_total_params = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)
    print("# Training Parameters: {:06}".format(pytorch_total_params))

    # https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
    # apply sync_bn when using multiple gpu and batch_size per gpu is lower than 4
    #  useful when gpu memory is limited.
    # because when bn is disable, the training will be very unstable or slow to converge,
    # apply sync_bn can solve it,
    # by packing all mini-batch across all gpus as one batch and normalize, then send it back to all gpus.
    # but it would also slow down the training by a little bit.
    if params.num_gpus > 1 and opt.batch_size // params.num_gpus < 4:
        model.apply(replace_w_sync_bn)
        use_sync_bn = True
    else:
        use_sync_bn = False

    writer = SummaryWriter(
        opt.log_path + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M")}/')

    # warp the model with loss function, to reduce the memory usage on gpu0 and speedup
    model = ModelWithLoss(model, debug=opt.debug)

    if params.num_gpus > 0:
        model = model.cuda()
        if params.num_gpus > 1:
            model = CustomDataParallel(model, params.num_gpus)
            if use_sync_bn:
                patch_replication_callback(model)

    if opt.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), opt.lr)
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), opt.lr, momentum=0.9, nesterov=True)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=1e6, verbose=True)

    # use apex for mixed precision training
    # model, optimizer = amp.initialize(model, optimizer)

    epoch = 0
    best_loss = 1e5
    best_epoch = 0
    step = max(0, last_step)
    model.train()

    num_iter_per_epoch = len(training_generator)

    try:
        for epoch in range(opt.num_epochs):
            last_epoch = step // num_iter_per_epoch
            if epoch < last_epoch:
                continue

            epoch_loss = []
            progress_bar = tqdm(training_generator)
            for it, data in enumerate(progress_bar):
                if it < step - last_epoch * num_iter_per_epoch:
                    progress_bar.update()
                    continue
                try:
                    imgs = data['img']
                    annot = data['annot']

                    if params.num_gpus == 1:
                        # if only one gpu, just send it to cuda:0
                        # elif multiple gpus, send it to multiple gpus in CustomDataParallel, not here
                        imgs = imgs.cuda()
                        annot = annot.cuda()

                    global_validation_it += 1
                    optimizer.zero_grad()

                    cls_loss, reg_loss = model(imgs, annot)
                    cls_loss = cls_loss.mean()
                    reg_loss = reg_loss.mean()

                    loss = cls_loss + reg_loss
                    if loss == 0 or not torch.isfinite(loss):
                        continue

                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer.step()

                    epoch_loss.append(float(loss))

                    progress_bar.set_description(
                        'Step: {}. Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Total loss: {:.5f}'.format(
                            step, epoch, opt.num_epochs, it + 1, num_iter_per_epoch, cls_loss.item(),
                            reg_loss.item(), loss.item()))
                    writer.add_scalars('Loss', {'train': loss}, step)
                    writer.add_scalars('Regression_loss', {
                        'train': reg_loss}, step)
                    writer.add_scalars('Classfication_loss', {
                        'train': cls_loss}, step)

                    # log learning_rate
                    current_lr = optimizer.param_groups[0]['lr']
                    writer.add_scalar('learning_rate', current_lr, step)

                    step += 1

                    if step % opt.save_interval == 0 and step > 0:
                        save_checkpoint(
                            model, f'efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth')
                        print('checkpoint...')

                except Exception as e:
                    print('[Error]', traceback.format_exc())
                    print(e)
                    continue
            scheduler.step(np.mean(epoch_loss))

            # sleep for 30 seconds, to reduce overheating
            import time
            time.sleep(30)

            if epoch % opt.val_interval == 0:
                model.eval()
                loss_regression_ls = []
                loss_classification_ls = []
                for it, data in enumerate(val_generator):
                    with torch.no_grad():
                        imgs = data['img']
                        annot = data['annot']
                        if params.num_gpus == 1:
                            # if only one gpu, just send it to cuda:0
                            # elif multiple gpus, send it to multiple gpus in CustomDataParallel, not here
                            imgs = imgs.cuda()
                            annot = annot.cuda()
                        if it < 12:
                            plot_tensorboard(
                                imgs, annot, model, writer, global_validation_it, it, "")
                            global_validation_it += 1

                        if params.num_gpus == 1:
                            imgs = imgs.cuda()
                            annot = annot.cuda()

                        cls_loss, reg_loss = model(
                            imgs, annot, obj_list=params.obj_list)
                        cls_loss = cls_loss.mean()
                        reg_loss = reg_loss.mean()

                        loss = cls_loss + reg_loss
                        if loss == 0 or not torch.isfinite(loss):
                            continue

                        loss_classification_ls.append(cls_loss.item())
                        loss_regression_ls.append(reg_loss.item())

                cls_loss = np.mean(loss_classification_ls)
                reg_loss = np.mean(loss_regression_ls)
                loss = cls_loss + reg_loss

                print(
                    'Val. Epoch: {}/{}. Classification loss: {:1.5f}. Regression loss: {:1.5f}. Total loss: {:1.5f}'.format(
                        epoch, opt.num_epochs, cls_loss, reg_loss, loss))
                writer.add_scalars('Loss', {'val': loss}, step)
                writer.add_scalars('Regression_loss', {'val': reg_loss}, step)
                writer.add_scalars('Classfication_loss', {
                    'val': cls_loss}, step)

                if loss + opt.es_min_delta < best_loss:
                    best_loss = loss
                    best_epoch = epoch

                    save_checkpoint(
                        model, f'efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth')

                model.train()

                # Early stopping
                if epoch - best_epoch > opt.es_patience > 0:
                    print('[Info] Stop training at epoch {}. The lowest loss achieved is {}'.format(
                        epoch, best_loss))
                    break
    except KeyboardInterrupt:
        save_checkpoint(
            model, f'efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth')
        writer.close()
    writer.close()


def save_checkpoint(model, name):
    if isinstance(model, CustomDataParallel):
        torch.save(model.module.model.state_dict(),
                   os.path.join(opt.saved_path, name))
    else:
        torch.save(model.model.state_dict(),
                   os.path.join(opt.saved_path, name))


if __name__ == '__main__':
    opt = get_args()
    train(opt)
