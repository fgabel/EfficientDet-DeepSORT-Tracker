import glob
import json
import os
import shutil
import operator
import sys
import argparse
import math
import torch

import numpy as np

MINOVERLAP = 0.5 # default value (defined in the PASCAL VOC2012 challenge)

parser = argparse.ArgumentParser()
parser.add_argument('-np', '--no-plot', help="no plot is shown.", action="store_true")
parser.add_argument('-q', '--quiet', help="minimalistic console output.", action="store_true")
# argparse receiving list of classes to be ignored (e.g., python main.py --ignore person book)
parser.add_argument('-i', '--ignore', nargs='+', type=str, help="ignore a list of classes.")
# argparse receiving list of classes with specific IoU (e.g., python main.py --set-class-iou person 0.7)
parser.add_argument('--set-class-iou', nargs='+', type=str, help="set IoU for a specific class.")
args = parser.parse_args()

'''
    0,0 ------> x (width)
     |
     |  (Left,Top)
     |      *_________
     |      |         |
            |         |
     y      |_________|
  (height)            *
                (Right,Bottom)
'''

# if there are no classes to ignore then replace None by empty list
if args.ignore is None:
    args.ignore = []

specific_iou_flagged = False
if args.set_class_iou is not None:
    specific_iou_flagged = True

# make sure that the cwd() is the location of the python script (so that every path makes sense)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

"""
 throw error and exit
"""
def error(msg):
    print(msg)
    sys.exit(0)

"""
 check if the number is a float between 0.0 and 1.0
"""
def is_float_between_0_and_1(value):
    try:
        val = float(value)
        if val > 0.0 and val < 1.0:
            return True
        else:
            return False
    except ValueError:
        return False

gt_file = "input/ground-truth/test1.txt"
"""
 Calculate the AP given the recall and precision array
    1st) We compute a version of the measured precision/recall curve with
         precision monotonically decreasing
    2nd) We compute the AP as the area under this curve by numerical integration.
"""
def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #     range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #     range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre


"""
 Convert the lines of a file to a list
"""
def file_lines_to_list(path):
    # open txt file lines to a list
    with open(path) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content

"""
 ground-truth
"""
# dictionary with counter per class
gt_counter_per_class = {}
counter_images_per_class = {}

def load_ground_truth():
    gt_per_class = {}
    txt_file = gt_file
    #print(txt_file)
    lines_list = file_lines_to_list(txt_file)
    # create ground-truth dictionary
    bounding_boxes = []
    already_seen_classes = []
    for line in lines_list:
        try:
            if "difficult" in line:
                    class_name, left, top, right, bottom, _difficult = line.split()
                    is_difficult = True
            else:
                    class_name, left, top, right, bottom = line.split()
        except ValueError:
            error_msg = "Error: File " + txt_file + " in the wrong format.\n"
            error_msg += " Expected: <class_name> <left> <top> <right> <bottom> ['difficult']\n"
            error_msg += " Received: " + line
            error_msg += "\n\nIf you have a <class_name> with spaces between words you should remove them\n"
            error_msg += "by running the script \"remove_space.py\" or \"rename_class.py\" in the \"extra/\" folder."
            error(error_msg)
        # check if class is in the ignore list, if yes skip
        if class_name in args.ignore:
            continue
        bbox = left + " " + top + " " + right + " " +bottom
        bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False})
        # count that object
        if class_name in gt_counter_per_class:
            gt_counter_per_class[class_name] += 1
        else:
            # if class didn't exist yet
            gt_counter_per_class[class_name] = 1

        if class_name not in already_seen_classes:
            if class_name in counter_images_per_class:
                counter_images_per_class[class_name] += 1
            else:
                # if class didn't exist yet
                counter_images_per_class[class_name] = 1
            already_seen_classes.append(class_name)

    for class_name in already_seen_classes:
        class_boxes = [x for x in bounding_boxes if x["class_name"] == class_name]
        tensor = torch.zeros([len(class_boxes), 4])
        for idx in range(len(class_boxes)):
            vals = class_boxes[idx]["bbox"].split()
            tensor[idx] = torch.FloatTensor( [float(vals[x]) for x in range(4)])
        gt_per_class[class_name] = tensor
    return gt_per_class

def load_detections(filename):
    box_per_class = {}
    for class_index, class_name in enumerate(gt_classes):
        bounding_boxes = []

        # the first time it checks if all the corresponding ground-truth files exist
        lines = file_lines_to_list(filename)
        for line in lines:
            try:
                tmp_class_name, confidence, left, top, right, bottom = line.split()
            except ValueError:
                error_msg = "Error: File " + dr_file + " in the wrong format.\n"
                error_msg += " Expected: <class_name> <confidence> <left> <top> <right> <bottom>\n"
                error_msg += " Received: " + line
                error(error_msg)
            if tmp_class_name == class_name:
                #print("match")
                bbox = left + " " + top + " " + right + " " +bottom
                bounding_boxes.append({"confidence":confidence, "bbox":bbox})
                #print(bounding_boxes)
        # sort detection-results by decreasing confidence
        bounding_boxes.sort(key=lambda x:float(x['confidence']), reverse=True)
        tensor = torch.zeros([len(bounding_boxes), 5])
        for idx in range(len(bounding_boxes)):
            tensor[idx, 0] = float(bounding_boxes[idx]["confidence"])
            vals = bounding_boxes[idx]["bbox"].split(" ")
            tensor[idx, 1:] = torch.FloatTensor([float(vals[x]) for x in range(4)])
        box_per_class[class_name] = tensor
    return box_per_class

"""
 Calculate the AP for each class
"""
def compute_mAP(detections_per_class, gt_per_class):
    sum_AP = 0.0
    ap_dictionary = {}
    used = {}
    for class_name in gt_classes:
        used[class_name] = set()
        curr_det = detections_per_class[class_name]
        """
            Assign detection-results to ground-truth objects
        """
        nd = curr_det.shape[0]
        tp = [0] * nd # creates an array of zeros of size nd
        fp = [0] * nd
        for idx in range(curr_det.shape[0]):
            ovmax = -1
            gt_match = -1
            # load detected object bounding-box
            bb = curr_det[idx, 1:]
            for gt_idx in range(gt_per_class[class_name].shape[0]):
                bbgt = gt_per_class[class_name][gt_idx]
                bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
                iw = bi[2] - bi[0] + 1
                ih = bi[3] - bi[1] + 1
                if iw > 0 and ih > 0:
                    # compute overlap (IoU) = area of intersection / area of union
                    ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                    + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                    ov = iw * ih / ua
                    if ov > ovmax:
                        ovmax = ov
                        gt_match = gt_idx

            # assign detection as true positive/don't care/false positive
            # set minimum overlap
            min_overlap = MINOVERLAP
            if ovmax >= min_overlap:
                if gt_match not in used[class_name]:
                    # true positive
                    tp[idx] = 1
                    used[class_name].add(gt_match)
                else:
                    # false positive (multiple detection)
                    fp[idx] = 1
            else:
                # false positive
                fp[idx] = 1
                if ovmax > 0:
                    status = "INSUFFICIENT OVERLAP"

        # compute precision/recall
        cumsum = 0
        for idx, val in enumerate(fp):
            fp[idx] += cumsum
            cumsum += val
        cumsum = 0
        for idx, val in enumerate(tp):
            tp[idx] += cumsum
            cumsum += val
        rec = tp[:]
        for idx, val in enumerate(tp):
            rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
        prec = tp[:]
        for idx, val in enumerate(tp):
            prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

        ap, mrec, mprec = voc_ap(rec[:], prec[:])
        sum_AP += ap
        text = "{0:.2f}%".format(ap*100) + " = " + class_name + " AP " #class_name + " AP = {0:.2f}%".format(ap*100)
        """
            Write to output.txt
        """
        rounded_prec = [ '%.2f' % elem for elem in prec ]
        rounded_rec = [ '%.2f' % elem for elem in rec ]
        ap_dictionary[class_name] = ap
        
    mAP = sum_AP / n_classes
    text = "mAP = {0:.2f}%".format(mAP*100)
    return mAP, ap_dictionary

if __name__ == "__main__":
    gt_file = "input/ground-truth/test1.txt"
    dr_file = "input/detection-results/test1.txt"
    gt_classes = ["foo", "bar"]

    detections_per_class = load_detections(dr_file)

    gt_per_class = load_ground_truth()
    gt_classes = list(gt_counter_per_class.keys())
    # let's sort the classes alphabetically
    gt_classes = sorted(gt_classes)
    n_classes = len(gt_classes)
    out = compute_mAP(detections_per_class, gt_per_class)
    print(out)