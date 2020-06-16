"""File containing functions to create a .bin file needed for packing a solution accordingly."""
from pathlib import Path
import sys
import tensorflow as tf
import tqdm
import uuid

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2
from waymo_open_dataset.utils import frame_utils


def _fancy_deep_learning(frame):
    """Creates a prediction objects file."""
    o_list = []

    for camera_labels in frame.camera_labels:
        for gt_label in camera_labels.labels:
            o = metrics_pb2.Object()
            # The following 3 fields are used to uniquely identify a frame a prediction
            # is predicted at.
            o.context_name = frame.context.name
            # The frame timestamp for the prediction. See Frame::timestamp_micros in
            # dataset.proto.
            o.frame_timestamp_micros = frame.timestamp_micros
            # This is only needed for 2D detection or tracking tasks.
            # Set it to the camera name the prediction is for.
            o.camera_name = camera_labels.name

            # Populating box and score.
            box = label_pb2.Label.Box()
            box.center_x = gt_label.box.center_x
            box.center_y = gt_label.box.center_y
            box.length =   gt_label.box.length
            box.width =    gt_label.box.width
            o.object.box.CopyFrom(box)
            # This must be within [0.0, 1.0]. It is better to filter those boxes with
            # small scores to speed up metrics computation.
            o.score = 0.9
            # Use correct type.
            o.object.type = gt_label.type
            o_list.append(o)

    return o_list


def create_bin(dataset_path):
    dataset = tf.data.TFRecordDataset([str(x.absolute()) for x in Path(dataset_path).rglob('*.tfrecord')])
    print([str(x.absolute()) for x in Path(dataset_path).rglob('*.tfrecord')])
    objects = metrics_pb2.Objects()
    for data in tqdm.tqdm(dataset):
        frame = dataset_pb2.Frame()
        print(frame.context.name)
        frame.ParseFromString(bytearray(data.numpy()))
        o_list = _fancy_deep_learning(frame)
        for o in o_list:
            objects.objects.append(o)

    f = open("/tmp/tmp.bin", 'wb')
    f.write(objects.SerializeToString())
    f.close()


def main():
    dataset_path = sys.argv[1]
    create_bin(dataset_path)

if __name__ == '__main__':
  main()
