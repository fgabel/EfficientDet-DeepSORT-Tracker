import argparse
import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import time
from waymo_open_dataset import dataset_pb2

def tfrecord_to_mp4(tfrecord_path, out_path):
    """
        Converts single tfrecord file to video files.
        One video file per camera is placed in out_path.
    """
    ds = tf.data.TFRecordDataset(args.path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videos = [None] * 5

    for data in ds:
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        for image in frame.images:
            img = tf.io.decode_jpeg(image.image).numpy()
            if videos[image.name-1] is None:
                video_path = os.path.join(out_path, '{}_{}.mp4'.format(frame.context.name, image.name))
                videos[image.name-1] = cv2.VideoWriter(video_path, fourcc, 20.0, (img.shape[1],img.shape[0]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            videos[image.name-1].write(img)

    for video in videos:
        video.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert tfrecord to video.')
    parser.add_argument('path', type=str, nargs='+',
                        help='path of tfrecord(s) to convert')
    parser.add_argument('out_path', type=str, nargs='+',
                        help='path of place output file(s)')
    args = parser.parse_args()
    tfrecord_to_mp4(args.path, args.out_path[0])
