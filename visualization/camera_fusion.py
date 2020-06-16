"""
    Utility script that extracts all images from a tfrecord / block of tfrecords
    and plots them as a single, fused image.
"""
from waymo_open_dataset import dataset_pb2
import tqdm
import tensorflow as tf
import sys
from pathlib import Path
import numpy as np
import imageio
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if __name__ == "__main__":
    if not len(sys.argv) == 3:
        print("Usage: python extract_frames.py <input_path.tfrecord> <out_folder>")
    tfrecord_path = [str(x.absolute()) for x in Path(sys.argv[1]).rglob('*.tfrecord')] if \
        not sys.argv[1].endswith(".tfrecord") else sys.argv[1]
    out_folder = sys.argv[2]

    dataset = tf.data.TFRecordDataset(tfrecord_path)

    idx = 0
    for data in tqdm.tqdm(dataset):
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        camera_images = frame.images
        w = 1920 - 225
        delta = 1280 - 886
        fused_arr = np.zeros([886, w * 5, 3], dtype=np.uint8)
        mapx = {1: 2, 2: 1, 3: 3, 4: 0, 5: 4}
        for camera_image in camera_images:
            if camera_image.name >= 1 and camera_image.name <= 3:
                arr = tf.image.decode_jpeg(camera_image.image)
                x = mapx[camera_image.name]
                fused_arr[:, x*w:(x+1)*w] = arr[delta:, :w]
            else:
                x = mapx[camera_image.name]
                arr = tf.image.decode_jpeg(camera_image.image)
                fused_arr[:, x*w:(x+1)*w] = arr[:, :w]
                # fused_arr[:, (x+1)*w-206:(x+1)*w] = 0
        imageio.imwrite(os.path.join(
            out_folder, "{:06d}.jpg".format(idx)), fused_arr)
        idx += 1
