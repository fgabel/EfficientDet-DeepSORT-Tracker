"""
    Utility script to pack the submission in a directory 
    to the required format of the Waymo submission tool.

    Usage: python pack_submission.py <dataset_path> <out_name>

    Expected Result: Generation of <out_name>.tar.gz file
"""
import os
from pathlib import Path
import sys

from create_bin import create_bin

if __name__ == "__main__":
    print("Note: Make sure to not use trailing slash in folder name!")
    # create_bin(sys.argv[1])
    path = sys.argv[1]
    Path(path).mkdir(parents=True, exist_ok=True)
    os.system('../../waymo-od/bazel-bin/waymo_open_dataset/metrics/tools/create_submission --input_filenames="{}" --output_filename="{}/" --submission_filename="submission.txtpb"'.format("/tmp/tmp.bin", path))
    os.system("tar cvf {}.tar {}/".format(path, path))
    os.system("gzip -f {}.tar".format(path))
