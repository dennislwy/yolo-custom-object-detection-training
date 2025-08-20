"""
Script to split a dataset into training and validation folders.

This script randomly splits images (and their corresponding annotation files)
from a dataset into train and validation sets, and copies them into
appropriately named folders.

Args:
    --datapath (str): Path to the data folder containing 'images' and 'labels' subfolders.
    --train_pct (float, optional): Ratio of images to go to the train folder (default: 0.8).

Returns:
    None

Raises:
    SystemExit: If the provided datapath does not exist or train_pct is out of bounds.
"""

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

# Define and parse user input arguments
parser = argparse.ArgumentParser(
    description="Split dataset into train and validation folders."
)
parser.add_argument(
    "--datapath",
    help="Path to data folder containing image and annotation files",
    required=True,
)
parser.add_argument(
    "--train_pct",
    help='Ratio of images to go to train folder; \
                    the rest go to validation folder (example: ".8")',
    default=0.8,
)

args = parser.parse_args()

data_path = args.datapath
train_percent = float(args.train_pct)

# Check for valid entries
if not os.path.isdir(data_path):
    print(
        "Directory specified by --datapath not found. Verify the path is correct (and uses double back slashes if on Windows) and try again."
    )
    sys.exit(0)
if train_percent < 0.01 or train_percent > 0.99:
    print("Invalid entry for train_pct. Please enter a number between .01 and .99.")
    sys.exit(0)
val_percent = 1 - train_percent

# Define path to input dataset
input_image_path = os.path.join(data_path, "images")
input_label_path = os.path.join(data_path, "labels")

# Define paths to image and annotation folders for train and validation sets
cwd = os.getcwd()
train_img_path = os.path.join(cwd, "data/train/images")
train_txt_path = os.path.join(cwd, "data/train/labels")
val_img_path = os.path.join(cwd, "data/validation/images")
val_txt_path = os.path.join(cwd, "data/validation/labels")

# Create folders if they don't already exist
for dir_path in [train_img_path, train_txt_path, val_img_path, val_txt_path]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"Created folder at {dir_path}.")

# Get list of all image and annotation files (recursively)
img_file_list = [path for path in Path(input_image_path).rglob("*")]
txt_file_list = [path for path in Path(input_label_path).rglob("*")]

print(f"Number of image files: {len(img_file_list)}")
print(f"Number of annotation files: {len(txt_file_list)}")

# Determine number of files to move to each folder
file_num = len(img_file_list)
train_num = int(file_num * train_percent)
val_num = file_num - train_num
print("Images moving to train: %d" % train_num)
print("Images moving to validation: %d" % val_num)

# Select files randomly and copy them to train or val folders
for i, set_num in enumerate([train_num, val_num]):
    # i == 0: train, i == 1: validation
    for ii in range(set_num):
        # Randomly select an image file
        img_path = random.choice(img_file_list)
        img_fn = img_path.name
        base_fn = img_path.stem
        txt_fn = base_fn + ".txt"
        txt_path = os.path.join(input_label_path, txt_fn)

        # Set destination directories based on split
        if i == 0:  # Copy to train folders
            new_img_path, new_txt_path = train_img_path, train_txt_path
        elif i == 1:  # Copy to validation folders
            new_img_path, new_txt_path = val_img_path, val_txt_path

        # Copy image file to destination
        shutil.copy(img_path, os.path.join(new_img_path, img_fn))
        # If annotation exists, copy it as well (skip if background image)
        if os.path.exists(
            txt_path
        ):  # If txt path does not exist, this is a background image, so skip txt file
            shutil.copy(txt_path, os.path.join(new_txt_path, txt_fn))

        # Remove the image from the list to avoid duplicate assignment
        img_file_list.remove(img_path)
