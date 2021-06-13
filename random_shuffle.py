# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 14:00:34 2021

@author: safayet_khan
"""


# Importing necessary libraries
import os
import random
import glob
import shutil


# Necessary constant values are being fixed
FOLDER_LIST = [
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
]
MAIN_PATH = "C:/Users/safayet_khan/Desktop/emotion_recognition"
READ_PATH = os.path.join(MAIN_PATH, "processed_data/emotion_data_flow/train")
WRITE_PATH = os.path.join(MAIN_PATH, "processed_data/emotion_data_flow/train_shuffled")


# As the images are shuffled randomly.
# Specific seed is used for reproduction of results
RANDOM_SEED = 42
random.seed(a=RANDOM_SEED)


# Creating destination folder if it's not exist already
if not os.path.exists(path=WRITE_PATH):
    os.mkdir(path=WRITE_PATH)


# Saving the shuffled images into specific directory
# with a fixed naming convention
for FOLDER_NAME in FOLDER_LIST:
    read_image_path_list = glob.glob(os.path.join(READ_PATH, FOLDER_NAME, "*.png"))
    random.shuffle(read_image_path_list)

    write_folder_path = os.path.join(WRITE_PATH, FOLDER_NAME)
    if not os.path.exists(path=write_folder_path):
        os.mkdir(path=write_folder_path)
    os.chdir(path=write_folder_path)
    COUNTER = 1
    for read_image_path in read_image_path_list:
        write_image_path = write_folder_path + "/{:05d}.png".format(COUNTER)
        shutil.copy(src=read_image_path, dst=write_image_path)
        COUNTER += 1
