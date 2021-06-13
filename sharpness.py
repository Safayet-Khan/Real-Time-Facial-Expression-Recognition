# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 16:36:25 2021

@author: safayet_khan
"""


# Importing necessary libraries
import os
import glob
import numpy as np
from PIL import Image, ImageEnhance
from skimage import io


# Necessary constant values are being fixed
SHARPNESS_FACTOR = 1.75
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
WRITE_PATH = os.path.join(MAIN_PATH, "processed_data/emotion_data_flow/train_sharpness")


# Creating destination folder if it's not exist already
if not os.path.exists(path=WRITE_PATH):
    os.mkdir(path=WRITE_PATH)


# Saving the processed image into specific directory
for FOLDER_NAME in FOLDER_LIST:
    read_image_path_list = glob.glob(os.path.join(READ_PATH, FOLDER_NAME, "*.png"))

    write_path = os.path.join(WRITE_PATH, FOLDER_NAME)
    if not os.path.exists(path=write_path):
        os.mkdir(path=write_path)
    os.chdir(path=write_path)

    for image_path in read_image_path_list:
        pillow_image = Image.open(image_path)
        image_sharpness = ImageEnhance.Sharpness(pillow_image)
        write_image = image_sharpness.enhance(SHARPNESS_FACTOR)

        write_image_path = write_path + "/sharpness" + image_path.split("\\")[-1]
        io.imsave(write_image_path, np.array(write_image).astype(np.uint8))
