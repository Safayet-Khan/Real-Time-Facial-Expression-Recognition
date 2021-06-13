# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 14:57:05 2021

@author: safayet_khan
"""


# Importing necessary libraries
import os
import glob
import numpy as np
from PIL import Image, ImageEnhance
from skimage import io


# Necessary constant values are being fixed
ENHANCE_FACTOR = 1.5
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
WRITE_PATH = os.path.join(MAIN_PATH, "processed_data/emotion_data_flow/train_contrast")


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
        image_contrast = ImageEnhance.Contrast(pillow_image)
        write_image = image_contrast.enhance(ENHANCE_FACTOR)

        write_image_path = write_path + "/contrast" + image_path.split("\\")[-1]
        io.imsave(write_image_path, np.array(write_image).astype(np.uint8))
