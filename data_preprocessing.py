# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 11:40:50 2021

@author: safayet_khan
"""


# Importing necessary libraries
import os
import pandas as pd
from PIL import Image
import numpy as np


# Necessary constant values are being fixed
IMAGE_SIZE = 48
ANNOTATION = [
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
    "unknown",
    "NF",
]
MAIN_DIR = "C:/Users/safayet_khan/Desktop/emotion_recognition"


# Loading of fer and fer-plus data in pandas dataframe
df_fer_plus = pd.read_csv(os.path.join(MAIN_DIR, "main_data_csv/fer2013new.csv"))
df_fer = pd.read_csv(os.path.join(MAIN_DIR, "main_data_csv/fer2013.csv"))


def string_to_image(image_string):
    """
    Take image string as an input and
    return an image object as an output.
    """
    image_array = image_string.split(" ")
    image_array = np.asarray(image_array, dtype=np.uint8).reshape(
        IMAGE_SIZE, IMAGE_SIZE
    )
    return Image.fromarray(image_array)


# Creating destination folder if it's not exist already
WRITE_DATA_PATH = os.path.join(MAIN_DIR, "processed_data/emotion_data")
if not os.path.exists(path=WRITE_DATA_PATH):
    os.mkdir(path=WRITE_DATA_PATH)

os.chdir(path=WRITE_DATA_PATH)
for folder_name in ANNOTATION:
    if not os.path.exists(path=folder_name):
        os.mkdir(path=folder_name)


# Turning the pixels data into images and
# saving the image into specific directory
for index_value in range(0, np.shape(df_fer)[0]):
    if isinstance(df_fer_plus["Image name"][index_value], float):
        continue
    image_obj = string_to_image(df_fer["pixels"][index_value])
    annotation_list = list(df_fer_plus.loc[index_value, ANNOTATION])
    number_index = annotation_list.index(max(annotation_list))
    image_obj.save(ANNOTATION[number_index] + "/" + "{}.png".format(index_value))
