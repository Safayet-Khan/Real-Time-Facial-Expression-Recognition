# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 11:15:43 2021

@author: safayet_khan
"""


# Importing necessary libraries
import glob
import os
import numpy as np
import matplotlib.pyplot as plt


# Necessary constant values are being fixed
MAIN_DIR = "C:/Users/safayet_khan/Desktop/emotion_recognition/processed_data"
CLASSES = [
    "anger",
    "contempt",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    "surprise",
]
DATA_DIR = os.path.join(MAIN_DIR, "emotion_data")
PROCESSED_DATA_DIR = os.path.join(MAIN_DIR, "emotion_data_flow/train")


# Number of Images in each training folder
number_of_images = np.empty(len(CLASSES), dtype=np.uint16)
number_of_processed_images = np.empty(len(CLASSES), dtype=np.uint16)

for index, folder_name in enumerate(CLASSES):
    number_of_images[index] = len(
        glob.glob(os.path.join(DATA_DIR, folder_name, "*.png"))
    )
    number_of_processed_images[index] = len(
        glob.glob(os.path.join(PROCESSED_DATA_DIR, folder_name, "*.png"))
    )


# Necessary values for plotting graph
GRAPH_WIDTH = 0.40
NUMBER_OF_CLASSES = len(CLASSES)
X_AXIS = np.arange(NUMBER_OF_CLASSES)

plt.bar(
    X_AXIS,
    number_of_images,
    color="b",
    width=GRAPH_WIDTH,
    edgecolor="black",
    label="Images per class of raw data",
)
plt.bar(
    X_AXIS + GRAPH_WIDTH,
    number_of_processed_images,
    color="g",
    width=GRAPH_WIDTH,
    edgecolor="black",
    label="Training images per class after oversampling",
)

plt.xlabel("Name of Classes")
plt.ylabel("Number of Images")
plt.title("Analysis of Training Data")
plt.grid(linestyle=(0, (1, 10)))  # (0, (1, 10)) -> 'loosely dotted'
plt.xticks(X_AXIS + GRAPH_WIDTH / 2, CLASSES)
plt.legend()
plt.show()
