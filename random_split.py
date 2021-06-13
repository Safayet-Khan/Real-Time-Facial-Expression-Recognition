# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 18:22:13 2021

@author: safayet_khan
"""


# Importing necessary libraries
import os
import splitfolders


# Necessary constant values are being fixed
RANDOM_SEED = 42
RATIO = (0.80, 0.20)
MAIN_DIR = "C:/Users/safayet_khan/Desktop/emotion_recognition"
SRC_DIR = os.path.join(MAIN_DIR, "processed_data/emotion_data")
DST_DIR = os.path.join(MAIN_DIR, "processed_data/emotion_data_flow")


# Creating destination folder if it's not exist already
if not os.path.exists(path=DST_DIR):
    os.mkdir(path=DST_DIR)


# Random shuffling train set and validation set with the fixed ratio
splitfolders.ratio(
    SRC_DIR, output=DST_DIR, seed=RANDOM_SEED, ratio=RATIO, group_prefix=None
)
