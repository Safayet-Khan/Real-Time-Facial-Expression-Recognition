# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 13:06:42 2021

@author: safayet_khan
"""


# Importing necessary libraries
import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow_addons.losses import SigmoidFocalCrossEntropy
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score


# Necessary constant values are being fixed
NUMBER_OF_EPOCH = 48
BATCH_SIZE = 256
IMAGE_SIZE = 64
IMAGE_SIZE_2D = (IMAGE_SIZE, IMAGE_SIZE)
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
MAIN_DIR = "C:/Users/safayet_khan/Desktop/emotion_recognition"
CHECKPOINT_PATH = os.path.join(MAIN_DIR, "code_file/checkpoint")
TEST_MODEL_PATH = os.path.join(
    CHECKPOINT_PATH, "emotion_{:02d}.h5".format(NUMBER_OF_EPOCH)
)


# Loading the trained model
if os.path.exists(TEST_MODEL_PATH):
    test_model = load_model(TEST_MODEL_PATH)
    test_model.summary()

# Data generator for the test set
# (Using validation set as test set - not recommended)
test_datagen = ImageDataGenerator(rescale=1 / 255.0)
test_generator = test_datagen.flow_from_directory(
    directory=os.path.join(MAIN_DIR, "processed_data/emotion_data_flow/val"),
    target_size=IMAGE_SIZE_2D,
    color_mode="grayscale",
    classes=None,
    class_mode=None,
    batch_size=BATCH_SIZE,
    shuffle=False,
)
print(test_generator.class_indices)


# Prediction using the trained model
TEST_STEPS = math.ceil(test_generator.samples / BATCH_SIZE)
prediction_probabilities = test_model.predict(
    test_generator, steps=TEST_STEPS, verbose=1
)
y_pred = np.argmax(prediction_probabilities, axis=1)


# Ground truth of the test image
y_true = np.array(
    [0] * 623
    + [1] * 44
    + [2] * 50
    + [3] * 164
    + [4] * 1871
    + [5] * 2582
    + [6] * 875
    + [7] * 893
)


# Calculating confusion matrix
cm_number = confusion_matrix(y_true, y_pred)
cm_ratio = cm_number / cm_number.astype(np.float).sum(axis=1, keepdims=True)
df_cm = pd.DataFrame(cm_ratio, index=CLASSES, columns=CLASSES)


# Visualizing the confusion matrix
plt.figure(figsize=(12, 12))
heatmap = sns.heatmap(df_cm, annot=True, cmap="Blues", fmt=".2g")
heatmap.set_title(
    "Number of Epoch {:02d}, Accuracy {:.3f}, F1(WEIGHTED) {:.3f}, F1(MACRO) {:.3f}".format(
        NUMBER_OF_EPOCH,
        accuracy_score(y_true, y_pred),
        f1_score(y_true, y_pred, average="weighted"),
        f1_score(y_true, y_pred, average="macro"),
    )
)
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.show()
