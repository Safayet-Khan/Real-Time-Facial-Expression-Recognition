# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 15:39:28 2021

@author: safayet_khan
"""


# Importing necessary libraries
import math
import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, CSVLogger
from tensorflow.keras.callbacks import LearningRateScheduler, TerminateOnNaN
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow_addons.losses import SigmoidFocalCrossEntropy

tf.config.experimental_run_functions_eagerly(True)


# Image size, batch size, and other necessary values are being fixed
RANDOM_SEED = 42
BATCH_SIZE = 256
IMAGE_SIZE = 64
CHANNEL_NUMBER = 1
IMAGE_SIZE_2D = (IMAGE_SIZE, IMAGE_SIZE)
IMAGE_SIZE_3D = (IMAGE_SIZE, IMAGE_SIZE, CHANNEL_NUMBER)
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
TRAIN_DIR = os.path.join(MAIN_DIR, "processed_data/emotion_data_flow/train")
VAL_DIR = os.path.join(MAIN_DIR, "processed_data/emotion_data_flow/val")


# Number of Images in each training folder
number_of_images = np.empty(len(CLASSES), dtype=np.uint16)

for index, folder_name in enumerate(CLASSES):
    number_of_images[index] = len(
        glob.glob(os.path.join(TRAIN_DIR, folder_name, "*.png"))
    )

print(CLASSES)
print(number_of_images)


# Defining Focal Loss
VALUE_OF_ALPHA = [0.70, 0.80, 0.75, 0.70, 0.50, 0.45, 0.70, 0.55]

focal_loss = SigmoidFocalCrossEntropy(
    from_logits=False,
    alpha=VALUE_OF_ALPHA,
    gamma=2.0,
    reduction=tf.keras.losses.Reduction.NONE,
)


# Rescaling and creating augmentation with ImageDataGenerator for
# training set and validation set
train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.80, 1.10],
    zoom_range=[1.00, 1.20],
    rescale=1 / 255.0,
    fill_mode="constant",
    cval=200,
)

validation_datagen = ImageDataGenerator(rescale=1 / 255.0)


# Data generator for the training set
train_generator = train_datagen.flow_from_directory(
    directory=TRAIN_DIR,
    target_size=IMAGE_SIZE_2D,
    color_mode="grayscale",
    classes=CLASSES,
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=RANDOM_SEED,
)
print(train_generator.class_indices)


# Data generator for the validation set
validation_generator = validation_datagen.flow_from_directory(
    directory=VAL_DIR,
    target_size=IMAGE_SIZE_2D,
    color_mode="grayscale",
    classes=CLASSES,
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=False,
)
print(validation_generator.class_indices)


# Building a VGG like model
model = Sequential()
model.add(Input(shape=IMAGE_SIZE_3D))

model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding="valid"))

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding="valid"))

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding="valid"))

model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding="valid"))

model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding="valid"))

model.add(Flatten())
model.add(Dense(units=512, activation="relu"))
model.add(Dropout(rate=0.6))
model.add(Dense(units=512, activation="relu"))
model.add(Dropout(rate=0.6))
model.add(
    Dense(
        units=len(CLASSES),
        activation="softmax",
        bias_initializer=tf.keras.initializers.Constant(value=-2.0),
    )
)

model.summary()


# Callback- Stopped after reaching a certain value
VALIDATION_ACCURACY_THRESHOLD = 0.99


class MyCallback(Callback):
    """
    Stop training after val_accuracy reached a certain number.
    """

    def on_epoch_end(self, epoch, logs=None):
        """
        Model validation will be done at the end of each epoch.
        """
        if logs.get("val_accuracy") > VALIDATION_ACCURACY_THRESHOLD:
            print("\nCOMPLETED!!!")
            self.model.stop_training = True


callbacks = MyCallback()


# ModelCheckpoint Callback
CHECKPOINT_PATH = os.path.join(MAIN_DIR, "code_file/checkpoint")
if not os.path.exists(path=CHECKPOINT_PATH):
    os.mkdir(path=CHECKPOINT_PATH)

MODEL_FILEPATH = os.path.join(CHECKPOINT_PATH, "emotion_{epoch:02d}.h5")
checkpoint_callback = ModelCheckpoint(
    filepath=MODEL_FILEPATH,
    monitor="val_accuracy",
    mode="max",
    verbose=1,
    save_weights_only=False,
    save_best_only=True,
    save_freq="epoch",
)


# LearningRateScheduler Callback
def lr_step_decay(epoch):
    """
    Reduce the learning rate by a certain percentage after a certain
    number of epochs.
    """
    drop_rate = 0.25
    epochs_drop = 5.0
    return INITIAL_LEARNING_RATE * math.pow(
        (1 - drop_rate), math.floor(epoch / epochs_drop)
    )


lr_callback = LearningRateScheduler(schedule=lr_step_decay, verbose=1)


# CSVLogger & TerminateOnNaN Callback
LOG_PATH = os.path.join(MAIN_DIR, "code_file/log")
if not os.path.exists(path=LOG_PATH):
    os.mkdir(path=LOG_PATH)

CSVLOGGER = os.path.join(LOG_PATH, "log_emotion.csv")
CSVLOGGER_callback = CSVLogger(filename=CSVLOGGER, separator=",", append=False)

Terminate_callback = TerminateOnNaN()


# Step per epoch, Validation steps, and number of Epochs to be trained
STEPS_PER_EPOCH = math.ceil(train_generator.samples / BATCH_SIZE)
VALIDATION_STEPS = math.ceil(validation_generator.samples / BATCH_SIZE)
NUMBER_OF_EPOCHS = 50


# Compiling the Model
INITIAL_LEARNING_RATE = 0.001

model.compile(
    optimizer=Adam(learning_rate=INITIAL_LEARNING_RATE),
    loss=focal_loss,
    metrics=["accuracy"],
)


# Training the Model
model.fit(
    train_generator,
    shuffle=True,
    epochs=NUMBER_OF_EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    verbose=1,
    validation_data=validation_generator,
    validation_steps=VALIDATION_STEPS,
    callbacks=[
        callbacks,
        checkpoint_callback,
        lr_callback,
        CSVLOGGER_callback,
        Terminate_callback,
    ],
)
