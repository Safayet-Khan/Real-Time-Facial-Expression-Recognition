# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 15:39:28 2021

@author: safayet_khan
"""


# Importing necessary libraries
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow_addons.losses import SigmoidFocalCrossEntropy

tf.config.experimental_run_functions_eagerly(True)


# Necessary constant values are being fixed
IMAGE_SIZE = 64
IMAGE_SIZE_2D = (IMAGE_SIZE, IMAGE_SIZE)
TEST_IMAGE_RESIZE = (900, 700)
THRESHOLD = 20
BOUNDING_BOX_COLOR = (0, 0, 255)
BOUNDING_BOX_THICKNESS = 7
TEXT_COLOR = (255, 255, 0)
TEXT_THICKNESS = 2
EMOTIONS = [
    "angry",
    "contempt",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]
MAIN_DIR = "C:/Users/safayet_khan/Desktop/emotion_recognition"


# Loading the trained model
trained_model = load_model(os.path.join(MAIN_DIR, "code_file/checkpoint/emotion_48.h5"))
trained_model.summary()


# Loading the default model haarcascade_frontalface xml file of open-cv
face_haarcascade = cv2.CascadeClassifier("haarcascade_frontalface.xml")
video_capture = cv2.VideoCapture(cv2.CAP_DSHOW)


# Opening the Webcam and using the trained model for prediction
while True:
    # captures frame and returns boolean value and captured image
    (ret, test_image) = video_capture.read()
    if not ret:
        continue
    gray_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

    faces_detected = face_haarcascade.detectMultiScale(
        gray_image, scaleFactor=1.3, minNeighbors=5
    )

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(
            test_image,
            (x - THRESHOLD, y - THRESHOLD),
            (x + w + THRESHOLD, y + h + THRESHOLD),
            color=BOUNDING_BOX_COLOR,
            thickness=BOUNDING_BOX_THICKNESS,
        )
        # cropping region of interest i.e. face area from  image
        roi_gray = gray_image[
            (y - THRESHOLD) : (y + w + THRESHOLD), (x - THRESHOLD) : (x + h + THRESHOLD)
        ]
        roi_gray = cv2.resize(roi_gray, IMAGE_SIZE_2D)
        image_pixels = image.img_to_array(roi_gray)
        image_pixels = np.expand_dims(image_pixels, axis=0)
        image_pixels /= 255.0

        predictions = trained_model.predict(image_pixels)
        max_index = np.argmax(predictions)
        predicted_emotion = EMOTIONS[max_index]

        cv2.putText(
            img=test_image,
            text="{} {}".format(
                predicted_emotion, float("{0:.2f}".format(np.max(predictions)))
            ),
            org=(int(x), int(y)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=TEXT_COLOR,
            thickness=TEXT_THICKNESS,
        )

    resized_image = cv2.resize(test_image, TEST_IMAGE_RESIZE)
    cv2.imshow("Facial emotion analysis ", resized_image)

    # wait until 'q' key is pressed
    if cv2.waitKey(10) == ord("q"):
        break

video_capture.release()
