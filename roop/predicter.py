import numpy
import opennsfw2
import os
from PIL import Image

from roop.typing import Frame

MAX_PROBABILITY = 0.85


def predict_frame(target_frame: Frame) -> bool:
    image = Image.fromarray(target_frame)
    image = opennsfw2.preprocess_image(image, opennsfw2.Preprocessing.YAHOO)
    model = opennsfw2.make_open_nsfw_model()
    views = numpy.expand_dims(image, axis=0)
    _, probability = model.predict(views)[0]
    return probability > MAX_PROBABILITY


def predict_image(target_path: str) -> bool:
    return opennsfw2.predict_image(target_path) > MAX_PROBABILITY

def predict_images(target_path: str) -> bool:
    image_files = []

    for root, dirs, files in os.walk(target_path):
        for file in files:
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                image_files.append(os.path.join(root, file))
    probabilities = opennsfw2.predict_images(image_files)
    return any(probability > MAX_PROBABILITY for probability in probabilities)

def predict_video(target_path: str) -> bool:
    _, probabilities = opennsfw2.predict_video_frames(video_path=target_path, frame_interval=100)
    return any(probability > MAX_PROBABILITY for probability in probabilities)
