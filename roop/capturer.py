from typing import Any
import cv2
import os


def get_video_frame(video_path: str, frame_number: int = 0) -> Any:
    capture = cv2.VideoCapture(video_path)
    frame_total = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    capture.set(cv2.CAP_PROP_POS_FRAMES, min(frame_total, frame_number - 1))
    has_frame, frame = capture.read()
    capture.release()
    if has_frame:
        return frame
    return None


def get_video_frame_total(video_path: str) -> int:
    capture = cv2.VideoCapture(video_path)
    video_frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return video_frame_total

def get_image_dir_total(image_dir_path: str) -> (int, [str]):
    image_extensions = ['.png', '.jpg', '.jpeg', '.webp']
    image_count = 0
    image_paths = []
    
    for filename in os.listdir(image_dir_path):
        file_extension = os.path.splitext(filename)[1].lower()
        if file_extension in image_extensions:
            image_count += 1
            image_paths.append(os.path.join(image_dir_path,filename))
            
    return image_count, image_paths
