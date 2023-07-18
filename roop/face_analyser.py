import threading
from typing import Any
import numpy as np
import insightface
from insightface.app.common import Face
import json
import roop.globals
from roop.typing import Frame
import os

FACE_ALIGNMENTS = None

FACE_ANALYSER = None
THREAD_LOCK = threading.Lock()


def get_face_analyser() -> Any:
    global FACE_ANALYSER

    with THREAD_LOCK:
        if FACE_ANALYSER is None:
            FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=roop.globals.execution_providers)
            FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))
    return FACE_ANALYSER
    
def get_face_alignments() -> Any:
    global FACE_ALIGNMENTS
    if FACE_ALIGNMENTS is None:
        alignmentPath = os.path.join(roop.globals.target_path,'facedata.json')
        with open(alignmentPath) as f:
            data = json.load(f)
        FACE_ALIGNMENTS = data
    return FACE_ALIGNMENTS
            

def get_one_face(frame: Frame) -> Any:
    face = get_face_analyser().get(frame)
    try:
        return min(face, key=lambda x: x.bbox[0])
    except ValueError:
        return None
#get one face for target with alignment file
def get_one_face_from_alignment(target_path: str) -> Any:
    filename = os.path.basename(target_path)
    facedata = get_face_alignments().get(filename, None)
    if facedata:
        boundingX1 = facedata['x1']
        boundingY1 = facedata['y1']
        boundingX2 = facedata['x2']
        boundingY2 = facedata['y2']
        kps = np.array(facedata['kps'], dtype=np.float32)
        bbox = np.array([boundingX1, boundingY1,boundingX2,boundingY2], dtype=np.float32)
        face = Face(bbox=bbox,kps=kps,det_score=0.8)
        return face
    return None
def get_many_faces(frame: Frame) -> Any:
    try:
        return get_face_analyser().get(frame)
    except IndexError:
        return None
