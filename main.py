import sys
from pathlib import Path
import os
from pathlib import Path
import importlib

import CONFIG
import importlib
importlib.reload(CONFIG)
from path_utils import PathUtils
import dense_lm
from dense_lm.morph import *
import onnx_inference
from onnx_inference import onnx_ae
from onnx_inference import modify_latent
import torch_face
from torch_face import face_part_segmentation as fps
import segmentation
from segmentation import extract_masks
import GUI
importlib.reload(GUI)

if __name__ == '__main__':
    working_dir = os.getcwd()

    example_texture_path = CONFIG.example_texture_path
    target_texture_path = CONFIG.target_texture_path
    app = GUI.SkinParameterAdjustmentApp(target_texture_path, example_texture_path)
    app.run()
