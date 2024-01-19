import sys
from pathlib import Path
from path_utils import PathUtils
import os
from pathlib import Path
from tkinter import *
from tkinter.ttk import *
import importlib
import dense_lm
#reload(media_pipe)
importlib.reload(dense_lm)

from dense_lm.morph import *
import transform_objects
importlib.reload(transform_objects)
from transform_objects import *
sys.path.append('onnx_inference')
import onnx_inference
from onnx_inference import onnx_ae
from onnx_inference import modify_latent
import torch_face
from torch_face import face_part_segmentation as fps
import segmentation
from segmentation import extract_masks

WIDTH = 512
HEIGHT = 512


if __name__ == '__main__':
    working_dir = os.getcwd()
    #age example texture
    example_texture_path = r"fitzpatrick\m32_4k.png"
    example_texture_path = r"C:\Users\joeli\Dropbox\Code\Python Projects\Texture_Image_Pipeline\fitzpatrick\m32_4k.png"
    #texture to be modified
    target_texture_path = r"fitzpatrick\ft_4_m53_4k.png"
    # C:\Users\joeli\Dropbox\Code\Python Projects\Texture_Image_Pipeline\fitzpatrick\m32_4k.png
    target_texture_path = r"C:\Users\joeli\Dropbox\Code\Python Projects\Texture_Image_Pipeline\fitzpatrick\ft_4_m53_4k.png"
    parent_dir = Path(__file__).resolve().parent
    warped_example_image, target_image, example_image = morph.morph_images(example_texture_path, target_texture_path)
    Cm, Bh, skin, face = extract_masks(warped_example_image)
    segmenter = fps.FacePartSegmentation()
    # def __init__(self, target_image, Cm, Bh, skin, target_texture_path, example_texture_path):
    app = SkinParameterAdjustmentApp(target_image, Cm, Bh, skin, target_texture_path, example_texture_path)
    app.run()
