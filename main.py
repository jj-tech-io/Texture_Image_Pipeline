import sys
from pathlib import Path
import os
from pathlib import Path
import importlib


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
    ### --- age example texture --- ###
    example_texture_path = r"C:\Users\joeli\Dropbox\Code\Python Projects\Texture_Image_Pipeline\fitzpatrick\m32_4k.png"
    example_texture_path = "txyz2mh_32_pc.jpg"
    # example_texture_path = "txyz2mhman_32_pc.jpg"

    ### --- texture to be modified --- ###
    target_texture_path = r"C:\Users\joeli\Dropbox\Code\Python Projects\Texture_Image_Pipeline\fitzpatrick\ft_4_m53_4k.png"
    target_texture_path = "FaceColor_MAIN.PNG"
    target_texture_path = r"D:\Desktop\man mh\FaceColor_MAIN.PNG"
    # target_texture_path = r"FaceColor_man.PNG"
    app = GUI.SkinParameterAdjustmentApp(target_texture_path, example_texture_path)
    app.run()
