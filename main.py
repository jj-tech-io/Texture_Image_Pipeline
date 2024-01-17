import os
from pathlib import Path
from tkinter import *
from tkinter.ttk import *
import importlib
import sys
sys.path.append('morph')
sys.path.append('transform_objects')
sys.path.append('segmentation')
import morph
from morph import *
import transform_objects
from transform_objects import *
sys.path.append('onnx_inference')
import onnx_inference
from onnx_inference import autoencoder
from onnx_inference import modify_latent
import torch_face
from torch_face import face_part_segmentation as fps
import segmentation
from segmentation import *
importlib.reload(segmentation)
importlib.reload(morph)
importlib.reload(transform_objects)
WIDTH = 512
HEIGHT = 512

# morph the source image to the target image
def morph_images(example_image_path, target_image_path):
    target_image = cv2.imread(target_image_path.as_posix())
    example_image = cv2.imread(example_image_path.as_posix())
    if example_image is None:
        print(f"Error: could not read source image {example_image_path}")
        sys.exit()
    if target_image is None:
        print(f"Error: could not read target image {target_image_path}")
        sys.exit()
    # Resize the images
    target_image = cv2.resize(target_image, (WIDTH, HEIGHT))
    example_image = cv2.resize(example_image, (WIDTH, HEIGHT))
    landmarks1 = get_landmarks(example_image)
    landmarks2 = get_landmarks(target_image)
    warped_example_image, delaunay, transformation_matrices = warp_image(example_image, target_image,
        landmarks1, landmarks2)
    return warped_example_image, target_image, example_image

def extract_masks(image):
    Cm, Ch, Bm, Bh, T = modify_latent.get_masks(image)
    landmarks = get_landmarks(image)
    combined_mask, face, av_skin_color = create_combined_mask(image)
    skin = threshold_face_skin_area(image,av_skin_color,mask=combined_mask)
    return Cm, Bh, skin, face

# apply masks and transformations to target's latent space
def apply_transforms(target_image, mel_aged, oxy_aged, skin, face):
    app = SkinParameterAdjustmentApp(image=target_image, mel_aged=mel_aged, oxy_aged=oxy_aged,skin=skin,face=face, save_name="recovered")
    app.run()

if __name__ == '__main__':
    working_dir = os.getcwd()
    #age example texture
    example_texture_path = r"fitzpatrick\m32_4k.png"
    example_texture_path = r"C:\Users\joeli\Dropbox\Code\Python Projects\Texture_Image_Pipeline\fitzpatrick\m32_4k.png"
    #texture to be modified
    target_texture_path = r"fitzpatrick\ft_4_m53_4k.png"
    # C:\Users\joeli\Dropbox\Code\Python Projects\Texture_Image_Pipeline\fitzpatrick\m32_4k.png
    target_texture_path = r"C:\Users\joeli\Dropbox\Code\Python Projects\Texture_Image_Pipeline\fitzpatrick\ft_4_m53_4k.png"
    warped_example_image, target_image, example_image = morph_images(Path(working_dir, example_texture_path), Path(working_dir, target_texture_path))
    Cm, Bh, skin, face = extract_masks(warped_example_image)
    segmenter = fps.FacePartSegmentation()
    apply_transforms(target_image, Cm, Bh, skin, face)
