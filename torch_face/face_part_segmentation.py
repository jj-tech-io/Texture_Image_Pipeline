import sys
from pathlib import Path
current_script_dir = Path(__file__).resolve()
# Get the parent directory (one level up)
parent_dir = current_script_dir.parent
# Get the root directory (two levels up or more if needed)
root_dir = current_script_dir.parents[1]  # Use parents[2], parents[3], etc., for higher levels
# Add both directories to sys.path
sys.path.append(str(parent_dir))
sys.path.append(str(root_dir))
import torch
import torchvision.transforms as transforms
# import .model 
from model import BiSeNet
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pathlib
from dense_lm import segmentation, morph
from segmentation import *
#remove background
import rembg
from rembg import remove

class FacePartSegmentation:
    n_classes = 19
    part_labels = {
        'skin': [1], 'l_brow': [2], 'r_brow': [3], 'l_eye': [4], 'r_eye': [5], 
        'eye_g': [6], 'l_ear': [7], 'r_ear': [8], 'ear_r': [9], 'nose': [10], 
        'mouth': [11], 'u_lip': [12], 'l_lip': [13], 'neck': [14], 'neck_l': [15], 
        'cloth': [16], 'hair': [17], 'hat': [18]
    }
    
    # Static device, as there is only one model and one device setting for all instances
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = r"models\79999_iter.pth"  # Assuming model_path is constant and known
    child_dir = pathlib.Path(__file__).parent.absolute()
    cwd = str(child_dir)
    model_path = os.path.join(cwd, model_path)
    def __init__(self):
        self.net = BiSeNet(self.n_classes)
        self.net.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.net.eval()
        self.net.to(self.device)

    def __model_inference(self, image):
        to_tensor = transforms.Compose([
            transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ]
            )
        tensor_image = to_tensor(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.net(tensor_image)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        return cv2.resize(parsing, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LANCZOS4)

    def __get_part_mask(self, image, parts):
        parsing = self.__model_inference(image)
        part_mask = np.zeros_like(parsing)
        for part in parts:
            part_mask += np.isin(parsing, self.part_labels[part]).astype(np.uint8)
        return part_mask

    def get_skin(self, image):
        image = remove(image)   
        bg = np.zeros_like(image)
        bg = np.where(image == 0, 255, bg)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        skin = self.__get_part_mask(image,['skin','l_ear', 'r_ear','l_brow', 'r_brow', 'nose', 'mouth', 'l_lip', 'u_lip', 'l_eye', 'r_eye'])
        skin = np.asarray(skin, dtype=np.uint8)
        image_skin = image * skin[:, :, np.newaxis]
        mask = cv2.GaussianBlur(skin, (5, 5), 0)
        mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
        mask *= 15.0
        return  mask, image_skin

    def get_eyes(self, image):
        return self.__get_part_mask(image, ['l_eye', 'r_eye'])

    def get_ears(self, image):
        return self.__get_part_mask(image, ['l_ear', 'r_ear'])

    def get_lips(self, image):
        return self.__get_part_mask(image, ['u_lip', 'l_lip'])
    def get_hair(self, image):
        return self.__get_part_mask(image, ['hair'])



    
if __name__ == '__main__':
    image_path = r"C:\Users\joeli\Dropbox\Code\Python Projects\Texture_Image_Pipeline\fitzpatrick\ft_4_ft_4_053609.png"
    image_path = r"C:\Desktop\joel.jpg"
    # Example Usage
    image = cv2.imread(image_path)
    fps = FacePartSegmentation()
    mask, image_skin = fps.get_skin(image)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(mask)
    ax[1].imshow(image_skin)

    plt.show()
        