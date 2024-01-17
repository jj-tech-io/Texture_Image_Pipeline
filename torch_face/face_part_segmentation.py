import torch
import torchvision.transforms as transforms
# import .model 
from .model import BiSeNet
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pathlib
import segmentation
import morph


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
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
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
        skin_mask = self.__get_part_mask(image, ['mouth','l_lip', 'u_lip', 'hair', 'l_eye', 'r_eye'])
        skin2 = self.__get_part_mask(image, ['skin'])

        return  skin_mask, skin2

    def get_eyes(self, image):
        return self.__get_part_mask(image, ['l_eye', 'r_eye'])

    def get_ears(self, image):
        return self.__get_part_mask(image, ['l_ear', 'r_ear'])

    def get_lips(self, image):
        return self.__get_part_mask(image, ['u_lip', 'l_lip'])
    
if __name__ == '__main__':
    image_path = r"C:\Users\joeli\Dropbox\Code\Python Projects\Texture_Image_Pipeline\fitzpatrick\ft_4_ft_4_053609.png"
    image_path = r"C:\Desktop\joel.jpg"
    # Example Usage
    image = cv2.imread(image_path)
    fps = FacePartSegmentation()
    skin_mask, skin2 = fps.get_skin(image)
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(image)
    ax[1].imshow(skin_mask)
    ax[2].imshow(skin2)
    plt.show()
    