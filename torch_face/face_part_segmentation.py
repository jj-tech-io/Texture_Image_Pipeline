import torch
import torchvision.transforms as transforms
# from PIL import Image
from . import model
from .model import BiSeNet
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pathlib

class FacePartSegmentation:
    def __init__(self,image_path, width=4096, height=4096):
        model_path=r"C:\Users\joeli\Dropbox\Code\Python Projects\Texture_Image_Pipeline\torch_face\79999_iter.pth"
        self.width = width
        self.height = height
        self.image_path = image_path
        self.image = None
        self.skin_mask = None
        self.skin = None
        self.skin_tile = None
        self.hair = None
        self.hair_mask = None
        self.parsing = None

        try :
            self.image = self.load_image()
        except Exception as e:
            print(f"Error loading image {image_path} with width {width} and height {height}, error {e}")
        try: 
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        except Exception as e:
            print(f"Error setting cuda:0 device, error {e}")
        self.n_classes = 19
        self.net = BiSeNet(self.n_classes)
        self.net.load_state_dict(torch.load(model_path))
        self.net.eval()
        self.net.to(self.device)
        self.part_labels = {
            'skin': [1], 'l_brow': [2], 'r_brow': [3], 'l_eye': [4], 'r_eye': [5], 
            'eye_g': [6], 'l_ear': [7], 'r_ear': [8], 'ear_r': [9], 'nose': [10], 
            'mouth': [11], 'u_lip': [12], 'l_lip': [13], 'neck': [14], 'neck_l': [15], 
            'cloth': [16], 'hair': [17], 'hat': [18]
        }
        self.combined_part_labels = {
            tuple(['eyes']): [4, 5],
            tuple(['ears']): [7, 8],
            tuple(['brows']): [2, 3],
            tuple(['lips']): [12, 13]
        }
        # self.combined = ['skin', 'r_ear', 'ear_r']
        self.combined = ['l_ear', 'r_ear', 'skin', 'mouth', 'nose', 'l_brow', 'r_brow', 'neck', 'neck_l', 'hair', 'l_lip', 'u_lip']
        print(f"Loaded model from {model_path}")
        self.parsed = self.model_inference()

    def load_image(self):
        try:
            image = cv2.imread(self.image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            print(f"self.width {self.width}, self.height {self.height}")
            self.image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LANCZOS4)
            print(f"Loaded image from {self.image_path}")
            print(f"Image shape {self.image.shape}")

        except:
            print(f"Error loading image from {self.image_path}")
            sys.exit(1)
            return None
        return image

    def model_inference(self):
        if self.image is None:
            print(f"Error: image is None")
            return None
        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        image = to_tensor(np.array(self.image))
        image = image.unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.net(image)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        #resize parsing to original image size
        self.parsing = cv2.resize(parsing, (self.image.shape[1], self.image.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        return self.parsing 

    def get_part_mask(self, parts):
        print(f"parts {parts}")
        part_mask = np.zeros_like(self.parsing)
        for part in parts:
            part_mask += np.isin(self.parsing, self.part_labels[part]).astype(np.uint8)
        return part_mask

    def get_image(self):
        return self.image

    def get_skin(self, parts=['l_ear', 'r_ear','skin', 'mouth','nose','l_brow', 'r_brow', 'neck', 'neck_l', 'l_lip', 'u_lip','hair']):
        part_mask = self.get_part_mask(parts)
        binary_skin_mask = (part_mask == 1).astype(np.uint8) * 255
        part_mask = cv2.resize(part_mask, (self.image.shape[1], self.image.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        skin_mask = cv2.resize(binary_skin_mask, (self.image.shape[1], self.image.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        skin_mask_stacked = np.dstack((self.skin_mask,) * 3)
        skin_uint8 = part_mask.astype(np.uint8)
        skin_mask = skin_uint8
        # Ensure the mask is the same size as the image
        if skin_uint8.shape[:2] != self.image.shape[:2]:
            # Resize mask to match image
            skin_uint8 = cv2.resize(skin_uint8, (self.image.shape[1], self.image.shape[0]))
        # Now apply the mask
        self.skin = cv2.bitwise_and(self.image, self.image, mask=skin_uint8)
        self.skin_mask = skin_uint8
        return self.skin_mask
    def get_skin_color(self):
        skin = cv2.bitwise_and(self.image, self.image, mask=self.skin_mask)
        #get rid of lightest and darkest 5% of pixels
        skin_hsv = cv2.cvtColor(skin, cv2.COLOR_RGB2HSV)
        skin_hsv = skin_hsv.reshape((skin_hsv.shape[0] * skin_hsv.shape[1], 3))
        skin_hsv = skin_hsv[np.argsort(skin_hsv[:, 2])]
        # Get the average color of the skin
        average_color_per_row = np.average(skin, axis=0)
        average_color = np.average(average_color_per_row, axis=0)
        average_color = np.uint8(average_color)
        #tile the average color
        average_color_tile = np.tile(average_color, (self.image.shape[0], self.image.shape[1], 1))
        self.skin_tile = average_color_tile
        return average_color
    def get_hair(self):
            part_mask = self.get_part_mask(['hair'])
            hair_value = 1
            # Resize the part_mask
            binary_hair_mask = (part_mask == 1).astype(np.uint8) * 255
            part_mask = cv2.resize(part_mask, (self.image.shape[1], self.image.shape[0]), interpolation=cv2.INTER_LANCZOS4)
            hair_mask = cv2.resize(binary_hair_mask, (self.image.shape[1], self.image.shape[0]), interpolation=cv2.INTER_LANCZOS4)
            hair_mask_stacked = np.dstack((hair_mask,) * 3)
            hair_uint8 = part_mask.astype(np.uint8)
            # Ensure the mask is the same size as the image
            if hair_uint8.shape[:2] != self.image.shape[:2]:
                # Resize mask to match image
                hair_uint8 = cv2.resize(hair_uint8, (self.image.shape[1], self.image.shape[0]))
            # Now apply the mask
            self.hair = cv2.bitwise_and(self.image, self.image, mask=hair_uint8)
            self.hair_mask = hair_uint8
            return self.hair
    def get_background(self):
        parts=['skin', 'r_ear', 'ear_r', 'l_ear', 'mouth', 'nose', 'l_brow', 'r_brow', 'neck', 'neck_l', 'hair', 'l_lip', 'u_lip', 'l_eye', 'r_eye']
        # Get the combined mask for the specified parts
        part_mask = self.get_part_mask(parts)
        binary_skin_mask = (part_mask == 1).astype(np.uint8) * 255
        # Resize part_mask to the original image size
        part_mask = cv2.resize(part_mask, (self.image.shape[1], self.image.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        binary_skin_mask = cv2.resize(binary_skin_mask, (self.image.shape[1], self.image.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        # Invert the mask to get the background mask
        background_mask = cv2.bitwise_not(binary_skin_mask)
        # Stack the mask to create a 3-channel image if needed
        if len(background_mask.shape) == 2:
            background_mask_stacked = np.dstack((background_mask,) * 3)
        else:
            background_mask_stacked = background_mask
        # Apply the background mask to the original image to get the background
        self.background = cv2.bitwise_and(self.image, self.image, mask=background_mask)
        return self.background
    def swap_background(self, original_img, decoded_img, mask):
        # Resize mask to match the original image
        mask_resized = cv2.resize(mask, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # No need to convert to RGB if images are already in the correct color space
        # original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        # decoded_img = cv2.cvtColor(decoded_img, cv2.COLOR_BGR2RGB)

        # Resize the images to match if they are not already the same size
        if original_img.shape[:2] != decoded_img.shape[:2]:
            original_img = cv2.resize(original_img, (decoded_img.shape[1], decoded_img.shape[0]), interpolation=cv2.INTER_LANCZOS4)

        # Ensure mask is single-channel for cv2.bitwise_and
        if mask_resized.ndim > 2:
            mask_resized = mask_resized[:, :, 0]

        # Invert mask to create a background mask
        background_mask = cv2.bitwise_not(mask_resized)

        # Extract the subject from the decoded image using the mask
        foreground = cv2.bitwise_and(decoded_img, decoded_img, mask=background_mask)

        # Extract the background from the original image using the inverted mask
        background = cv2.bitwise_and(original_img, original_img, mask=mask_resized)

        # Combine the subject and the background
        combined_img = cv2.add(foreground, background)

        return combined_img

if __name__ == '__main__':
    image_dir = r"C:\Users\joeli\Dropbox\Code\Python Projects\Texture_Image_Pipeline\fitzpatrick"
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    for image_path in image_paths:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        segmenter = FacePartSegmentation(width=2048, height=2048)
        eyes = ['l_eye', 'r_eye']
        ears = ['l_ear', 'r_ear']
        lips = ['u_lip', 'l_lip']
        combined = ears + eyes + lips
        print(f"combined {combined}")
        part_mask = segmenter.get_part_mask(image, combined)
        plt.imshow(part_mask)
        plt.show()
        # image = ... # load your image as a numpy array
        # part_mask = segmenter.get_part_mask(image, 'ears')

