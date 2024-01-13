import torch
import torchvision.transforms as transforms
# from PIL import Image
from model import BiSeNet
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

class FacePartSegmentation:
    def __init__(self,width, height, model_path=r"C:\Users\joeli\Dropbox\Code\Python Projects\Texture_Image_Pipeline\torch_face\79999_iter.pth", device='cuda'):
        self.width = width
        self.height = height
        self.device = device
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
        print(f"Loaded model from {model_path}")

    def model_inference(self, image):
        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        image = to_tensor(np.array(image))
        image = image.unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.net(image)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        return parsing

    def get_part_mask(self, parsing_anno, part_label, stride=1):
        parsing_anno = cv2.resize(parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
        mask = np.zeros(parsing_anno.shape, dtype=np.uint8)
        index = self.part_labels[part_label][0]
        mask[parsing_anno == index] = 255
        return mask

    def process_image(self, image, part_labels):
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LANCZOS4)
        masks = []
        if tuple(part_labels) in self.combined_part_labels:
            part_indices = self.combined_part_labels[tuple(part_labels)]
        for part_label in part_labels:
            #check if part_label is valid
            parsing = self.model_inference(image)
            # Correctly pass both lists of labels to get_part_mask
            part_mask = self.get_part_mask(parsing, part_label, stride=1)
            masks.append(part_mask)
        part_mask = np.any(masks, axis=0)
        return part_mask

    def get_skin(self, image, combined=['skin', 'r_ear', 'ear_r']):
        # Convert PIL Image to NumPy array if it's not already
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LANCZOS4)
        image = np.array(image).astype(np.uint8)
        parsing = self.model_inference(image)
        # combined = ['l_ear', 'r_ear','skin', 'mouth']
        combined = ['l_ear', 'r_ear','skin', 'mouth', 'nose', 'l_brow', 'r_brow', 'neck', 'neck_l', 'hair', 'l_lip', 'u_lip']
        part_mask = self.process_image(image, combined)
        print(f"part_mask shape {part_mask.shape}")
        skin_value = 1
        binary_skin_mask = (part_mask == skin_value).astype(np.uint8) * 255
        binary_skin_mask_stacked = np.dstack((binary_skin_mask,) * image.shape[2])
        print(f"binary_skin_mask_stacked shape {binary_skin_mask_stacked.shape}")
        print(f"image shape {image.shape}")
        # binary_skin_mask_stacked = np.dstack((binary_skin_mask, binary_skin_mask, binary_skin_mask))
        # skin = cv2.bitwise_and(image, binary_skin_mask_stacked)
        binary_skin_mask_stacked = cv2.resize(binary_skin_mask_stacked, (image.shape[1], image.shape[0]))
        skin = cv2.bitwise_and(image, binary_skin_mask_stacked)
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(skin)
        ax[1].imshow(part_mask)
        plt.show()
        return skin, part_mask
    
    def get_skin_tile(self, image):
        # skin, part_mask = self.get_skin(image, combined=['l_ear', 'r_ear', 'skin', 'mouth'])
        skin, part_mask = self.get_skin(image, combined=['l_ear', 'r_ear', 'skin', 'mouth', 'nose'])
        print(f"skin shape {skin.shape}")
        non_black_mask = np.any(skin != [0, 0, 0], axis=-1)
        avg_color = np.mean(skin[non_black_mask], axis=0)
        
        # Ensure avg_color is a one-dimensional array with 3 elements
        assert avg_color.shape == (3,), "Average color must be a 3-element vector."
        
        # Create an image filled with the average color
        avg_color_tile = np.tile(avg_color.reshape(1, 1, 3), (self.width,self.height,1))
        
        return avg_color_tile

    def get_hair(self, image):
        parsing = self.model_inference(image)
        part_mask = self.process_image(image,['hair'])
        hair_value = 1
        binary_hair_mask = (part_mask == hair_value).astype(np.uint8) * 255
        binary_hair_mask_stacked = np.dstack((binary_hair_mask, binary_hair_mask, binary_hair_mask))
        hair = cv2.bitwise_and(image, binary_hair_mask_stacked)
        return hair

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
        part_mask = segmenter.process_image(image, combined)
        plt.imshow(part_mask)
        plt.show()
        # image = ... # load your image as a numpy array
        # part_mask = segmenter.process_image(image, 'ears')

