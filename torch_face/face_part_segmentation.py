import torch
import torchvision.transforms as transforms
# from PIL import Image
from model import BiSeNet
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

class FacePartSegmentation:
    def __init__(self,image_path, width=4096, height=4096):
        model_path=r"C:\Users\joeli\Dropbox\Code\Python Projects\Texture_Image_Pipeline\torch_face\79999_iter.pth"
        self.image = None
        self.part_mask = None
        self.skin = None
        self.skin_tile = None
        self.parsing = None
        self.width = width
        self.height = height
        self.image_path = image_path    
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
        self.parsing = out.squeeze(0).cpu().numpy().argmax(0)
        return self.parsing 

    def get_part_mask(self, parsing_anno, part_label, stride=1):
        parsing_anno = cv2.resize(parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_LANCZOS4)
        mask = np.zeros(parsing_anno.shape, dtype=np.uint8)
        index = self.part_labels[part_label][0]
        mask[parsing_anno == index] = 255
        #resize mask to original image size
        mask = cv2.resize(mask, (self.image.shape[1], self.image.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        return mask

    def process_image(self):
        masks = []
        part_labels = self.combined
        if tuple(part_labels) in self.combined_part_labels:
            part_indices = self.combined_part_labels[tuple(part_labels)]
        for part_label in part_labels:
            #check if part_label is valid
            if self.image is None:
                print(f"Error: image is None")
                return None
            parsing = self.model_inference()
            # Correctly pass both lists of labels to get_part_mask
            self.part_mask = self.get_part_mask(parsing, part_label, stride=1)
            masks.append(self.part_mask)
        self.part_mask = np.any(masks, axis=0)
        # Convert boolean array to uint8
        self.part_mask = self.part_mask.astype(np.uint8) * 255

        # Resize the part_mask
        self.part_mask = cv2.resize(self.part_mask, (self.image.shape[1], self.image.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        return self.part_mask
    def get_image(self):
        return self.image
    
    def get_skin(self):
        part_mask = self.process_image()
        
        skin_value = 1

        binary_skin_mask = (part_mask == skin_value).astype(np.uint8) * 255
        print(np.unique(binary_skin_mask))

        binary_skin_mask_stacked = np.dstack((binary_skin_mask,) * 3)
        binary_skin_mask_stacked = cv2.resize(binary_skin_mask_stacked, (self.image.shape[1], self.image.shape[0]))
        skin = cv2.bitwise_and(self.image, binary_skin_mask_stacked)
        skin = cv2.resize(skin, (self.width, self.height))
        self.skin = skin
        return self.skin, binary_skin_mask_stacked
    
    def get_skin_tile(self):
        if self.skin_tile is not None:
            return self.skin_tile
        non_black_mask = np.any(self.skin != [0, 0, 0], axis=-1)
        avg_color = np.mean(self.skin[non_black_mask], axis=0)

        self.skin_tile = np.tile(avg_color.reshape(1, 1, 3), (self.width, self.height, 1))
        self.skin_tile = self.skin_tile.astype(np.uint8)
        self.skin_tile = cv2.resize(self.skin_tile, (self.width, self.height))
        return self.skin_tile

    def get_hair(self):
        if self.parsing is None:
            try:
                parsing = self.model_inference()  # No argument passed
            except Exception as e:
                print(f"Error parsing image, error {e}")
                return None
        else:
            part_mask = self.process_image()
            parsing = self.model_inference()
            part_mask = self.process_image()
            hair_value = 1
            binary_hair_mask = (part_mask == hair_value).astype(np.uint8) * 255
            binary_hair_mask_stacked = np.dstack((binary_hair_mask, binary_hair_mask, binary_hair_mask))
            hair = cv2.bitwise_and(self.image, binary_hair_mask_stacked)
            hair = cv2.resize(hair, (self.width, self.height))
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

