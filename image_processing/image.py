import sys
import cv2
import numpy as np

class Jimage:
    def __init__(self, image_path=None, width=None, height=None):
        self.image_path = image_path
        try:
            if image_path is not None:
                self.image = self.load_image(image_path)
            elif image is not None:
                self.image = image
            else:
                print("Error: image_path and image are None")
                return None
        except Exception as e:
            print(f"Error loading image from {image_path}, error {e}")
            return None
            
    def load_image(self, image_path, width=None, height=None):
        if width is None or height is None:
            return cv2.imread(image_path)
        else:
            return cv2.resize(cv2.imread(image_path), (width, height))

    def get_image(self):
        return self.image

    def set_image(self, image = None, path = None):
        try:
            if image is not None:
                self.image = image
            elif path is not None:
                self.image = self.load_image(path)
            else:
                print("Error: image and path are None")
                return None
        except Exception as e:
            print(f"Error loading image from {path}, error {e}")
            return None