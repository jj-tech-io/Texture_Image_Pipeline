import os
import sys
import numpy as np
import cv2
import time
import tensorflow as tf
from tensorflow.keras.models import load_model

class TensorFlowAutoencoder:
    def __init__(self, encoder_path, decoder_path, device='cuda'):
        self.device = device
        try:
            self.encoder = load_model(encoder_path)
            self.decoder = load_model(decoder_path)
        except Exception as e:
            print(f"An error occurred while loading models: {e}")
            sys.exit(1)

    def reverse_gamma_correction(self, img):
        return np.where(img > 0.04045, ((img + 0.055) / 1.055) ** 2.4, img / 12.92)

    def gamma_correction(self, img):
        return np.where(img > 0.0031308, 1.055 * (img ** (1 / 2.4)) - 0.055, 12.92 * img)

    def encode(self, img):
        img = np.asarray(img).reshape(-1, 3).astype('float32')
        if np.max(img) > 1:
            img = img / 255.0
        img = self.reverse_gamma_correction(img)
        with tf.device('/device:GPU:0'):
            start = time.time()
            pred_maps = self.encoder.predict_on_batch(img)
            elapsed = time.time() - start
        return pred_maps, elapsed

    def decode(self, encoded):
        with tf.device('/device:GPU:0'):
            start = time.time()
            recovered = self.decoder.predict_on_batch(encoded)
            elapsed = time.time() - start
        if np.max(recovered) > 2:
            recovered = recovered / 255.0 
        recovered = self.gamma_correction(recovered)
        return recovered, elapsed