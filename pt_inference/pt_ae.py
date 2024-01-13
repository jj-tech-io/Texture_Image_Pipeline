#%%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
from torch.optim.lr_scheduler import StepLR
import time
from datetime import datetime
import sys
import importlib
import cv2

class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, num_neurons):
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, num_neurons),
            nn.ReLU(),
            nn.Linear(num_neurons, num_neurons),
            nn.ReLU(),
            nn.Linear(num_neurons, output_dim)
        )

    def forward(self, x):
        x = self.layers(x)
        x = (x - x.min()) / (x.max() - x.min())
        return x

class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, num_neurons):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, num_neurons),
            nn.ReLU(),
            nn.Linear(num_neurons, num_neurons),
            nn.ReLU(),
            nn.Linear(num_neurons, output_dim)
        )

    def forward(self, x):
        x = self.layers(x)
        return x

class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, pixel, parameter):
        encoder_layer = self.encoder(pixel)
        decoder_layer = self.decoder(parameter)
        end_to_end_layer = self.decoder(encoder_layer)
        return encoder_layer, decoder_layer, end_to_end_layer

class LossFunctions:
    @staticmethod
    def albedo_loss(y_true, y_pred):
        return torch.mean(torch.sum(torch.abs(y_pred - y_true), dim=-1))

    @staticmethod
    def parameter_loss(y_true, y_pred):
        l2_norm = torch.sqrt(torch.sum(torch.square(y_pred - y_true), dim=-1))
        return torch.mean(l2_norm)  # Ensure it's a scalar

    @staticmethod
    def end_to_end_loss(y_true, y_pred):
        return torch.mean(torch.sum(torch.abs(y_pred - y_true), dim=-1))
        
    @staticmethod
    def spectral_angle_loss(y_true, y_pred, eps=1e-8):
        """
        Compute the mean spectral angle loss between y_true and y_pred.
        y_true and y_pred should be of the same shape.
        Args:
            y_true (tensor): Ground truth tensor.
            y_pred (tensor): Predicted tensor.
            eps (float): A small epsilon value to avoid division by zero.
        Returns:
            torch.Tensor: The mean spectral angle loss.
        """ 
        # Normalize the vectors to unit vectors
        y_true_norm = y_true / (torch.norm(y_true, p=2, dim=-1, keepdim=True) + eps)
        y_pred_norm = y_pred / (torch.norm(y_pred, p=2, dim=-1, keepdim=True) + eps)

        # Compute the dot product
        dot_product = torch.sum(y_true_norm * y_pred_norm, dim=-1)

        # Clamp the dot product values to avoid numerical issues with arccos
        dot_product = torch.clamp(dot_product, -1.0, 1.0)

        # Calculate the spectral angle
        spectral_angle = torch.acos(dot_product)

        # Return the mean spectral angle loss
        return torch.mean(spectral_angle)
def albedo_loss(y_true, y_pred):
    return torch.mean(torch.sum(torch.abs(y_pred - y_true), dim=-1))

def parameter_loss(y_true, y_pred):
    l2_norm = torch.sqrt(torch.sum(torch.square(y_pred - y_true), dim=-1))
    return torch.mean(l2_norm)  # Ensure it's a scalar

def end_to_end_loss(y_true, y_pred):
    return torch.mean(torch.sum(torch.abs(y_pred - y_true), dim=-1))




class MyAutoencoder:
    def __init__(self, device='cuda:0'):
        self.device = device
        self.encoder = Encoder(input_dim=3, output_dim=5, num_neurons=75).to(self.device)
        self.decoder = Decoder(input_dim=5, output_dim=3, num_neurons=75).to(self.device)
        self.autoencoder = AutoEncoder(self.encoder, self.decoder).to(self.device)
        file_dir = os.path.dirname(os.path.realpath(__file__))
        encoder_path = r'encoder_rgb_best_316.pth'
        decoder_path = r'decoder_rgb_best_316.pth'
        encoder_path = os.path.join(file_dir, encoder_path)
        decoder_path = os.path.join(file_dir, decoder_path)
        # C:\Users\joeli\Dropbox\Code\Python Projects\AE_Pytorch\autoencoder_rgb_best_316.pth
        encoder_path = r'C:\Users\joeli\Dropbox\Code\Python Projects\AE_Pytorch\encoder_rgb_best_316.pth'
        decoder_path = r'C:\Users\joeli\Dropbox\Code\Python Projects\AE_Pytorch\decoder_rgb_best_316.pth'
        self.load_state(encoder_path, decoder_path)
        # Set the models to evaluation mode
        self.encoder.eval()
        self.decoder.eval()
    def get_encoder(self):
        return self.encoder
    def get_decoder(self):
        return self.decoder
    def load_state(self, encoder_path, decoder_path):
        # encoder_state_dict = torch.load(encoder_path)
        # self.encoder.load_state_dict(encoder_state_dict)
        # Load state dictionaries
        encoder_state_dict = torch.load(encoder_path)
        decoder_state_dict = torch.load(decoder_path)
        # Attempt to update the model with the loaded state dictionaries
        try:
            self.encoder.load_state_dict(encoder_state_dict)
            self.decoder.load_state_dict(decoder_state_dict)
        except RuntimeError as e:
            print("Error loading state dict for encoder:", e)
    def reverse_gamma_correction(self, img):
        """Reverse gamma correction on an image."""
        if np.max(img) > 1:
            img = img / 255.0
        # return np.where(img > 0.04045, ((img + 0.055) / 1.055) ** 2.4, img / 12.92)
        return img

    def gamma_correction(self, img):
        """Gamma correction on an image."""
        if np.max(img) > 1:
            img = img / 255.0
        # return np.where(img > 0.0031308, 1.055 * (img ** (1 / 2.4)) - 0.055, 12.92 * img)
        return img

    def encode_image(self, image, width=2048, height=2048):
        self.width = width
        self.height = height
        """Encodes an image using the encoder part of the autoencoder."""
        # image = image[:,:,:3]  # Ensure image has 3 channels
        image = cv2.resize(image, (width, height))
        image = np.asarray(image, dtype=np.float32)
        image = self.reverse_gamma_correction(image)
        width, height, _ = image.shape
        image = torch.from_numpy(image).reshape(width * height, 3)
        image = image.to(self.device, dtype=torch.float32)
        with torch.no_grad():
            self.encoder.eval()
            encoded = self.encoder(image)
            encoded_features = encoded.cpu().numpy()
            # encoded_features = [encoded[:, i].cpu().numpy().reshape(width, height) for i in range(encoded.size(1))]
        encoded_features = np.asarray(encoded_features)
        print("encoded_features shape:", encoded_features.shape)
        return encoded_features

    def decode_image(self, encoded, width=2048, height=2048):
        """Decodes an encoded image using the decoder part of the autoencoder."""
        #reshape encoded
        encoded = np.asarray(encoded).reshape(-1, 5)
        encoded_tensor = torch.tensor(encoded).to(self.device, dtype=torch.float32)
        with torch.no_grad():
            self.decoder.eval()
            decoded = self.decoder(encoded_tensor)
            decoded = decoded.cpu().numpy()
            # decoded = self.gamma_correction(decoded)
        decoded = decoded.reshape(width, height, 3)
        return decoded

# %%
