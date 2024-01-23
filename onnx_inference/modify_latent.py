import cv2
import numpy as np

import onnx_inference
from onnx_inference import autoencoder
from onnx_inference import modify_latent

from matplotlib import pyplot as plt

def age_mel(v, t, r=0.08):
    """
    v is original volume fraction of melanin
    t is number of decades
    r is rate of decline (typical is 8%)
    """
    v_prime = v - (t * r) * v
    return v_prime


def age_hem(v, t, r_Hbi=0.06, r_Hbe=0.1, zeta=0.5):
    """
    v is original volume fraction of hemoglobin
    t is number of decades
    r is rate of decline (typical is 6%)
    """
    v_prime = v - t * (r_Hbi + zeta * r_Hbe) * v
    return v_prime


def get_masks(image):
    # image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    WIDTH = image.shape[0]
    HEIGHT = image.shape[1]
    image = cv2.resize(image, (WIDTH, HEIGHT))
    image = image.astype(np.float32)
    image = image.reshape((-1,3))
    if np.max(image) > 1:
        image = image / 255.0
    # parameter_maps = np.asarray(onnx_ae.encode(image))
    parameter_maps = np.asarray(autoencoder.encode(image))
    print(f"parameter_maps.shape: {parameter_maps.shape}")
    Cm = parameter_maps[:, 0].reshape(WIDTH, HEIGHT)
    # Cm = (Cm - np.min(Cm)) / (np.max(Cm) - np.min(Cm))
    Ch = parameter_maps[:, 1].reshape(WIDTH, HEIGHT)
    # Ch = (Ch - np.min(Ch)) / (np.max(Ch) - np.min(Ch))
    Bm = parameter_maps[:, 2].reshape(WIDTH, HEIGHT)
    # Bm = (Bm - np.min(Bm)) / (np.max(Bm) - np.min(Bm))
    Bh = parameter_maps[:, 3].reshape(WIDTH, HEIGHT)
    # Bh = (Bh - np.min(Bh)) / (np.max(Bh) - np.min(Bh))
    T = parameter_maps[:, 4].reshape(WIDTH, HEIGHT)
    # T = (T - np.min(T)) / (np.max(T) - np.min(T))
    # Cm = np.clip(Cm, 0, 1)
    # Ch = np.clip(Ch, 0, 1)
    # Bm = np.clip(Bm, 0, 1)
    # Bh = np.clip(Bh, 0, 1)
    # T = np.clip(T, 0, 1)
    return Cm, Ch, Bm, Bh, T