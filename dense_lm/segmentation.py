import matplotlib.pyplot as plt
import os
import json
import cv2
import numpy as np
import mediapipe as mp
import skimage
from skimage.transform import PiecewiseAffineTransform, warp
import cv2
import mediapipe as mp
import skimage
from skimage.transform import PiecewiseAffineTransform, warp
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import mediapipe as mp
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import face_mesh as mp_face_mesh
import random

import sys
from pathlib import Path
current_script_dir = Path(__file__).resolve()
# Get the parent directory (one level up)
parent_dir = current_script_dir.parent
# Get the root directory (two levels up or more if needed)
# root_dir = current_script_dir.parents[1]  # Use parents[2], parents[3], etc., for higher levels
# Add both directories to sys.path
sys.path.append(str(parent_dir))
# sys.path.append(str(root_dir))
import dense_lm

from dense_lm import landmarks as lm
import onnx_inference
from onnx_inference import onnx_ae
from onnx_inference import modify_latent


LIPS = frozenset([
    # Lips.
    (61, 146),
    (146, 91),
    (91, 181),
    (181, 84),
    (84, 17),
    (17, 314),
    (314, 405),
    (405, 321),
    (321, 375),
    (375, 291),
    (61, 185),
    (185, 40),
    (40, 39),
    (39, 37),
    (37, 0),
    (0, 267),
    (267, 269),
    (269, 270),
    (270, 409),
    (409, 291),
    (78, 95),
    (95, 88),
    (88, 178),
    (178, 87),
    (87, 14),
    (14, 317),
    (317, 402),
    (402, 318),
    (318, 324),
    (324, 308),
    (78, 191),
    (191, 80),
    (80, 81),
    (81, 82),
    (82, 13),
    (13, 312),
    (312, 311),
    (311, 310),
    (310, 415),
    (415, 308)])
LEFT_EYE = frozenset([
    # Left eye.
    (263, 249),
    (249, 390),
    (390, 373),
    (373, 374),
    (374, 380),
    (380, 381),
    (381, 382),
    (382, 362),
    (263, 466),
    (466, 388),
    (388, 387),
    (387, 386),
    (386, 385),
    (385, 384),
    (384, 398),
    (398, 362) ])
LEFT_EYEBROW = frozenset([
    # Left eyebrow.
    (276, 283),
    (283, 282),
    (282, 295),
    (295, 285),
    (300, 293),
    (293, 334),
    (334, 296),
    (296, 336) ])
RIGHT_EYE = frozenset([
    # Right eye.
    (33, 7),
    (7, 163),
    (163, 144),
    (144, 145),
    (145, 153),
    (153, 154),
    (154, 155),
    (155, 133),
    (33, 246),
    (246, 161),
    (161, 160),
    (160, 159),
    (159, 158),
    (158, 157),
    (157, 173),
    (173, 133) ])
RIGHT_EYEBROW = frozenset([
    # Right eyebrow.
    (46, 53),
    (53, 52),
    (52, 65),
    (65, 55),
    (70, 63),
    (63, 105),
    (105, 66),
    (66, 107) ])
FACE_OVAL = frozenset([
    # Face oval.
    (10, 338),
    (338, 297),
    (297, 332),
    (332, 284),
    (284, 251),
    (251, 389),
    (389, 356),
    (356, 454),
    (454, 323),
    (323, 361),
    (361, 288),
    (288, 397),
    (397, 365),
    (365, 379),
    (379, 378),
    (378, 400),
    (400, 377),
    (377, 152),
    (152, 148),
    (148, 176),
    (176, 149),
    (149, 150),
    (150, 136),
    (136, 172),
    (172, 58),
    (58, 132),
    (132, 93),
    (93, 234),
    (234, 127),
    (127, 162),
    (162, 21),
    (21, 54),
    (54, 103),
    (103, 67),
    (67, 109),
    (109, 10)])
LEFT_NOSE = frozenset([
    # Left nose.
    (97,75),
    (75,79),
    (79,237),
    (237,242),
    (242,97)])
RIGHT_NOSE = frozenset([
    (328,354),
    (354,274),
    (274,309),
    (309,305),
    (305,328)])
UPPER_LIPS = frozenset([
    (61, 146),
    (146, 91),
    (91, 181),
    (181, 84),
    (84, 17),
    (17, 314),
    (314, 405),
    (405, 321),
    (321, 375),
    (375, 291),
    (61, 185),
    (185, 40),
    (40, 39),
    (39, 37),
    (37, 0),
    (0, 267),
    (267, 269),
    (269, 270),
    (270, 409),
    (409, 291),])
LOWER_LIPS = frozenset([
    (78, 95),
    (95, 88),
    (88, 178),
    (178, 87),
    (87, 14),
    (14, 317),
    (317, 402),
    (402, 318),
    (318, 324),
    (324, 308),
    (78, 191),
    (191, 80),
    (80, 81),
    (81, 82),
    (82, 13),
    (13, 312),
    (312, 311),
    (311, 310),
    (310, 415),
    (415, 308)])
NOSE = frozenset([
    #197,98,2,327,197
    #197,48,64,98,97,326,327,294,278
    (197,48),
    (48,64),
    (64,98),
    (98,97),
    (97,326),
    (326,327),
    (327,294),
    (294,278),
    (278,197)
    ])
EYE_BAG_LEFT = frozenset([
    #155,22,23,24,110,25,31,228,229,230,231,155
    (112,22),
    (22,23),
    (23,24),
    (24,110),
    (110,25),
    (25,31),
    (31,228),
    (228,229),
    (229,230),
    (230,231),
    (231,112)])
EYE_BAG_RIGHT = frozenset([
    #453,253,254,339,255,261,448,449,450,451,453
    (453,253),
    (253,254),
    (254,339),
    (339,255),
    (255,261),
    (261,448),
    (448,449),
    (449,450),
    (450,451),
    (451,453)])
FRECKLE_MASK = frozenset([
    #6,351,153,452,451,450,449,448,261,340,352,411,427,426,423,358, 19,237,129,203,206,207,187,123,228,229,230,231,233,122,6
    (6,351),
    (351,153),
    (153,452),
    (452,451),
    (451,450),
    (450,449),
    (449,448),
    (448,261),
    (261,340),
    (340,352),
    (352,411),
    (411,427),
    (427,426),
    (426,423),
    (423,358),
    (358, 19),
    (19,237),
    (237,129),
    (129,203),
    (203,206),
    (206,207),
    (207,187),
    (187,123),
    (123,228),
    (228,229),
    (229,230),
    (230,231),
    (231,233),
    (233,122),
    (122,6)])
BLUSH_MASK = frozenset([
    (6,451),
    (451,261),
    (261,372),
    (372,345),
    (345,433),
    (433,416),
    (416,434),
    (434,432),
    (432,426),
    (426,423),
    (423,2),
    (2,203),
    (203,212),
    (212,214),
    (214,192),
    (192,213),
    (213,116),
    (116,35),
    (35,228),
    (228,231),
    (231,6)])

LIPS = np.array(list(LIPS))
EYES = np.array(list(LEFT_EYE) + list(RIGHT_EYE))
NOSTRILS = np.array(list(LEFT_NOSE) + list(RIGHT_NOSE))
EYEBROWS = np.array(list(LEFT_EYEBROW) + list(RIGHT_EYEBROW))
EYE_BAGS = np.array(list(EYE_BAG_LEFT) + list(EYE_BAG_RIGHT))
LEFT_EYE = np.array(list(LEFT_EYE))
LEFT_EYEBROW = np.array(list(LEFT_EYEBROW))
RIGHT_EYE = np.array(list(RIGHT_EYE))
RIGHT_EYEBROW = np.array(list(RIGHT_EYEBROW))
FACE_OVAL = np.array(list(FACE_OVAL))
LEFT_NOSE = np.array(list(LEFT_NOSE))
RIGHT_NOSE = np.array(list(RIGHT_NOSE))
UPPER_LIPS = np.array(list(UPPER_LIPS))
LOWER_LIPS = np.array(list(LOWER_LIPS))
NOSE = np.array(list(NOSE))
EYE_BAG_LEFT = np.array(list(EYE_BAG_LEFT))
EYE_BAG_RIGHT = np.array(list(EYE_BAG_RIGHT))
FRECKLE_MASK = np.array(list(FRECKLE_MASK))
BLUSH_MASK = np.array(list(BLUSH_MASK))

def annotate_landmarks(image, face_landmarks, segments, color=(255,0,0), thickness=5):
    points = []
    for segment in segments:
        p1, p2 = int(segment[0]), int(segment[1])
        p1x = int(face_landmarks.landmark[p1].x * image.shape[1])
        p1y = int(face_landmarks.landmark[p1].y * image.shape[0])
        p2x = int(face_landmarks.landmark[p2].x * image.shape[1])
        p2y = int(face_landmarks.landmark[p2].y * image.shape[0])
        if [p1x, p1y] not in points:
            points.append([p1x, p1y])
        if [p2x, p2y] not in points:
            points.append([p2x, p2y])
        cv2.putText(image, str(p1), (p1x, p1y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, str(p2), (p2x, p2y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return image
def generate_mask(image, landmarks, draw_function, indices, blur_kernel=(5,5),translate=(0,0)):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask = draw_function(mask, landmarks, indices, color=(255,255,255),translate=translate)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)
    mask = cv2.GaussianBlur(mask, blur_kernel, 0)
    return mask

def draw_lines(image, face_landmarks, segments, color=(255,0,0), thickness=5, translate=(0,0)):
    points = []
    for i in segments:
        p1 = i[0]
        p2 = i[1]
        p1x = int(face_landmarks.landmark[p1].x * image.shape[1])+translate[0]
        p1y = int(face_landmarks.landmark[p1].y * image.shape[0])+translate[1]
        p2x = int(face_landmarks.landmark[p2].x * image.shape[1])+translate[0]
        p2y = int(face_landmarks.landmark[p2].y * image.shape[0])+translate[1]
        points.append([p1x, p1y])
        points.append([p2x, p2y])
        #flood = fill_contour(image, color, (p1x, p1y))
        #poly line p1 to p2
        cv2.line(image, (p1x, p1y), (p2x, p2y), color, thickness=thickness)
        # cv2.polylines(image, np.array([points]), False, color, thickness=50)
    # Draw lines
    for i in range(0, len(points), 2):
        p1 = points[i]
        p2 = points[i + 1]
        cv2.line(image, p1, p2, color, thickness=5)
    seed = points[-1]
    cv2.floodFill(image, None, seedPoint=(seed[0], seed[1]), newVal=color, loDiff=(0,0,0,0), upDiff=(0,0,0,0), flags=4 | cv2.FLOODFILL_MASK_ONLY)
    center = (sum([p[0] for p in points])//len(points), sum([p[1] for p in points])//len(points))

    return image
def create_combined_mask(image):
    print(f"image shape: {image.shape}")
    mp_face_mesh = mp.solutions.face_mesh
    # Process the image with MediaPipe Face Mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        min_detection_confidence=0.15) as face_mesh:
        results = face_mesh.process(image)
        if not results.multi_face_landmarks:
            return None, None
        face_landmarks = results.multi_face_landmarks[0]
        # Generate masks for different parts
        lips = generate_mask(image, face_landmarks, draw_lines, LIPS)
        eyes = generate_mask(image, face_landmarks, draw_lines, np.array(list(LEFT_EYE) + list(RIGHT_EYE)), translate=(0,0)) 
        nose = generate_mask(image, face_landmarks, draw_lines, np.array(list(LEFT_NOSE) + list(RIGHT_NOSE)), translate=(0,0))
        face = generate_mask(image, face_landmarks, draw_lines, FACE_OVAL)
        combined_mask = cv2.bitwise_or(lips, eyes)
        combined_mask = cv2.bitwise_or(combined_mask, nose)
        av_skin_color = np.mean(image[face == 255], axis=0)
        combined_mask = cv2.bitwise_not(combined_mask)
        combined_mask = cv2.bitwise_and(combined_mask, face)
        print(f"image shape: {image.shape}")
        return combined_mask, face, av_skin_color

def threshold_face_skin_area(img,av_skin_color,mask):
    masked_hsv = cv2.cvtColor(cv2.bitwise_or(img, img, mask=mask), cv2.COLOR_BGR2HSV)
    # Compute mean and standard deviation for each channel using the masked area
    mean_hue = np.mean(masked_hsv[:,:,0][masked_hsv[:,:,0] > 0])
    std_hue = np.std(masked_hsv[:,:,0][masked_hsv[:,:,0] > 0])
    mean_sat = np.mean(masked_hsv[:,:,1][masked_hsv[:,:,1] > 0])
    std_sat = np.std(masked_hsv[:,:,1][masked_hsv[:,:,1] > 0])
    mean_val = np.mean(masked_hsv[:,:,2][masked_hsv[:,:,2] > 0])
    std_val = np.std(masked_hsv[:,:,2][masked_hsv[:,:,2] > 0])
    skin_color_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Define thresholds based on the mean and standard deviation for each channel
    lower_bound = [max(0, mean_hue - 2*std_hue), max(0, mean_sat - 2*std_sat), max(0, mean_val - 2*std_val)]
    upper_bound = [max(255, mean_hue + 2*std_hue), min(255, mean_sat + 2*std_sat), min(255, mean_val + 2*std_val)]
    #make bounds based on av
    av_skin_color_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Convert lists to numpy arrays
    LOWER_THRESHOLD = np.array(lower_bound, dtype=np.uint8)
    UPPER_THRESHOLD = np.array(upper_bound, dtype=np.uint8)
    # Create a binary mask where the skin color is within the threshold
    skinMask = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), LOWER_THRESHOLD, UPPER_THRESHOLD)
    skin = cv2.bitwise_or(img, img, mask=skinMask)
    if mask is not None:
        skin = cv2.bitwise_and(skin, skin, mask=mask)
    skin = np.array(skin, dtype=np.uint8)
    return skin

def extract_masks(image):
    Cm, Ch, Bm, Bh, T = modify_latent.get_masks(image)
    landmarks = lm.get_landmarks(image)
    combined_mask, face, av_skin_color = create_combined_mask(image)
    skin = threshold_face_skin_area(image,av_skin_color,mask=combined_mask)
    return Cm, Ch, Bm, Bh, T, skin


    
    