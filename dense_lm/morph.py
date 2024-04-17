import sys
from pathlib import Path
current_script_dir = Path(__file__).resolve()
# Get the parent directory (one level up)
parent_dir = current_script_dir.parent
# Get the root directory (two levels up or more if needed)
# Add both directories to sys.path
sys.path.append(str(parent_dir))
from scipy.spatial import Delaunay
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

def get_landmarks(image):
    print("Getting landmarks")
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    landmarks = []
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for i in range(0, 468):
                print(f"landmark {i}: {face_landmarks.landmark[i]}")
                landmark = face_landmarks.landmark[i]
                x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                landmarks.append((x, y))
    if len(landmarks) == 0:
        print("No landmarks found")
        return None
    return np.array(landmarks)


def get_extended_landmarks(landmarks, image_shape):
    if isinstance(landmarks, list):
        landmarks = np.array(landmarks, dtype=np.float32)
    if landmarks.size == 0:
        raise ValueError("The landmarks array is empty.")
    
    # Ensure the landmarks are in the correct format
    landmarks = np.array(landmarks, dtype=np.float32)
    
    # Calculate the convex hull
    hull = cv2.convexHull(landmarks)
    offset = 0
    boundary_points = np.array([
        [offset, offset],
        [image_shape[1] // 2, offset],
        [image_shape[1] - 1 - offset, offset],
        [image_shape[1] - 1 - offset, image_shape[0] // 2],
        [image_shape[1] - 1 - offset, image_shape[0] - 1 - offset],
        [image_shape[1] // 2, image_shape[0] - 1 - offset],
        [offset, image_shape[0] - 1 - offset],
        [offset, image_shape[0] // 2]
    ])
    extended_landmarks = np.vstack((landmarks, boundary_points))
    extended_landmarks = np.vstack((extended_landmarks, hull.squeeze()))
    return extended_landmarks

def warp_image(target, source, landmarks1, landmarks2):
    if isinstance(landmarks1, list):
        landmarks1 = np.array(landmarks1, dtype=np.float32)
    if isinstance(landmarks2, list):
        landmarks2 = np.array(landmarks2, dtype=np.float32)
    print(f"landmarks1: {landmarks1.shape}, landmarks2: {landmarks2.shape}")
    delaunay = Delaunay(landmarks1)
    warped_image = target.copy_source_image, delaunay, transformation_matrices = warp_image(source_image, target_image, landmarks1, landmarks2)

    transformation_matrices = []
    # Iterate through each triangle
    for simplex in delaunay.simplices[:-1]:
        if np.any(simplex >= len(landmarks1)) or np.any(simplex >= len(landmarks2)):
            continue
        src_triangle = landmarks1_extended[simplex]
        dest_triangle = landmarks2_extended[simplex]
        src_rect = cv2.boundingRect(np.float32([src_triangle]))
        dest_rect = cv2.boundingRect(np.float32([dest_triangle]))
        src_cropped_triangle = target[src_rect[1]:src_rect[1] + src_rect[3], src_rect[0]:src_rect[0] + src_rect[2]]
        dest_cropped_triangle = np.zeros((dest_rect[3], dest_rect[2], 3), dtype=np.float32)
        src_triangle_adjusted = src_triangle - (src_rect[0], src_rect[1])
        dest_triangle_adjusted = dest_triangle - (dest_rect[0], dest_rect[1])
        matrix = cv2.getAffineTransform(np.float32(src_triangle_adjusted), np.float32(dest_triangle_adjusted))
        warped_triangle = cv2.warpAffine(src_cropped_triangle, matrix, (dest_rect[2], dest_rect[3]))
        mask = np.zeros((dest_rect[3], dest_rect[2]), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.int32(dest_triangle_adjusted), (1, 1, 1), 16, 0)
        warped_image[dest_rect[1]:dest_rect[1] + dest_rect[3], dest_rect[0]:dest_rect[0] + dest_rect[2]] = \
            warped_image[dest_rect[1]:dest_rect[1] + dest_rect[3], dest_rect[0]:dest_rect[0] + dest_rect[2]] * (1 - mask[:, :, None]) \
            + warped_triangle * mask[:, :, None]
        transformation_matrices.append((matrix, src_triangle, dest_triangle))
    return warped_image.astype(np.uint8), delaunay, transformation_matrices




def apply_transformations_to_single_channel_image(original_image, transformation_matrices):
    warped_image = np.zeros(original_image.shape, dtype=np.uint8)
    for matrix, src_triangle, dest_triangle in transformation_matrices:
        dest_rect = cv2.boundingRect(np.float32([dest_triangle]))
        cropped_region = original_image[dest_rect[1]:dest_rect[1] + dest_rect[3], dest_rect[0]:dest_rect[0] + dest_rect[2]]
        dest_triangle_adjusted = dest_triangle - (dest_rect[0], dest_rect[1])
        # Compute the affine transformation
        matrix_inv = cv2.invertAffineTransform(matrix)
        # Warp the corresponding region to the shape of the source triangle
        warped_region = cv2.warpAffine(cropped_region, matrix_inv, (dest_rect[2], dest_rect[3]), None, 
                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        # Mask for the source triangle
        mask = np.zeros((dest_rect[3], dest_rect[2]), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.int32(dest_triangle_adjusted), 1, 16)
        # Place the warped region in the warped image
        warped_image[dest_rect[1]:dest_rect[1] + dest_rect[3], dest_rect[0]:dest_rect[0] + dest_rect[2]] = \
            warped_image[dest_rect[1]:dest_rect[1] + dest_rect[3], dest_rect[0]:dest_rect[0] + dest_rect[2]] * (1 - mask) + \
            warped_region * mask
    return warped_image

def morph_images(example_image_path, target_image_path,width, height):
    print(f"morphing {example_image_path} to {target_image_path}")
    
    example_image_path = str(example_image_path)
    target_image_path = str(target_image_path)
    try:
        target_image = cv2.imread(target_image_path)
        target_image = cv2.resize(target_image, (width, height),interpolation=cv2.INTER_LANCZOS4)
    except:
        print(f"Error: could not read source image {example_image_path}")
        sys.exit()
    try:
        example_image = cv2.imread(example_image_path)
        example_image = cv2.resize(example_image, (width, height),interpolation=cv2.INTER_LANCZOS4)
    except:
        print(f"Error: could not read source image {example_image_path}")
        sys.exit()
    # landmarks1 = get_landmarks(example_image)
    # landmarks2 = get_landmarks(target_image)
    # warped_example_image, delaunay, transformation_matrices = warp_image(example_image, target_image,landmarks1, landmarks2)
    # return warped_example_image, target_image
    return example_image, target_image
def load_images(example_image_path, target_image_path,width, height):
    print(f"morphing {example_image_path} to {target_image_path}")
    
    example_image_path = str(example_image_path)
    target_image_path = str(target_image_path)
    try:
        target_image = cv2.imread(target_image_path)
        target_image = cv2.resize(target_image, (width, height),interpolation=cv2.INTER_LANCZOS4)
    except:
        print(f"Error: could not read source image {example_image_path}")
        sys.exit()
    try:
        example_image = cv2.imread(example_image_path)
        example_image = cv2.resize(example_image, (width, height),interpolation=cv2.INTER_LANCZOS4)
    except:
        print(f"Error: could not read source image {example_image_path}")
        sys.exit()
    return example_image, target_image
# if __name__ == '__main__':

#     # ma "D:\Unity Projects\flame\Assets\33346.png"
#     # mb ""D:\Unity Projects\flame\Assets\m46_4k.png""
#     # target_image_path = r"C:\Users\joeli\Dropbox\Data\models_4k\light\m32_4k.png"
#     # source_image_path = r"C:\Users\joeli\Dropbox\Data\face_image_data\facescape\2\models_reg\1_neutral.jpg"
   
    
#     target_image_path = r"D:\Unity Projects\flame\Assets\33346.png"
#     source_image_path = r"D:\Unity Projects\flame\Assets\m46_4k.png"
#     target_image = cv2.imread(target_image_path)
#     source_image = cv2.imread(source_image_path)
    
#     WIDTH = 1024
#     HEIGHT = 1024
#     # Original sizes
#     original_target_width, original_target_height = target_image.shape[1], target_image.shape[0]
#     original_source_width, original_source_height = source_image.shape[1], source_image.shape[0]

#     # Calculate scaling factors
#     scale_factor_target_width = WIDTH / original_target_width 
#     scale_factor_target_height = HEIGHT / original_target_height
#     scale_factor_source_width = WIDTH / original_source_width
#     scale_factor_source_height = HEIGHT / original_source_height
#     print(f"scale_factor_target_width: {scale_factor_target_width}, scale_factor_target_height: {scale_factor_target_height}")
#     print(f"scale_factor_source_width: {scale_factor_source_width}, scale_factor_source_height: {scale_factor_source_height}")
#     target_image = cv2.resize(target_image, (WIDTH, HEIGHT))
#     source_image = cv2.resize(source_image, (WIDTH, HEIGHT))

#     if source_image is None or target_image is None:
#         print("Error: could not read one of the images.")
#         # Handle the error, for example by exiting the script
#         sys.exit()

#     # load landmarks
#     #m1 is target
#     #m2 is source
#     # i,u1,v1,u2,v2
#     # 0,0.4945696,0.5239752,0.5040277,0.5081507
#     # 1,0.6577965,0.5985687,0.6957955,0.6291448
#     # 2,0.573979,0.6166188,0.5784552,0.6308378 ...
#     target_lm_path = r"D:\Unity Projects\flame\Assets\points1.txt"
#     source_lm_path = r"D:\Unity Projects\flame\Assets\points2.txt"
#     #read each row 
#     target_landmarks = []
#     source_landmarks = []
#     WIDTH = 1024
#     HEIGHT = 1024
#     count = 0
#     with open(target_lm_path, "r") as file:
#         for line in file:
#             if line.startswith("#") or line.startswith("m") or line.startswith("i"):
#                 continue
#             if line.startswith("i"):
#                 continue
#             # print(line)
#             i,u1,v1 = line.split(",")
#             t_lm = (int(float(u1)*WIDTH), int(float(v1)*HEIGHT))
#             if count < 2:
#                 print(f"t_lm: {t_lm}")
#             count += 1
#             target_landmarks.append(t_lm)
#     with open(source_lm_path, "r") as file:
#         for line in file:
#             if line.startswith("#"):
#                 continue
#             if line.startswith("i"):
#                 continue
#             # print(line)
#             i,u2,v2 = line.split(",")
#             s_lm = (int(float(u2)*WIDTH), int(float(v2)*HEIGHT))
#             if count < 2:
#                 print(f"s_lm: {s_lm}")
#             count += 1
#             source_landmarks.append(s_lm)

 
#     target_landmarks2 = get_landmarks(target_image)
#     source_landmarks2 = get_landmarks(source_image)

#     if target_landmarks2 is None or source_landmarks2 is None:
#         print("Error: could not get landmarks for one of the images.")
#         # Handle the error, for example by exiting the script
#         sys.exit()
#     #    #combine the landmarks
#     # target_lm_combined = np.vstack((target_landmarks, target_landmarks2))
#     # source_lm_combined = np.vstack((source_landmarks, source_landmarks2))
#     print(f"target_landmarks: {np.asarray(target_landmarks).shape}, target_landmarks2: {np.asarray(target_landmarks2).shape}")
#     print(f"source_landmarks: {np.asarray(source_landmarks).shape}, source_landmarks2: {np.asarray(source_landmarks2).shape}")
#     # print lm 2
#     print(f"target_landmarks2: {target_landmarks2.shape}, source_landmarks2: {source_landmarks2.shape}")
    
#     # print(f"target_lm_combined: {target_lm_combined.shape}, source_lm_combined: {source_lm_combined.shape}")
#     # print(f"target_landmarks_combined: {target_lm_combined.shape}, source_landmarks_combined: {source_lm_combined.shape}")
#     # #combined shape
#     # print(f"target_landmarks_combined: {target_lm_combined.shape}, source_landmarks_combined: {source_lm_combined.shape}")
#     annotated_target_image = target_image.copy()
#     annotated_source_image = source_image.copy()
#     for i in range(len(source_landmarks2)):
#         x_source, y_source = source_landmarks2[i]
#         x_target, y_target = target_landmarks2[i]

#         lm_str = f"{i} ({x_source},{y_source}) ({x_target},{y_target})"
#         # cv2.putText(annotated_source_image, str(lm_str), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


#         #red for target landmarks 
#         cv2.circle( annotated_target_image, (x_target, y_target), 5, (0, 0, 255), -1)
#         cv2.circle(annotated_source_image, (x_source, y_source), 5, (0, 255, 0), -1)
#         # print(f"landmark {i}: {lm_str}")
#         # cv2.putText(annotated_target_image, str(lm_str), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
#     # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
#     # ax[0].imshow(cv2.cvtColor(annotated_source_image, cv2.COLOR_BGR2RGB))
#     # ax[0].set_title("Source image")
#     # ax[1].imshow(cv2.cvtColor(annotated_target_image, cv2.COLOR_BGR2RGB))
#     # ax[1].set_title("Target image")
#     # plt.show()
#     warped_source_image, delaunay, transformation_matrices = warp_image(target_image, source_image, target_landmarks2, source_landmarks2)
#     fig, ax = plt.subplots(1, 2, figsize=(10, 5))
#     ax[0].imshow(warped_source_image)
#     # original example image
#     ax[1].imshow(source_image)
#     ax[1].set_title("Original example image")
#     plt.show()
    

    
