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
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    landmarks = []
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for i in range(0, 468):
                landmark = face_landmarks.landmark[i]
                x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
                landmarks.append((x, y))
    return np.array(landmarks)


def get_extended_landmarks(landmarks, image_shape):
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
    landmarks1_extended = get_extended_landmarks(landmarks1, target.shape)[:-2]
    landmarks2_extended = get_extended_landmarks(landmarks2, source.shape)[:-2]
    delaunay = Delaunay(landmarks1_extended)
    warped_image = target.copy()
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
    landmarks1 = get_landmarks(example_image)
    landmarks2 = get_landmarks(target_image)
    warped_example_image, delaunay, transformation_matrices = warp_image(example_image, target_image,landmarks1, landmarks2)
    return warped_example_image, target_image
if __name__ == '__main__':
    target_image_path = r"C:\Users\joeli\Dropbox\Data\models_4k\light\m32_4k.png"
    source_image_path = r"C:\Users\joeli\Dropbox\Data\face_image_data\facescape\2\models_reg\1_neutral.jpg"
    #switch
    # Read the images into NumPy arrays
    target_image = cv2.imread(target_image_path)
    source_image = cv2.imread(source_image_path)
    # Check if the images are read correctly
    if source_image is None or target_image is None:
        print("Error: could not read one of the images.")
        # Handle the error, for example by exiting the script
        sys.exit()
    # Define the new width and height
    WIDTH = 4096
    HEIGHT = 4096
    target_image = cv2.resize(target_image, (WIDTH, HEIGHT))
    source_image = cv2.resize(source_image, (WIDTH, HEIGHT))
    source_landmarks = get_landmarks(source_image)
    target_landmarks = get_landmarks(target_image)
    warped_source_image, delaunay, transformation_matrices = warp_image(source_image, target_image, source_landmarks, target_landmarks)
    warped_landmarks = get_landmarks(warped_source_image)
    plt.imshow(cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB))
    plt.title("Source image")
    plt.show()
    plt.imshow(cv2.cvtColor(warped_source_image, cv2.COLOR_BGR2RGB))
    plt.title("Warped image")
    plt.show()

    plt.imshow(cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB))
    plt.title("Target image")
    plt.show()

    mel = r"masks\Cm.png"
    mel = cv2.imread(mel, cv2.IMREAD_GRAYSCALE)
    mel = cv2.resize(mel, (WIDTH, HEIGHT))
    plt.imshow(mel)
    plt.title("Mel")
    plt.show()
    #transform mel
    mel_warped = apply_transformations_to_single_channel_image(mel, transformation_matrices)
    plt.imshow(mel_warped)
    plt.title("Mel warped")
    plt.show()

    oxy = r"masks\Bh.png"
    oxy = cv2.imread(oxy, cv2.IMREAD_GRAYSCALE)
    oxy = cv2.resize(oxy, (WIDTH, HEIGHT))
    plt.imshow(oxy)
    plt.title("Oxy")
    plt.show()
    #transform oxy
    oxy_warped = apply_transformations_to_single_channel_image(oxy, transformation_matrices)
    plt.imshow(oxy_warped)
    plt.title("Oxy warped")
    plt.show()
    
