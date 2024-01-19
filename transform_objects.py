import sys
from pathlib import Path
current_script_dir = Path(__file__).resolve()
# Get the parent directory (one level up)
parent_dir = current_script_dir.parent
sys.path.append(str(parent_dir))
print(f"parent_dir: {parent_dir}")
import importlib
import sys
import time
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageTk
import CONFIG
from pathlib import Path

import torch_face
from torch_face import face_part_segmentation as fps
importlib.reload(CONFIG)
import numpy as np
import dense_lm
from dense_lm import segmentation, morph
importlib.reload(dense_lm)
import onnx_inference
from onnx_inference import autoencoder
from onnx_inference import modify_latent
import dense_lm
#reload(dense_lm)
importlib.reload(dense_lm)
from dense_lm.morph import morph_images
from dense_lm.segmentation import extract_masks

class SkinParameterAdjustmentApp:
    def __init__(self,target_texture_path, example_texture_path):
        self.example_texture_path = example_texture_path
        self.target_texture_path = target_texture_path
        self.original_image = None
        self.modified_image = None
        self.original_label = None
        self.modified_label = None
        self.WIDTH = 512
        self.HEIGHT = 512
        self.load_images(512, 512)
        self.init_app()

    def init_app(self):
        self.root = tk.Tk()
        self.root.configure(background='black')
        self.root["bg"] = "black"
        self.root.title("Interactive Skin Parameter Adjustment")
        self.create_gui()

    def load_images(self, width, height):
        try:
            warped_example_image, original_image = morph_images(self.example_texture_path, self.target_texture_path, width, height)
            Cm, Ch, Bm, Bh, T, skin = extract_masks(warped_example_image)
            torch_skin = fps.FacePartSegmentation()
            mask, image_skin = torch_skin.get_skin(original_image)
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(skin)
            ax[1].imshow(mask)
            ax[2].imshow(image_skin)
            plt.show()
            self.original_image = original_image
            self.mel_aged = Cm
            self.oxy_aged = Bh
            self.skin = skin
        except Exception as e:
            print(f"Error: could not load images: {e}")
            sys.exit()
        self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        self.original_image = cv2.resize(self.original_image, (self.WIDTH, self.HEIGHT),interpolation=cv2.INTER_LANCZOS4)
        self.modified_image = self.original_image.copy()
        self.parameter_maps = autoencoder.encode(self.original_image.reshape(-1, 3) / 255.0)
        self.parameter_maps_original = self.parameter_maps.copy()

    def create_slider1(self, parent, label, from_, to, resolution, default_value):
        frame = ttk.Frame(parent)
        label = ttk.Label(frame, text=label)
        label.pack(side=tk.LEFT)
        slider = tk.Scale(frame, from_=from_, to=to, orient='horizontal', length=200, resolution=resolution)
        slider.set(default_value)
        slider.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        frame.pack()
        return slider
    def create_slider(self, parent, label_text, from_, to, resolution, default_value):
        frame = ttk.Frame(parent)
        frame.grid(sticky='ew')
        parent.grid_columnconfigure(0, weight=1)  # This makes the frame expand to fill the grid cell
        label = ttk.Label(frame, text=label_text)
        label.grid(row=0, column=0, sticky='w')  # Align label to the left (west)
        slider = tk.Scale(frame, from_=from_, to=to, orient='horizontal', length=200, resolution=resolution)
        slider.set(default_value)
        slider.grid(row=0, column=1, sticky='ew')  # Align slider to the right, expand horizontally
        frame.grid_columnconfigure(1, weight=1)  # This allows the slider to expand
        return slider
    def display_4k_image(self):
        self.WIDTH = 4096
        self.HEIGHT = 4096
        self.load_images(4096, 4096)
        self.update_plot(display=False)
        try:
            display_path = r"C:\Users\joeli\Dropbox\Code\Python Projects\Texture_Image_Pipeline\output"
            display_path += f"\{time.strftime('%m_%d_%H_%M')}.png"
            #print cwd
            print(f"Saving image to {display_path}")
            cv2.imwrite(display_path, cv2.cvtColor(self.modified_image, cv2.COLOR_BGR2RGB))
            print(f"displayd image to {display_path}")
        except:
            print(f"Error: could not display image to {display_path}")
        self.WIDTH = 512
        self.HEIGHT = 512
        self.load_images(512, 512)
    
    def update_plot(self, display=True):
        parameter_maps = self.parameter_maps_original.copy()
        age_coef = self.age_coef_slider.get()
        scale_c_m = self.cm_slider.get()
        scale_c_h = self.ch_slider.get()
        scale_b_m = self.bm_slider.get()
        scale_b_h = self.bh_slider.get()
        scale_t = self.t_slider.get()
        cm_mask_slider = self.cm_mask_slider.get()
        bh_mask_slider = self.bh_mask_slider.get()
        parameter_maps[:, 0] = modify_latent.age_mel(parameter_maps[:, 0], age_coef)
        parameter_maps[:, 1] = modify_latent.age_hem(parameter_maps[:, 1], age_coef)
        parameter_maps[:, 0] = scale_c_m * parameter_maps[:, 0]
        parameter_maps[:, 1] = scale_c_h * parameter_maps[:, 1]
        parameter_maps[:, 2] = scale_b_m * parameter_maps[:, 2]
        parameter_maps[:, 3] = scale_b_h * parameter_maps[:, 3]
        parameter_maps[:, 4] = scale_t * parameter_maps[:, 4]
        cm_new =  (cm_mask_slider * self.mel_aged.reshape(-1)) + (1 - cm_mask_slider) * parameter_maps[:, 0]
        parameter_maps[:, 0] = cm_new
        bh_new = (bh_mask_slider * self.oxy_aged.reshape(-1)) + (1 - bh_mask_slider) * parameter_maps[:, 3]
        parameter_maps[:, 3] = bh_new
        recovered = autoencoder.decode(parameter_maps).reshape((self.WIDTH, self.HEIGHT, 3)) * 255
        self.parameter_maps = parameter_maps
        self.modified_image = recovered
        if display:
            self.update_images(self.original_image, self.modified_image)
        return recovered

    def update_images(self, original, modified, display=False):
        if np.max(original) < 1:    
            original *= 255
        if np.max(modified) < 1:
            modified *= 255
        try:
            original_pil = Image.fromarray(np.uint8(original))
            modified_pil = Image.fromarray(np.uint8(modified))
            if display:
                display_path = f"{self.display_name}.png"
                modified_pil.display(display_path)
        except:
            print(f"Error: could not convert original or modified to PIL images")
            sys.exit()
        original_photo = ImageTk.PhotoImage(original_pil.resize((512, 512)))
        modified_photo = ImageTk.PhotoImage(modified_pil.resize((512, 512)))
        if self.original_label is None:
            self.original_label = ttk.Label(self.frame_images, image=original_photo)
            self.original_label.image = original_photo
            self.original_label.pack(side=tk.LEFT, padx=10, pady=10)
        else:
            self.original_label.configure(image=original_photo)
            self.original_label.image = original_photo

        if self.modified_label is None:
            self.modified_label = ttk.Label(self.frame_images, image=modified_photo)
            self.modified_label.image = modified_photo
            self.modified_label.pack(side=tk.LEFT, padx=10, pady=10)
        else:
            self.modified_label.configure(image=modified_photo)
            self.modified_label.image = modified_photo

    def load_new(self):
        self.WIDTH = 512
        self.HEIGHT = 512
        self.load_images(512, 512)
        self.update_plot()  # This is just a placeholder for whatever update you need to do

    def load_new_image(self):
        # Open the file dialog to select an image
        file_path = filedialog.askopenfilename(title="Select an image",
                                               filetypes=(("png files", "*.png"), ("jpeg files", "*.jpg"), ("all files", "*.*")))
        self.target_texture_path = file_path

        if file_path:  # If a file was selected
            print(f"Selected image: {file_path}")
            self.load_new()
    def load_example_image(self):
        # Open the file dialog to select an image
        file_path = filedialog.askopenfilename(title="Select an image",
                                               filetypes=(("png files", "*.png"), ("jpeg files", "*.jpg"), ("all files", "*.*")))
        self.example_texture_path = file_path
        if file_path:  # If a file was selected
            print(f"Selected image: {file_path}")
            self.load_new()

    def create_gui(self):
        self.frame_sliders = ttk.Frame(self.root)
        self.frame_sliders.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.frame_sliders.configure(width=900)
        self.frame_buttons = ttk.Frame(self.root)
        self.frame_buttons.pack(side=tk.BOTTOM, fill=tk.X, expand=False)
        self.frame_images = ttk.Frame(self.root)
        self.frame_images.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.age_coef_slider = self.create_slider(self.frame_sliders, "Age(decades):", 0, 10, 0.1,2.0)
        self.cm_slider = self.create_slider(self.frame_sliders, "Cm:", 0, 2, 0.1, 1)
        self.ch_slider = self.create_slider(self.frame_sliders, "Ch:", 0, 2, 0.1, 1)
        self.bm_slider = self.create_slider(self.frame_sliders, "Bm:", 0, 2, 0.1, 1)
        self.bh_slider = self.create_slider(self.frame_sliders, "Bh:", 0, 2, 0.1, 0.9)
        self.t_slider = self.create_slider(self.frame_sliders, "T:", 0, 2, 0.1, 1)
        self.cm_mask_slider = self.create_slider(self.frame_sliders, "Melanin Mask:", -1, 1, 0.1, 0.4)
        self.bh_mask_slider = self.create_slider(self.frame_sliders, "Oxy-Hb Mask:", -0.2, 0.2, 0.01, 0.0)
        self.display_button = ttk.Button(self.frame_buttons, text="Save 4K Image", command=self.display_4k_image)
        self.display_button.pack(side=tk.RIGHT, padx=5, pady=5)
        self.age_coef_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot())
        self.cm_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot())
        self.ch_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot())
        self.bm_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot())
        self.bh_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot())
        self.t_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot())
        self.cm_mask_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot())
        self.bh_mask_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot())
        self.display_button.bind("<ButtonRelease-1>", lambda event: self.display_4k_image())
        load_image_button = ttk.Button(self.frame_buttons, text="Load New Image", command=self.load_new_image)
        load_image_button.pack(side=tk.LEFT, padx=5, pady=5)
        load_example_image_button = ttk.Button(self.frame_buttons, text="Load Example Image", command=self.load_example_image)
        load_example_image_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.root.resizable(True, True)
        self.root.geometry("1100x900")
        self.root.geometry("+0+0")
        self.update_plot()

    def create_gui1(self):
        self.root.state('zoomed')
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        self.frame_sliders = ttk.Frame(self.root)
        self.frame_sliders.grid(row=0, column=0, sticky="ew", padx=10)
        self.frame_images = ttk.Frame(self.root)
        self.frame_images.grid(row=1, column=0, sticky="nsew")
        self.frame_buttons = ttk.Frame(self.root)
        self.frame_buttons.grid(row=2, column=0, sticky="ew")
        self.frame_sliders.grid_columnconfigure(0, weight=1)
        self.age_coef_slider = self.create_slider(self.frame_sliders, "Age(decades):", 0, 10, 0.1, 2.0)
        self.cm_slider = self.create_slider(self.frame_sliders, "Cm:", 0, 2, 0.1, 1)
        self.ch_slider = self.create_slider(self.frame_sliders, "Ch:", 0, 2, 0.1, 1)
        self.bm_slider = self.create_slider(self.frame_sliders, "Bm:", 0, 2, 0.1, 1)
        self.bh_slider = self.create_slider(self.frame_sliders, "Bh:", 0, 2, 0.1, 0.9)
        self.t_slider = self.create_slider(self.frame_sliders, "T:", 0, 2, 0.1, 1)
        self.cm_mask_slider = self.create_slider(self.frame_sliders, "Melanin Mask:", -1, 1, 0.1, 0.6)
        self.bh_mask_slider = self.create_slider(self.frame_sliders, "Oxy-Hb Mask:", -1, 1, 0.1, 0.1)
        slider_names = ['age_coef', 'cm', 'ch', 'bm', 'bh', 't', 'cm_mask', 'bh_mask']
        for i, name in enumerate(slider_names):
            getattr(self, f"{name}_slider").grid(row=0, column=i, padx=5, pady=5, sticky="ew")
        load_image_button = ttk.Button(self.frame_buttons, text="Load New Image", command=self.load_new_image)
        load_image_button.grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.display_button = ttk.Button(self.frame_buttons, text="Save 4K Image", command=self.display_4k_image)
        self.display_button.grid(row=0, column=1, padx=5, pady=5, sticky="e")
        for name in slider_names:
            getattr(self, f"{name}_slider").bind("<ButtonRelease-1>", lambda event, name=name: self.update_plot(changed_slider=name))
        self.root.geometry("1100x900")
        self.root.geometry("+0+0")
        self.update_plot()
    def run(self):
        self.root.mainloop()