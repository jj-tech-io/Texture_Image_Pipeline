import sys
import os
from pathlib import Path
current_script_dir = Path(__file__).resolve()
# Get the parent directory (one level up)
parent_dir = current_script_dir.parent
sys.path.append(str(parent_dir))
import importlib
import sys
import time
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageTk
import CONFIG
from pathlib import Path

import torch_face
from torch_face import face_part_segmentation as fps

import numpy as np
import dense_lm
from dense_lm import segmentation, morph
import onnx_inference
from onnx_inference import autoencoder
from onnx_inference import modify_latent
import dense_lm
from dense_lm.morph import morph_images, load_images
from dense_lm.segmentation import extract_masks
importlib.reload(dense_lm)
importlib.reload(CONFIG)

class SkinParameterAdjustmentApp:
    no_background_image = None
    def __init__(self,target_texture_path, example_texture_path):
        self.example_texture_path = example_texture_path
        self.target_texture_path = target_texture_path
        self.original_image = None
        self.modified_image = None
        self.original_label = None
        self.modified_label = None
        self.temp = None
        self.DIM = 256
        self.SAVE_DIM = 4096
        self.WIDTH = self.DIM
        self.HEIGHT = self.DIM
        self.warp_example_var = False
        self.load_images(self.DIM, self.DIM)

        self.init_app()

    def init_app(self):
        self.root = tk.Tk()
        self.root.configure(background='black')
        self.root["bg"] = "black"
        self.root.title("Interactive Skin Parameter Adjustment")
        self.create_gui()

    def load_images(self, width, height):
        try:
            if self.warp_example_var:
                warped_example_image, original_image = morph_images(self.example_texture_path, self.target_texture_path, width, height)
            else:
                warped_example_image, original_image = load_images(self.example_texture_path, self.target_texture_path, width, height)
            original_image = cv2.imread(self.target_texture_path)
            blush_mask = cv2.imread(self.example_texture_path)
            # warped_example_image = cv2.cvtColor(warped_example_image, cv2.COLOR_BGR2RGB)
            # original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

            torch_skin = fps.FacePartSegmentation()
            mask, image_skin = torch_skin.get_skin(original_image)
            image_skin = cv2.resize(image_skin, (width, height),interpolation=cv2.INTER_LANCZOS4)
            # warped_example_image[mask == 0] = 0
            Cm, Ch, Bm, Bh, T = extract_masks(warped_example_image)
            self.temp = warped_example_image
            self.skin = np.where(image_skin == 0, 0, warped_example_image)
            self.original_image = original_image
            self.cm_blend = Cm
            self.ch_blend = Ch
            self.bm_blend = Bm
            self.bh_blend = Bh
            self.t_blend = T
            self.skin = image_skin
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
    def start_save_4k_image(self):
        # Show the progress bar
        self.progress_bar.pack()  # Using pack to show
        self.progress_bar['value'] = 0
        self.root.update_idletasks()

        # Start the save operation in a separate thread
        threading.Thread(target=self.save_4k_image).start()
    def save_4k_image(self):
        self.WIDTH = self.SAVE_DIM
        self.HEIGHT = self.SAVE_DIM
        self.load_images(self.WIDTH, self.HEIGHT)
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
        self.WIDTH = self.DIM
        self.HEIGHT = self.DIM
        self.load_images(self.DIM, self.DIM)
    
    def update_plot(self, display=True):
        parameter_maps = self.parameter_maps_original.copy()
        age_coef = self.age_coef_slider.get()
        scale_c_m = self.cm_slider.get()
        bias_c_m = self.cm_bias_slider.get()
        scale_c_h = self.ch_slider.get()
        bias_c_h = self.ch_bias_slider.get()
        scale_b_m = self.bm_slider.get()
        bias_b_m = self.bm_bias_slider.get()
        scale_b_h = self.bh_slider.get()
        bias_b_h = self.bh_bias_slider.get()
        scale_t = self.t_slider.get()
        bias_t = self.t_bias_slider.get()
        cm_blend_slider = self.cm_blend_slider.get()
        ch_blend_slider = self.ch_blend_slider.get()
        bh_blend_slider = self.bh_blend_slider.get()
        bm_blend_slider = self.bm_blend_slider.get()
        t_blend_slider = self.t_blend_slider.get()
        parameter_maps[:, 0] = modify_latent.age_mel(parameter_maps[:, 0], age_coef)
        parameter_maps[:, 1] = modify_latent.age_hem(parameter_maps[:, 1], age_coef)
        parameter_maps[:, 0] = scale_c_m * parameter_maps[:, 0] + bias_c_m
        parameter_maps[:, 1] = scale_c_h * parameter_maps[:, 1] + bias_c_h
        parameter_maps[:, 2] = scale_b_m * parameter_maps[:, 2] + bias_b_m
        parameter_maps[:, 3] = scale_b_h * parameter_maps[:, 3] + bias_b_h
        parameter_maps[:, 4] = scale_t * parameter_maps[:, 4] + bias_t
        cm_new =  (cm_blend_slider * self.cm_blend.reshape(-1)) + (1 - cm_blend_slider) * parameter_maps[:, 0]
        parameter_maps[:, 0] = cm_new
        ch_new = (cm_blend_slider * self.ch_blend.reshape(-1)) + (1 - cm_blend_slider) * parameter_maps[:, 1]
        parameter_maps[:, 1] = ch_new
        bm_new = (cm_blend_slider * self.bm_blend.reshape(-1)) + (1 - cm_blend_slider) * parameter_maps[:, 2]
        parameter_maps[:, 2] = bm_new
        bh_new = (bh_blend_slider * self.bh_blend.reshape(-1)) + (1 - bh_blend_slider) * parameter_maps[:, 3]
        parameter_maps[:, 3] = bh_new
        t_new = (cm_blend_slider * self.t_blend.reshape(-1)) + (1 - cm_blend_slider) * parameter_maps[:, 4]
        parameter_maps[:, 4] = t_new
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
        original_photo = ImageTk.PhotoImage(original_pil.resize((self.DIM, self.DIM)))
        modified_photo = ImageTk.PhotoImage(modified_pil.resize((self.DIM, self.DIM)))
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
        self.WIDTH = self.DIM
        self.HEIGHT = self.DIM
        self.load_images(self.DIM, self.DIM)
        self.update_plot()  # This is just a placeholder for whatever update you need to do
        self.update_images(self.original_image, self.modified_image)

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
    def apply_mask(self):
        print("apply mask")
        print(f"isBackgroundToggled: {self.isBackgroundToggled.get()}")

        if self.isBackgroundToggled.get():
            # Apply the mask
            self.modified_image = self.original_image.copy()
            self.temp = self.modified_image.copy()
            self.temp[self.skin == 0] = 0
            self.modified_image = self.temp
        else:
            # Remove the mask by reverting to the original image
            self.modified_image = self.original_image.copy()

        self.update_images(self.original_image, self.modified_image)

    def create_slider(self, parent, label, from_, to, resolution, default_value):
        # Create the label for the slider
        label = ttk.Label(parent, text=label, background='#4D4D4D', foreground='white')
        label.grid(sticky='ew')  # Use grid layout

        # Create the slider
        slider = tk.Scale(parent, from_=from_, to=to, orient=tk.HORIZONTAL, resolution=resolution, 
                        bg='#4D4D4D', fg='white', troughcolor='#333333', sliderrelief='flat', 
                        highlightbackground='#4D4D4D', highlightthickness=0)
        slider.set(default_value)
        slider.config(length=200, width=10)  # Adjusted size
        slider.grid(sticky='ew')  # Use grid layout
        #add padding to the slider
        slider.grid(padx=10, pady=0)

        return slider
    
    def toggle_warp_example(self):
        self.warp_example_var = not self.warp_example_var
        if self.warp_example_var:
            self.warp_status_label.config(text="Warp applied")
        else:
            self.warp_status_label.config(text="Warp not applied")
        self.load_new()
        
    def create_gui(self):
        w = self.root.winfo_screenwidth()
        h = self.root.winfo_screenheight()
        self.root.geometry(f"{w//2}x{h}")
        self.root.geometry("+0+0")
        self.root.resizable(True, True)
        self.root.configure(bg='#4D4D4D') 
        style = ttk.Style()
        style.configure('TFrame', background='#4D4D4D')
        style = ttk.Style(self.root)
        style.theme_use('clam')
        # Configure the background color of the TFrame to be dark grey
        style.configure('TFrame', background='#4D4D4D')
        style.configure("TScale", background='#4D4D4D', foreground='white', troughcolor='#4D4D4D')
        # Configure the style for the Button to match the dark theme
        style.configure('TButton', background='#4D4D4D', foreground='white', borderwidth=1)
        style.map('TButton', background=[('active', '#666666')])
        # Configure the style for the Checkbutton to match the dark theme
        style.configure('TCheckbutton', background='#4D4D4D', foreground='white')
        style.map('TCheckbutton', background=[('active', '#666666')])
        # Configure the TScale style
        style.configure("Vertical.TScale", background='#4D4D4D', troughcolor='#333333', sliderrelief='flat')
        style.map('Vertical.TScale', background=[('active', '#4D4D4D'), ('!disabled', '#4D4D4D')], 
            troughcolor=[('active', '#333333'), ('!disabled', '#333333')],
            slider=[('active', '#666666'), ('!disabled', '#666666')])
        # Frames setup
        # Frames setup with grid layout for efficient space usage
        self.frame_sliders = ttk.Frame(self.root, style='TFrame')
        self.frame_sliders.grid(row=0, column=0, sticky='nsew')
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self.frame_buttons = ttk.Frame(self.root, style='TFrame')
        self.frame_buttons.grid(row=1, column=0, sticky='ew')

        self.frame_images = ttk.Frame(self.root, style='TFrame')
        self.frame_images.grid(row=2, column=0, sticky='nsew')

        # Configure the frame_sliders grid to fill and expand
        self.frame_sliders.grid_columnconfigure(0, weight=1)
        self.frame_sliders.grid_columnconfigure(1, weight=1)
        self.frame_sliders_col1 = ttk.Frame(self.frame_sliders, style='TFrame')
        self.frame_sliders_col1.grid(row=0, column=0, sticky='nsew')
        self.frame_sliders_col2 = ttk.Frame(self.frame_sliders, style='TFrame')
        self.frame_sliders_col2.grid(row=0, column=1, sticky='nsew')
        self.frame_sliders_col3 = ttk.Frame(self.frame_sliders, style='TFrame')
        self.frame_sliders_col3.grid(row=0, column=2, sticky='nsew')

        # After creating the slider, set its style to the custom one
        self.age_coef_slider = self.create_slider(self.frame_sliders_col1, "Age(decades):", 0, 10, 0.1, 0.0)
        self.cm_blend_slider = self.create_slider(self.frame_sliders_col1, "Cm blend:", 0, 1, 0.01, 0.0)
        self.ch_blend_slider = self.create_slider(self.frame_sliders_col1, "Ch blend:", 0, 1, 0.01, 0.0)
        self.bm_blend_slider = self.create_slider(self.frame_sliders_col1, "Bm blend:", 0, 1, 0.01, 0.0)
        self.bh_blend_slider = self.create_slider(self.frame_sliders_col1, "Bh blend:", 0, 1, 0.01, 0.0)
        self.t_blend_slider = self.create_slider(self.frame_sliders_col1, "T blend:", 0, 1, 0.01, 0.0)

        # Sliders in the right column (All other sliders)
        self.cm_slider = self.create_slider(self.frame_sliders_col2, "Cm:", 0, 2, 0.1, 1)
        self.cm_bias_slider = self.create_slider(self.frame_sliders_col3, "Cm bias:", -0.1, 0.1, 0.01, 0.0)
        self.ch_slider = self.create_slider(self.frame_sliders_col2, "Ch:", 0, 2, 0.1, 1)
        self.ch_bias_slider = self.create_slider(self.frame_sliders_col3, "Ch bias:", -0.1, 0.1, 0.01, 0.0)
        self.bm_slider = self.create_slider(self.frame_sliders_col2, "Bm:", 0, 2, 0.1, 1)
        self.bm_bias_slider = self.create_slider(self.frame_sliders_col3, "Bm bias:", -0.1, 0.1, 0.01, 0.0)
        self.bh_slider = self.create_slider(self.frame_sliders_col2, "Bh:", 0, 2, 0.1, 1)
        self.bh_bias_slider = self.create_slider(self.frame_sliders_col3, "Bh bias:", -0.1, 0.1, 0.01, 0.0)
        self.t_slider = self.create_slider(self.frame_sliders_col2, "T:", 0, 2, 0.1, 1)
        self.t_bias_slider = self.create_slider(self.frame_sliders_col3, "T bias:", -0.1, 0.1, 0.01, 0.0)

        # Step 1: Create a BooleanVar variable to track the state of the checkbox
        self.isBackgroundToggled = tk.BooleanVar()
        self.isBackgroundToggled.set(False)

        # Step 2: Create the Checkbutton widget with the command option
        self.checkbox_3d = ttk.Checkbutton(self.frame_buttons, text="Toggle Background", variable=self.isBackgroundToggled, command=self.apply_mask)
        self.warp_example_var = tk.BooleanVar(value=False)
        self.warp_example_checkbox = ttk.Checkbutton(self.frame_buttons, text="Warp Example", variable=self.warp_example_var, command=self.toggle_warp_example)
        self.warp_example_checkbox.pack(side=tk.LEFT, padx=5, pady=5)
        self.warp_status_label = ttk.Label(self.frame_buttons, text="Warp not applied", background='#4D4D4D', foreground='white')
        self.warp_status_label.pack(side=tk.LEFT, padx=5, pady=5)
        # Step 3: Place the checkbox in the GUI
        self.checkbox_3d.pack(side=tk.LEFT, padx=5, pady=5)
        self.save_button = ttk.Button(self.frame_buttons, text="Save 4K Image", command=self.save_4k_image)
        self.save_button.pack(side=tk.RIGHT, padx=5, pady=5)
        self.age_coef_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot())
        self.cm_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot())
        self.cm_bias_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot())
        self.ch_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot())
        self.ch_bias_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot())
        self.bm_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot())
        self.bm_bias_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot())
        self.bh_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot())
        self.bh_bias_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot())
        self.t_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot())
        self.t_bias_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot())
        self.cm_blend_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot())
        self.ch_blend_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot())
        self.bh_blend_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot())
        self.bm_blend_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot())
        self.t_blend_slider.bind("<ButtonRelease-1>", lambda event: self.update_plot())
        self.save_button.bind("<ButtonRelease-1>", lambda event: self.save_4k_image())
        #add labels for original and modified images
        self.original_label = ttk.Label(self.frame_images, text="Original Image", background='#4D4D4D', foreground='white')
        self.original_label.pack(side=tk.LEFT, padx=10, pady=10)
        self.modified_label = ttk.Label(self.frame_images, text="Modified Image", background='#4D4D4D', foreground='white')
        self.modified_label.pack(side=tk.LEFT, padx=10, pady=10)
        load_image_button = ttk.Button(self.frame_buttons, text="Load New Image", command=self.load_new_image)
        load_image_button.pack(side=tk.LEFT, padx=5, pady=5)
        load_example_image_button = ttk.Button(self.frame_buttons, text="Load Example Image", command=self.load_example_image)
        load_example_image_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.root.geometry(f"{w//2}x{h}+0+0")
        self.update_plot()

   
    def run(self):
        self.root.mainloop()

