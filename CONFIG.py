import pathlib 
import os
RUN_LOCAL = True
ENCODER_PATH = None
DECODER_PATH = None
CWD = pathlib.Path.cwd()

print("Current Working Directory:", CWD)

DECODER_PATH = r"onnx_inference\no_duplicates_75_2_mask_decoder.onnx"
ENCODER_PATH = r"onnx_inference\no_duplicates_75_2_mask_encoder.onnx"
ENCODER_PATH = r"onnx_inference\small_batch_size_encoder.onnx"
DECODER_PATH = r"onnx_inference\small_batch_size_decoder.onnx"
# decoder_model_path = r"C:\Users\joeli\Dropbox\Code\Python Projects\Texture_Image_Pipeline\onnx_inference\april_3_2024_decoder.onnx"
# encoder_model_path = r"C:\Users\joeli\Dropbox\Code\Python Projects\Texture_Image_Pipeline\onnx_inference\april_3_2024_encoder.onnx"
example_texture_path = r"C:\Users\joeli\Dropbox\Code\Python Projects\Texture_Image_Pipeline\fitzpatrick\m32_4k.png"
example_texture_path = "txyz2mh_32_pc.jpg"

target_texture_path = r"D:\Desktop\man mh\FaceColor_MAIN.PNG"
target_texture_path = "FaceColor_MAIN.PNG"

test_image_path = r"C:\Desktop\PXL_20240202_063955440.jpg"


DECODER_PATH = pathlib.Path(CWD, DECODER_PATH)
ENCODER_PATH = pathlib.Path(CWD, ENCODER_PATH)
print(DECODER_PATH)
print(ENCODER_PATH)
