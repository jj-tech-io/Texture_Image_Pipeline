import os
CWD = os.getcwd()
current_script_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(current_script_path)
from .onnx_ae import ONNXAutoencoder
print(f"initializing onnx_ae from {current_script_path}")
# Paths to your ONNX model files
# encoder_model_path = 'no_duplicates_75_2_mask_encoder.onnx'
# decoder_model_path = 'no_duplicates_75_2_mask_decoder.onnx'
encoder_model_path ='small_batch_size_encoder.onnx'
decoder_model_path ='small_batch_size_decoder.onnx'

# C:\Users\joeli\Dropbox\Code\Python Projects\AE_2024_01 - Copy\weights and biases\bm_decoder.onnx
# decoder_model_path = r"C:\Users\joeli\Dropbox\Code\Python Projects\Texture_Image_Pipeline\onnx_inference\april_3_2024_decoder.onnx"
# encoder_model_path = r"C:\Users\joeli\Dropbox\Code\Python Projects\Texture_Image_Pipeline\onnx_inference\april_3_2024_encoder.onnx"
print("Current Working Directory:", current_script_path)
encoder_model_path = os.path.join(current_script_path, encoder_model_path)
decoder_model_path = os.path.join(current_script_path, decoder_model_path)
# Create a shared instance of ONNXAutoencoder
autoencoder = ONNXAutoencoder(encoder_model_path, decoder_model_path)
#reset working directory
os.chdir(CWD)