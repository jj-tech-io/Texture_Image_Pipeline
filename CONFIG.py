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
DECODER_PATH = pathlib.Path(CWD, DECODER_PATH)
ENCODER_PATH = pathlib.Path(CWD, ENCODER_PATH)
print(DECODER_PATH)
print(ENCODER_PATH)
