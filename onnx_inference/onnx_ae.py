import onnxruntime as ort
import numpy as np

class ONNXAutoencoder:
    def __init__(self, encoder_path, decoder_path):
        # Load ONNX models
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            self.encoder_session = ort.InferenceSession(encoder_path, providers=['CUDAExecutionProvider'])
            self.decoder_session = ort.InferenceSession(decoder_path, providers=['CUDAExecutionProvider'])
            print("CUDAExecutionProvider available. Using GPU.")
        else:
            self.encoder_session = ort.InferenceSession(encoder_path)
            self.decoder_session = ort.InferenceSession(decoder_path)
            print("CUDAExecutionProvider not available. Using CPU.")


        self.encoder_session = ort.InferenceSession(encoder_path)
        self.decoder_session = ort.InferenceSession(decoder_path)

    def run_inference(self, session, input_data):
        input_name = session.get_inputs()[0].name
        result = session.run(None, {input_name: input_data})
        return result

    def encode(self, img):
        # Preprocess and reshape img if needed
        # Example: img = img.reshape((1, *img.shape))
        img = img.astype(np.float32)
        return self.run_inference(self.encoder_session, img)

    def decode(self, encoded_data):
        # Reshape encoded_data if needed
        # Example: encoded_data = encoded_data.reshape((1, *encoded_data.shape))
        encoded_data = encoded_data.astype(np.float32)
        return self.run_inference(self.decoder_session, encoded_data)

        # Prepare your input data (example data provided here)
