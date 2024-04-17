import onnxruntime as ort
import numpy as np
print(ort.get_device())
print(ort.get_available_providers())

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
        img = img.astype(np.float32).reshape((-1,3))
        encoded_data = np.asarray(self.run_inference(self.encoder_session, img)[0])
        return encoded_data

    def decode(self, encoded_data):
        encoded_data = encoded_data.astype(np.float32).reshape((-1,5))
        decoded_data = np.asarray(self.run_inference(self.decoder_session, encoded_data)[0])
        return decoded_data

