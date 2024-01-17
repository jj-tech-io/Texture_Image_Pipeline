import torch
import torch.onnx
import onnx
import torch_face
from torch_face.model import ConvBNReLU
import torch
import torch.onnx
import onnx
from torch_face.model import BiSeNet  # Import BiSeNet model class
from torch_face.face_part_segmentation import FacePartSegmentation  # Import FacePartSegmentation class

# Function to export BiSeNet model within FacePartSegmentation to ONNX
def export_bisenet_model_to_onnx(model_path, onnx_model_path, input_shape, opset=13):

    net = BiSeNet(19)
    net.cuda()
    net.eval()
    in_ten = torch.randn(16, 3, 640, 480).cuda()
    out = net(in_ten)

    # Export the BiSeNet model to ONNX
    torch.onnx.export(net, in_ten, onnx_model_path, opset_version=opset, do_constant_folding=True)

    # Optional: Check the ONNX model for validity
    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model {onnx_model_path} is valid.")

# Example usage
py_seg_path = r"C:\Users\joeli\Dropbox\Code\Python Projects\Texture_Image_Pipeline\torch_face\79999_iter.pth"
export_bisenet_model_to_onnx(py_seg_path, "py_seg.onnx", input_shape=(1, 3, 512, 512))
