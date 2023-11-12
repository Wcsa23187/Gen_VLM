import torch
import numpy as np
from PIL import Image

def load_image_as_tensor(path):
    
    pil_image = Image.open(path).convert('RGB')

    numpy_image = np.array(pil_image)

    numpy_image = np.transpose(numpy_image, (2, 0, 1))

    tensor_image = torch.from_numpy(numpy_image).float()
    tensor_image = (tensor_image / 127.5) - 1
    tensor_image = tensor_image.unsqueeze(0)
    return tensor_image

x = load_image_as_tensor('/home/changsheng/glide-text2im/output_image_256.png')
print(x.shape)
print(x)
