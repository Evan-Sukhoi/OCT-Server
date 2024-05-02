import torch
from PIL import Image
import torchvision.transforms as transforms

from build.CTTIF import CTTIF

model = CTTIF('config/test.yaml')

image1_path = "image.png"

image_transform = transforms.Compose([
    transforms.Resize((128, 128)),  
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

image1 = Image.open(image1_path)
image1 = image_transform(image1)
image1 = image1.unsqueeze(0)  

result = model.test(image1)
# image2 = Image.open(image2_path)
# image2 = image_transform(image2)
# image2 = image2.unsqueeze(0)  

# concatenated_image = torch.cat((image1), dim=1)

