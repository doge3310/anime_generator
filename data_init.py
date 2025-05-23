import os
import kagglehub
from PIL import Image
from torchvision import transforms


path = kagglehub.dataset_download("splcher/animefacedataset") + r"\images"
images = os.listdir(path)[: 1000]
size = 64
images_list = []
transform = transforms.Compose([transforms.Resize(size),
                                transforms.CenterCrop(size),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,) * 3, (0.5,) * 3)])

for index, image in enumerate(images):
    image = Image.open(fr"{path}\{image}").resize((size, size)).convert("RGB")
    images_list.append(transform(image))
