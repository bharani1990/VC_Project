import torch
from torchvision import transforms
from PIL import Image
from lpips import LPIPS

lpips_model = LPIPS(net='alex', version='0.1')

def resize_and_convert(image_path, target_size):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor

def calculate_lpips(original_path, compressed_path):
    original_img = resize_and_convert(original_path, (256, 256))
    compressed_img = resize_and_convert(compressed_path, (256, 256))
    lpips_distance = lpips_model(original_img, compressed_img)
    return lpips_distance.item()
