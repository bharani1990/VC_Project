import os
from PIL import Image
import torch
from torchvision import transforms
from scipy.fft import dct


def compress_lt(model, image_path, save_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    img = Image.open(image_path).convert('L')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        compressed_img = model(img_tensor)
    
    compressed_img = transforms.ToPILImage()(compressed_img.squeeze(0).cpu())
    filename = os.path.basename(image_path)
    compressed_path = os.path.join(save_path, os.path.splitext(filename)[0] + '_lt_compressed.jpg')
    compressed_img.save(compressed_path)
    return compressed_path

def compress_dct(image_path, save_path):
    img = Image.open(image_path).convert('L')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0)
    img_np = img_tensor.squeeze(0).numpy()
    dct_img = dct(dct(img_np, axis=0, norm='ortho'), axis=1, norm='ortho')
    dct_tensor = torch.tensor(dct_img, dtype=torch.float32).unsqueeze(0)
    compressed_img = transforms.ToPILImage()(dct_tensor.squeeze(0).clamp(0, 1))
    filename = os.path.basename(image_path)
    compressed_path = os.path.join(save_path, os.path.splitext(filename)[0] + '_dct_compressed.jpg')
    compressed_img.save(compressed_path)
    return compressed_path
