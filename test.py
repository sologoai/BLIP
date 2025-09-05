from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from models.blip import blip_decoder

image_size = 384
transform = transforms.Compose([
    transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ]) 


model_path = '/workspace/models/model_base_caption_capfilt_large.pth'
model = blip_decoder(pretrained=model_path, image_size=384, vit='base')

model.eval()
model = model.to(device)

def inference(image_path, model, strategy="Nucleus sampling"):
    raw_image = Image.open(image_path).convert("RGB") # Ensure the image is RGB
    image = transform(raw_image).unsqueeze(0).to(device)   
    with torch.no_grad():
        if strategy == "Beam search":
            caption = model.generate(image, sample=False, num_beams=3, max_length=20, min_length=5)
        else:
            caption = model.generate(image, sample=True, top_p=0.9, max_length=20, min_length=5)
    return 'Caption: '+caption[0]

# 输入图片路径
image_path = './4885.png'
print(inference(image_path, model))