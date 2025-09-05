from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_demo_image(image_size,device):
    img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
    # raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')   
    raw_image = Image.open('./4885.png').convert("RGB") # Ensure the image is RGB
    
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    image = transform(raw_image).unsqueeze(0).to(device)   
    return image


from models.blip import blip_decoder

image_size = 384
image = load_demo_image(image_size=image_size, device=device)

# model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
# model = blip_decoder(pretrained=model_url, image_size=image_size, vit='base')
model_path = '/workspace/models/model_base_caption_capfilt_large.pth'
model = blip_decoder(pretrained=model_path, image_size=384, vit='base')
model.eval()
model = model.to(device)

with torch.no_grad():
    # beam search
    beam_search_caption = model.generate(image, sample=False, num_beams=1, max_length=48, min_length=24) 
    print('beam_search_caption: '+beam_search_caption[0])