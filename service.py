from flask import Flask, request, jsonify
import base64
import io
import requests
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from models.blip import blip_decoder

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = '/workspace/models/model_base_caption_capfilt_large.pth'
model = blip_decoder(pretrained=model_path, image_size=384, vit='base')
model.eval()
model = model.to(device)

def generate_caption(image,num_beams=1, max_length=48, min_length=24):
    with torch.no_grad():
        # beam search
        beam_search_caption = model.generate(image, sample=False, num_beams=num_beams, max_length=max_length, min_length=min_length) 
    return beam_search_caption[0]

def load_image(image_data, device):
    transform = transforms.Compose([
        transforms.Resize((384, 384), interpolation = InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ]) 
    image = transform(image_data).unsqueeze(0).to(device)   
    return image

@app.route('/post', methods=['POST'])
def post():
    data = request.json
    num_beams = data.get('num_beams', 1)
    max_length = data.get('max_length', 48)
    min_length = data.get('min_length', 24)
    outputs = []
    if 'img_urls' in data:
        for img_url in data['img_urls']:
            raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
            image = load_image(raw_image, device)
            text = generate_caption(image, num_beams=num_beams, max_length=max_length, min_length=min_length)
            outputs.append({"text": text})
    if 'img_base64s' in data:
        for img_base64 in data['img_base64s']:
            img_data = base64.b64decode(img_base64)
            raw_image = Image.open(io.BytesIO(img_data)).convert('RGB')
            image = load_image(raw_image, device)
            text = generate_caption(image, num_beams=num_beams, max_length=max_length, min_length=min_length)
            outputs.append({"text": text})
    if not outputs:
        return jsonify({"status": 1, "msg": "No valid images provided.", "data": {}})
    return jsonify({"status": 0, "msg": "ok", "data": {"output": outputs}})

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=52001)