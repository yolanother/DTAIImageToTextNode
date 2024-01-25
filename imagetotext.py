import requests
import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

def image_url_to_text(img_url):
    print("---===to image_url_to_text===---")
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
    return image_to_text(raw_image)

def image_to_text(raw_image):
    # unconditional image captioning
    text = ""
    if not isinstance(raw_image,Image.Image):
        for image in raw_image:
            i = 255. * image.cpu().numpy()
            image = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            inputs = processor(image, return_tensors="pt").to("cuda")
            out = model.generate(**inputs)
            text = processor.decode(out[0], skip_special_tokens=True)
    else:
        inputs = processor(raw_image, return_tensors="pt").to("cuda")
        out = model.generate(**inputs)
        text = processor.decode(out[0], skip_special_tokens=True)
    print("---===to image_to_text===---:",text)
    return text

# if main
if __name__ == "__main__":
    print(image_url_to_text('http://127.0.0.1:8188/view?filename=ComfyUI_temp_hbfsi_00001_.png&type=temp&subfolder=&t=1705981100322'))