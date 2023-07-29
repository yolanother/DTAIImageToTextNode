import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

def image_url_to_text(img_url):
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
    return image_to_text(raw_image)

def image_to_text(raw_image):
    # unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt").to("cuda")

    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

# if main
if __name__ == "__main__":
    print(image_url_to_text('https://doubtech-aiart.s3.amazonaws.com/images/db50fb5c-650e-4723-b5e9-22c7d81e662c.png'))