from io import BytesIO

import torch
from PIL import Image, ImageOps
from transformers import pipeline
from transformers import ViTFeatureExtractor, AutoTokenizer, VisionEncoderDecoderModel
import requests

vit_gpt2_img_caption = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")


def vit_gpt2_img_caption_from_url(url):
    return vit_gpt2_img_caption(url)[0]['generated_text']


def load_image_from_url(url):
    try:
        # Send a GET request to fetch the image data
        response = requests.get(url)

        # Check if the request was successful
        response.raise_for_status()

        # Read the image data and create a PIL image
        image = Image.open(BytesIO(response.content))

        return image

    except requests.exceptions.RequestException as e:
        print(f"Error loading image from URL: {url}")
        print(e)
        return None

# Model
model_id = "nttdataspain/vit-gpt2-stablediffusion2-lora"
model = VisionEncoderDecoderModel.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_id)

# Predict function
def predict_prompts(list_images, max_length=16):
    model.eval()
    pixel_values = feature_extractor(images=list_images, return_tensors="pt").pixel_values
    with torch.no_grad():
        output_ids = model.generate(pixel_values, max_length=max_length, num_beams=4, return_dict_in_generate=True).sequences

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds

def predict_prompts_from_url(url, max_length=256):
    img = load_image_from_url(url)
    return predict_prompts([img], max_length=256)


if __name__ == "__main__":
    print(predict_prompts_from_url('https://doubtech-aiart.s3.amazonaws.com/images/db50fb5c-650e-4723-b5e9-22c7d81e662c.png'))
    #print(image_url_to_text('https://doubtech-aiart.s3.amazonaws.com/images/db50fb5c-650e-4723-b5e9-22c7d81e662c.png'))