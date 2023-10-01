# Inference an Image on ImageChat-2, Image File Url
import requests
import json
from io import BytesIO

from custom_nodes.DTAIImageToTextNode import imagechatconfig


def predict_api(model_id, image_file_url, api_key=None, parameters=None):

    payload = {
        "image_file_url": image_file_url,
        "model_id": model_id,
        "parameters": parameters,
    }

    url = "https://apiv2.chooch.ai/predict?api_key={}".format(api_key)
    response = requests.put(url, data=json.dumps(payload))
    json_data = json.loads(response.content)
    return json_data


# Chooch ImageChat-2 model_id
model_id_image_chat_pt = "ad420c2a-d565-48eb-b963-a8297a0e4000"
image_file_url = ""

parameters = {}

# Default is True. If a prompt is given only 1 class will be returned and deep_detection will be turned off
parameters["deep_inference"] = True
#parameters["prompt"] = ["describe in detail what is in the image", "describe the lighting in the image", "describe the camera work of the image"]

parameters["workflow_predict_objects_only"] = False

parameters["workflow_padding"] = 0.6

parameters["workflow_conf_thresh"] = 0.3
parameters["workflow_nms_thresh"] = 0.45


def image_url_to_text(url):
    return predict_api(
        model_id_image_chat_pt, url, api_key=imagechatconfig.apikey, parameters=parameters
    )


# main
if __name__ == "__main__":
    print(
        image_url_to_text(
            "https://doubtech-aiart.s3.amazonaws.com/images/83cf4e5b-5147-4f9d-bf50-3fa4bca05b81.png"
        )
    )