from custom_nodes.DTAIImageToTextNode.imagetotext import image_url_to_text, image_to_text


class DTAIImageUrlToTextNode:
    def __init__(self):
        self.url = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "url": ("STRING", {
                    "multiline": False,  # True if you want the field to look like the one on the ClipTextEncode node
                    "default": "https://doubtech.ai/img/logo.png"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    # RETURN_NAMES = ("image_output_name",)

    FUNCTION = "imagetotext"

    # OUTPUT_NODE = False

    CATEGORY = "DoubTech/Image/Image To Text"

    @classmethod
    def IS_CHANGED(self, url):
        return self.url != url

    def imagetotext(self, url):
        self.url = url
        caption = image_url_to_text(url)
        print("Image appears to be: " + caption)
        return (caption,)


class DTAIImageToTextNode:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",)
            },
        }

    RETURN_TYPES = ("STRING",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "imagetotext"

    #OUTPUT_NODE = False

    CATEGORY = "DoubTech/Image/Image To Text"

    def imagetotext(self, image):
        caption = image_to_text(image)
        print("Image appears to be: " + caption)
        return (caption,)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "DTAIImageToTextNode": DTAIImageToTextNode,
    "DTAIImageUrlToTextNode": DTAIImageUrlToTextNode,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "DTAIImageToTextNode": "Image to Text",
    "DTAIImageUrlToTextNode": "Image URL to Text"
}
