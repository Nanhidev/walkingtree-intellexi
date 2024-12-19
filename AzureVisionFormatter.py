import anthropic
import base64
import requests
import base64
import os
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
load_dotenv()

VISION_KEY=os.environ["VISION_KEY"] 
VISION_ENDPOINT=os.environ["VISION_ENDPOINT"]



try:
    endpoint = os.environ["VISION_ENDPOINT"]
    key = os.environ["VISION_KEY"]
except KeyError:
    print("Missing environment variable 'VISION_ENDPOINT' or 'VISION_KEY'")
    print("Set them before running this sample.")
    exit()

ANTHROPIC_API_KEY=os.environ["ANTHROPIC_API_KEY"] 
print("-------------------------------ANTHROPIC--------------",ANTHROPIC_API_KEY)


class AzureVisionBasedFormatter:
    """
    NOTE: The image Url needs to be Path to the image that can be accessible by Azure, and not on local file directory.
    """

    def __init__(self, image_url) -> None:
        self.image_url = image_url
        self.text_identified = []
        self.text = ""
        self.endpoint = os.environ["VISION_ENDPOINT"]
        self.key = os.environ["VISION_KEY"]
        self.text_words_confidence={}

    def extract(self):
        # Create an Image Analysis client
        client = ImageAnalysisClient(
            endpoint=self.endpoint, credential=AzureKeyCredential(self.key)
        )
        # Get a caption for the image. This will be a synchronously (blocking) call.
        result = client.analyze_from_url(
            image_url=self.image_url,
            visual_features=[VisualFeatures.CAPTION, VisualFeatures.READ],
            gender_neutral_caption=True,  # Optional (default is False)
        )
        print("Image analysis results:")
        # Print caption results to the console
        print(" Caption:")
        if result.caption is not None:
            print(
                f"   '{result.caption.text}', Confidence {result.caption.confidence:.4f}"
            )

        if result.read is not None:
            for line in result.read.blocks[0].lines:
                self.text_identified.append(line["text"])
                self.text_words_confidence[line["text"]]={word["text"]:word["confidence"] for word in line["words"]}
        self.text = " ".join(self.text_identified)
        return self.text, self.text_words_confidence


class AzureAssistedGPTFormatter:
    def __init__(
        self, image_url, doc_id, prompt_instructions, conversion_ontology, vision_text
    ) -> None:
        self.image_url = image_url
        self.conversion_ontology= conversion_ontology
        self.prompt=prompt_instructions
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        }
        self.file_type=self.get_type()

        self.text = vision_text
    def get_type(self):
        ext=self.image_url.split(".")[-1]
        if ext =="jpg":
            return "jpeg"
        else:
            return ext
    def encode_image(self):
        with open(
            self.image_url,"rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def payload(self, base64_image):
        print("file type -----------------",self.file_type)
        return {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{self.prompt}\nThe final Required Json Format is {self.conversion_ontology}\nFor reference  Use the azure vision extracted text from the same image in case if you do not understand any of the handwritten content.\nThe Azure vision Extarcted Text is {self.text}\nStrictly return the Json response Only, no more description. The Final Json is:",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{self.file_type};base64,{base64_image}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 1000,
        }

    def encode_image(self):
        with open(self.image_url, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def extract(self):
        base64_image = self.encode_image()
        payload = self.payload(base64_image=base64_image)
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=self.headers,
            json=payload,
        )
        print(response.json())
        json_res=response.json()
        print("----------------json is-----------------",json_res)
        return json_res["choices"][0]["message"]["content"]
        # return json_res


class AzureAssistedClaudeFormatter:
    def __init__(
        self, image_url, doc_id, prompt_instructions, conversion_ontology, vision_text
    ) -> None:
        self.image_url = image_url
        self.doc_id = doc_id
        self.prompt = prompt_instructions
        self.conversion_ontology = conversion_ontology
        self.text = vision_text
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def encode_image(self):
        with open(
            self.image_url,"rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    # def encode_image(self):
    #     response = requests.get(self.image_url)
    #     if response.status_code == 200:
    #         return base64.b64encode(response.content).decode("utf-8")
    #     else:
    #         print("Failed to fetch the image:", response.status_code)
    #         return None

    def extract_using_claude(self):
        print("     --------------- image Url", self.image_url)
        image_data = self.encode_image()
        image_type = self.image_url.split(".")[-1]
        if image_type == "jpg":
            image_type = "jpeg"
        image_media_type = f"image/{image_type}"
        message = self.client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=2024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": image_media_type,
                                "data": image_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": f"{self.prompt}\nThe final Required Json Format is {self.conversion_ontology}\nFor reference  Use the azure vision extracted text from the same image in case if you do not understand any of the handwritten content.\nThe Azure vision Extarcted Text is {self.text}\nStrictly return the Json response Only, no more description. The Final Json is:",
                        },
                    ],
                }
            ],
        )
        print(message.content[0].text)
        return message.content[0].text