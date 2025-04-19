import os
from dotenv import load_dotenv
import logging
import base64
from PIL import Image
import requests
import io

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(f"GROQ_API_KEY has not been set in file .env")

def process_img(img_path, query):
    try:
        with open(img_path, "rb") as img_file:
            img_content = img_file.read()
            encoded_img = base64.b64encode(img_content).decode("utf-8")
            try:
                img = Image.open(io.BytesIO(img_content))
                img.verify()
            except Exception as e:
                logger.error(f"Invalid image: {str(e)}")
                return (f"Invalid image: {str(e)}")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_img}"}}
                ]
            }
        ]

        def make_api_response(model):
            response = requests.post(
                GROQ_API_URL,
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": 1000
                },
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json"
                },
                timeout=30
            )
            return response

        llama_11b_response = make_api_response("llama-3.2-11b-vision-preview")
        llama_90b_response = make_api_response("llama-3.2-90b-vision-preview")
        
        responses = {}

        for model, response in [("llama_11b", llama_11b_response), ("llama_90b", llama_90b_response)]:
            if response.status_code == 200:
                result = response.json()
                answer = result["choices"][0]["message"]["content"]
                logger.info(f"Processed response from {model} API: {answer}")
                responses[model] = answer
            else:
                logger.error(f"Error from {model} API: {response.status_code} - {response.text}")
                return (f"Error from {model} API: {response.status_code}")

        return responses

    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        return (f"An unexpected error occureed: {str(e)}")

if __name__ == "__main__":
    img_path = "cancer.jpg"
    query = "What is the disease in the picture ?"

    responses = process_img(img_path, query)

    for model, response in responses.items():
        print(model)
        print(response)