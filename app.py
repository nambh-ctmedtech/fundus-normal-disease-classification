import requests
import base64
import cv2

from fastapi import FastAPI, Query

from configs.api_configs import get_image_from_url
from process import process

# ================================================================= #

app = FastAPI()

@app.post("/normal-disease-classification")
async def normal_disease_classification(image_url: str = Query(..., description=".")):
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image = get_image_from_url(response.content)
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to retrieve image from URL: {e}"}
    
    pred, prob = process(image)

    _, encoded_image = cv2.imencode('.png', image)
    base64_image = base64.b64encode(encoded_image).decode("utf-8")

    results = {
        "image": f"data:image/png;base64,{base64_image}",
        "predict": pred,
        "probability": prob
    }

    return results
