import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
import base64
import cv2

from model import get_efficientnet_model

# ================================================================= #

def process(image: np.ndarray):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = get_efficientnet_model("efficientnet_b3", num_classes=1, pretrained=False)
    model.load_state_dict(torch.load("checkpoints/normal_disease_512.pth"))
    model.to(device)
    
    model.eval()

    transform = T.Compose([
        # T.Resize((512, 512)), 
        # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
        T.ToTensor() 
    ]) 
    input_image = transform(Image.fromarray(image)).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_image)
        prob = torch.sigmoid(output).item()
        pred = prob > 0.5
        if pred:
            pred = "disease"
        else:
            pred = "normal"
            prob = 1 - prob

    return pred, prob

