import io
import cv2
import numpy as np

# ================================================================= #

def get_image_from_url(content):
    image_stream = io.BytesIO(content)
    image = cv2.imdecode(np.frombuffer(image_stream.read(), np.uint8), 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # cropped_image = mask_crop(image)
    # resized_image = resize(cropped_image, None, 512, 512)

    return image