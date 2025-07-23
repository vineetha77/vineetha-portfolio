import easyocr
import cv2

# Initialize once
reader = easyocr.Reader(['en'])

def recognize_jersey_number(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None

    result = reader.readtext(image, detail=0, paragraph=False)
    for text in result:
        cleaned = text.strip().replace(" ", "")
        if cleaned.isdigit():
            return cleaned
    return None