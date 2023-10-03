"""Detects labels in the file."""
from google.cloud import vision
import io

def detect_labels(image_content):   
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_content)
    response = client.label_detection(image=image, max_results=30) # set topN with max_results
    labels = response.label_annotations
    
    # print('labels in one image:')
    # for label in labels:
    #     print(f'{label.description}: {label.score}')
    
    return labels

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))