import torch
import numpy as np
from torchvision import transforms
from PIL import Image

# Load your model - adjust the path and model class as needed
model = torch.load('path_to_your_model.pth')
model.eval()

def single_prediction(image_path):
    # Preprocess image
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Model prediction
    with torch.no_grad():
        output = model(image)
    return output.numpy()

def batch_prediction(image_paths):
    images = []
    for image_path in image_paths:
        image = Image.open(image_path)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        image = transform(image)
        images.append(image)
    images = torch.stack(images)

    # Model prediction
    with torch.no_grad():
        outputs = model(images)
    return outputs.numpy()

# Example usage (replace with your actual test paths)
if __name__ == "__main__":
    single_result = single_prediction('path_to_single_image.jpg')
    print("Single prediction result:", single_result)

    batch_result = batch_prediction(['path_to_image1.jpg', 'path_to_image2.jpg'])
    print("Batch prediction results:", batch_result)
