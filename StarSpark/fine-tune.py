import os
import torch
from torchvision import transforms
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

# Load LLaMA 3.2 Vision processor and model
model_name = "meta-llama/llama-3.2-vision"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForVision2Seq.from_pretrained(model_name)

# Define dataset path
dataset_path = r"C:\Users\jijih\.cache\kagglehub\datasets\reevald\geometric-shapes-mathematics\versions\4\dataset"

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match model input size
    transforms.ToTensor(),  # Convert to tensor
])

# Function to load images and prepare data for training
def load_images_and_labels(split="train"):
    split_path = os.path.join(dataset_path, split)
    data = []

    for shape_folder in os.listdir(split_path):
        shape_path = os.path.join(split_path, shape_folder)
        if not os.path.isdir(shape_path):
            continue  # Skip non-folder files

        images = [f for f in os.listdir(shape_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_file in images:
            img_path = os.path.join(shape_path, img_file)
            image = Image.open(img_path).convert("RGB")
            image = transform(image)  # Apply transformation
            
            # The "label" is the name of the shape folder (e.g., "circle", "square")
            label = f"This is a {shape_folder}."  

            data.append({"image": image, "label": label})

    return data

# Load training data
train_data = load_images_and_labels("train")
val_data = load_images_and_labels("val")

print(f"Loaded {len(train_data)} training images and {len(val_data)} validation images.")
