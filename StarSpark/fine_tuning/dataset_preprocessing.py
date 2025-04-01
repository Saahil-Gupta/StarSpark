# import os
# from torchvision import transforms
# from PIL import Image
# from datasets import Dataset

# # Define dataset path and transform
# dataset_path = r"C:\Users\jijih\.cache\kagglehub\datasets\reevald\geometric-shapes-mathematics\versions\4\dataset"
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])

# def load_images_and_labels(split="train"):
#     split_path = os.path.join(dataset_path, split)
#     data = []

#     for shape_folder in os.listdir(split_path):
#         shape_path = os.path.join(split_path, shape_folder)
#         if not os.path.isdir(shape_path):
#             continue

#         images = [f for f in os.listdir(shape_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
#         for img_file in images:
#             img_path = os.path.join(shape_path, img_file)
#             try:
#                 image = Image.open(img_path).convert("RGB")
#                 if image.mode != 'RGB':
#                     image = image.convert("RGB")
#                 image = transform(image)
#             except Exception as e:
#                 print(f"Error loading image {img_path}: {e}")
#                 continue  # Skip this image if it fails to load

#             # Convert tensor to a list (this makes it serializable)
#             image_list = image.detach().numpy().tolist()  # Converting tensor to list

#             # Label = folder name
#             label = f"This is a {shape_folder}."

#             # Append the data as a dictionary
#             data.append({"image": image_list, "label": label})

#     return data

# # Load train and validation datasets
# train_data = load_images_and_labels("train")
# val_data = load_images_and_labels("val")

# # Check the format of the first entry
# print(train_data[0])  # Should print a dictionary with "image" (list) and "label"

# # Save as Hugging Face Dataset
# train_dataset = Dataset.from_list(train_data)
# val_dataset = Dataset.from_list(val_data)

# # Optionally, save or return datasets
# # train_dataset.save_to_disk("train_dataset")
# # val_dataset.save_to_disk("val_dataset")

import cv2
from PIL import Image
from torchvision import transforms
import os
from datasets import Dataset
import numpy as np

# Define dataset path and transform
dataset_path = r"C:\Users\jijih\.cache\kagglehub\datasets\reevald\geometric-shapes-mathematics\versions\4\dataset"
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Converts the image to a PyTorch tensor
])

def load_images_and_labels(split="train"):
    split_path = os.path.join(dataset_path, split)
    data = []

    for shape_folder in os.listdir(split_path):
        shape_path = os.path.join(split_path, shape_folder)
        if not os.path.isdir(shape_path):
            continue

        images = [f for f in os.listdir(shape_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for img_file in images:
            img_path = os.path.join(shape_path, img_file)
            try:
                # Use OpenCV to load the image (returns a NumPy array)
                image = cv2.imread(img_path)
                # Convert from BGR (OpenCV default) to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Convert NumPy array to PIL Image
                image = Image.fromarray(image)
                # Apply transformations (Resize, ToTensor)
                image = transform(image)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue  # Skip this image if it fails to load

            # Label = folder name (shape name)
            label = shape_folder  # Using the folder name directly as the label

            data.append({"image": image, "label": label})

    return data

# Load train and validation datasets
train_data = load_images_and_labels("train")
val_data = load_images_and_labels("val")

# Display the first entry to verify the structure
print(train_data[0])  # Should print a dictionary with "image" and "label"

# Save as Hugging Face Dataset
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

# Optional: Save the datasets to disk for later use
train_dataset.save_to_disk('train_dataset')
val_dataset.save_to_disk('val_dataset')

print("Datasets saved successfully.")
