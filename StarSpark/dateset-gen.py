import os
import matplotlib.pyplot as plt
from PIL import Image

# Path to dataset
dataset_path = r"C:\Users\jijih\.cache\kagglehub\datasets\reevald\geometric-shapes-mathematics\versions\4\dataset"

# Define categories (train, test, val)
data_splits = ["train", "test", "val"]

# Iterate through dataset splits
for split in data_splits:
    split_path = os.path.join(dataset_path, split)

    if not os.path.exists(split_path):
        print(f"Warning: '{split}' folder not found.")
        continue

    print(f"\nDisplaying images from '{split}' split:\n")

    # Iterate through shape folders (e.g., circle, rectangle, etc.)
    for shape_folder in os.listdir(split_path):
        shape_path = os.path.join(split_path, shape_folder)

        if not os.path.isdir(shape_path):  # Skip non-folder files
            continue

        print(f"Shape: {shape_folder}")

        # Get all image files in the shape folder
        images = [f for f in os.listdir(shape_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not images:
            print(f"  No images found for {shape_folder}.")
            continue

        # Display first 2 images from this shape folder
        for i, img_file in enumerate(images[:2]):
            img_path = os.path.join(shape_path, img_file)
            img = Image.open(img_path)

            plt.figure(figsize=(4, 4))
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"{split} - {shape_folder} ({img_file})")
            plt.show()

        # Limit to first shape folder per split
        break  # Remove this `break` if you want to loop through all shapes
