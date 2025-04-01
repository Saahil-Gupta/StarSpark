
# from transformers import AutoProcessor, AutoModelForVision2Seq, Trainer, TrainingArguments
# from datasets import load_from_disk

# # Load the pre-trained LLaMA 3.2 model
# model_name = "meta-llama/llama-3.2-vision"
# processor = AutoProcessor.from_pretrained(model_name)
# model = AutoModelForVision2Seq.from_pretrained(model_name)

# # Load the datasets from disk (after running dataset preprocessing)
# train_dataset = load_from_disk("train_dataset")
# val_dataset = load_from_disk("val_dataset")

# # Define training arguments
# training_args = TrainingArguments(
#     output_dir="./llama3-finetuned",
#     evaluation_strategy="epoch",
#     per_device_train_batch_size=4,
#     per_device_eval_batch_size=4,
#     save_strategy="epoch",
#     num_train_epochs=3,
#     logging_dir="./logs",
#     logging_steps=10,
# )

# # Trainer setup
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
# )

# # Start training
# trainer.train()

import os
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

# Model ID
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

# Load model and processor
model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

# Define dataset path
dataset_path = r"C:\Users\jijih\.cache\kagglehub\datasets\reevald\geometric-shapes-mathematics\versions\4\dataset"
circle_path = os.path.join(dataset_path, "train", "circle")

# Get the first image from "circle" folder
image_files = [f for f in os.listdir(circle_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
if not image_files:
    raise ValueError("No images found in 'circle' folder.")

image_path = os.path.join(circle_path, image_files[0])  # Take first image
image = Image.open(image_path)

# Define messages for Llama 3.2 Vision
messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "What shape is this?"}
    ]}
]

# Process input
input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
inputs = processor(image, input_text, add_special_tokens=False, return_tensors="pt").to(model.device)

# Generate output
output = model.generate(**inputs, max_new_tokens=30)

# Decode and print response
response = processor.decode(output[0], skip_special_tokens=True)
print("Llama 3.2 Vision Output:", response)

