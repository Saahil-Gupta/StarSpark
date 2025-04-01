from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
from torchvision import transforms

# Load the pre-trained and fine-tuned LLaMA model
model_name = "meta-llama/llama-3.2-vision"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForVision2Seq.from_pretrained("./llama3-finetuned")

# Define transformation for incoming images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load an image for testing
test_image_path = r"C:\path\to\image.png"
test_image = Image.open(test_image_path).convert("RGB")
test_image = transform(test_image)

# Prepare image for model
inputs = processor(images=test_image, return_tensors="pt")

# Generate description
outputs = model.generate(**inputs)
description = processor.decode(outputs[0], skip_special_tokens=True)

# Output the description
print("Generated Description:", description)
