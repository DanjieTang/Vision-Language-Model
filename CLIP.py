import torch
from PIL import Image
from transformers import CLIPVisionModel, CLIPImageProcessor
import os

# 1. Load the specific Vision Transformer (ViT) part of CLIP
model_id = "openai/clip-vit-large-patch14"
model = CLIPVisionModel.from_pretrained(model_id)
processor = CLIPImageProcessor.from_pretrained(model_id)

# 2. Prepare your images from local paths
file_paths = [
    "coco2017/train2017/000000000025.jpg",
    "coco2017/ /000000000025.jpg"
]

images = []
for path in file_paths:
    if os.path.exists(path):
        # Using .convert("RGB") ensures images are in the format CLIP expects
        img = Image.open(path).convert("RGB")
        images.append(img)
    else:
        raise Exception("Oh no")

inputs = processor(images=images, return_tensors="pt")

# 3. Get the "Meaning" (Hidden States)
with torch.no_grad():
    outputs = model(**inputs)
    
    # last_hidden_state shape: [batch_size, num_patches + 1, hidden_size]
    # Indexing [:, 0, :] extracts the [CLS] token, representing the global image features
    visual_tokens = outputs.last_hidden_state[:, 0, :]

print(f"Processed {len(images)} images.")
print(f"Tokens available for LLM cross-attention: {visual_tokens.shape}")