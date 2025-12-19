from ollama_llm import OllamaLLM
from pathlib import Path
from tqdm import tqdm

llm = OllamaLLM(model="qwen3-vl")

image_dir_path = Path("coco2017/train2017")
annotation_dir_path = Path("annotation/train2017")

system_prompt = "You are an image annotator"
user_prompt = "Please describe to me what's in the image. Please keep the description short, and do not output anything other than the description because the output is going directly to the caption of the image and will be presented to my client."

for file in tqdm(image_dir_path.iterdir()):
    # Skip files that are not jpg
    if file.is_file() and file.suffix.lower() != '.jpg':
        continue
    
    # Check if the annotated file already exist
    annotation_file = annotation_dir_path / f"{file.stem}.txt"
    if annotation_file.exists():
        continue
    
    # Retry Logic Configuration
    max_retries = 3

    for attempt in range(max_retries):
        try:
            # Call LLM
            annotation: str = llm.llm_response(user_prompt, str(file), system_message=system_prompt)
            annotation_file.write_text(annotation.strip(), encoding="utf-8")
            break  # Exit the retry loop on success
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for {file.name}: {e}")
            if attempt == max_retries - 1:
                # If this was the last attempt, raise the exception to stop the script
                raise Exception(f"Failed to process {file.name} after {max_retries} attempts. Error: {e}")