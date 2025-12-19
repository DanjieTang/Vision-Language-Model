from ollama_llm import OllamaLLM

llm = OllamaLLM(model="qwen3-vl:32b")

print(llm.llm_response("Please describe to me what's in the image. Please keep the description short, and do not output anything other than the description because the output is going directly to the caption of the image and will be presented to my client.", "/Users/danjietang/Documents/Github/VLM/coco2017/train2017/000000000009.jpg", system_message="You are an image annotator"))