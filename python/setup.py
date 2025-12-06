from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from huggingface_hub import login
from dotenv import load_dotenv 
load_dotenv() 
from sentence_transformers import SentenceTransformer
HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN') 

save_path = "/root/github/python_model/bge-small-en-v1.5"
# save_path = "models/bge-small-en-v1.5"
embedding_model = SentenceTransformer(
    "BAAI/bge-small-en-v1.5",
    cache_folder=save_path
)

embedding_model.save(save_path)

model_path = "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load model correctly
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,   # correct param name
    device_map="auto",           # load on your GPU
    low_cpu_mem_usage=True
)

# Save model locally
save_dir = "/root/github/python_model/Qwen2.5-3B-Instruct-GPTQ-Int4"
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

print("Model saved successfully!")
