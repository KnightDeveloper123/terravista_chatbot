from huggingface_hub import hf_hub_download  , login 
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama 
import time 
from dotenv import load_dotenv 
load_dotenv() 
import os 
import sys 

HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')

login(HUGGINGFACE_TOKEN) 

# sentence_transformer = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# sentence_transformer.save('./models/all-MiniLM-L6-v2')
print("✔ successfully Download the senetence Transformer model............") 

LLM_model = hf_hub_download(
    repo_id='Qwen/Qwen2.5-3B-Instruct-GGUF'  ,  
    filename='qwen2.5-3b-instruct-q4_k_m.gguf' , 
    local_dir= './models'
    
) 


print("✔ successfully Download the LLM model............") 
 