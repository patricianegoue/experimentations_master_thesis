import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def loginHub(token):
    from huggingface_hub import login
    login(token=token)

# Connexion à Hugging Face
loginHub(token="token")

def load_model_and_tokenizer(model_id="distilgpt2"):
    print("[INFO] Chargement du modèle en FP32 sans LoRA ni quantization...")
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
