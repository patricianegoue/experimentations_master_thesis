import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import matplotlib.pyplot as plt
import os

# --- Charger modèle de base ---
model_path = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# --- Charger dataset de test ---
df_test = pd.read_csv("/media/patricia/Donnees/Federated_learning/test_local_finetune/test_dataset.csv")
df_test = df_test.dropna(subset=["label"])

# --- Prompt Few-shot ---
FEWSHOT = """You are a helpful assistant on semantic table interpretation in Cybersecurity domain.
Your domain is to provide the URI of the entity in the SEPSES knowledge graph for CWE entities.
Please do not include any other text in your response, just give the URI of the entity if you know it.
If you don't know the URI, return "NIL" or "I don't know".

Here are few examples of your tasks:
Q: What is the SEPSES URI of the CWE entity 20?
A: http://w3id.org/sepses/resource/cwe/applicablePlatform/CWE-20
Q: What is the SEPSES URI of the CWE entity 79?
A: http://w3id.org/sepses/resource/cwe/applicablePlatform/CWE-79
Q: What is the SEPSES URI of the CWE entity 89?
A: http://w3id.org/sepses/resource/cwe/applicablePlatform/CWE-89
"""

# --- Fonction d'inférence ---
def generate_annotation(label, max_length=50):
    prompt = FEWSHOT + f"\nQ: What is the SEPSES URI of the CWE entity {label}?\nA:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=inputs["input_ids"].shape[1] + max_length,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if f"CWE entity {label}?\nA:" in generated:
        result = generated.split(f"CWE entity {label}?\nA:")[1].strip().split("\n")[0].strip()
    else:
        result = generated.strip()

    return result

# --- Listes pour les mesures ---
inference_times = []
memory_usages = []

predictions = []

for i, label in enumerate(df_test["label"]):
    print(f"🔍 [{i+1}/{len(df_test)}] Génération pour : {label}")

    # Mesurer mémoire avant
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start_time = time.perf_counter()

    try:
        annotation = generate_annotation(label)
    except Exception as e:
        print(f"⚠️ Erreur pour '{label}' : {e}")
        annotation = "ERROR"

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    # Mémoire après
    if device.type == "cuda":
        mem_used = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # En Mo
    else:
        mem_used = 0  # Pas applicable CPU

    inference_times.append(elapsed_time)
    memory_usages.append(mem_used)

    predictions.append(annotation)

df_test["entity"] = predictions

# --- Sauvegarder résultats ---
output_path = "/media/patricia/Donnees/Federated_learning/inference/inference2_base_model.csv"
df_test.to_csv(output_path, index=False)
print(f"✅ Annotations terminées. Résultat sauvegardé dans : {output_path}")

# --- Tracer les courbes ---
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(inference_times, marker='o')
plt.title("Temps d'inférence par exemple")
plt.xlabel("Index de l'exemple")
plt.ylabel("Temps (secondes)")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(memory_usages, marker='o', color='orange')
plt.title("Mémoire GPU utilisée par exemple")
plt.xlabel("Index de l'exemple")
plt.ylabel("Mémoire (Mo)")
plt.grid(True)

plt.tight_layout()
plt.savefig("/media/patricia/Donnees/Federated_learning/inference/perf_curves.png")
plt.show()
