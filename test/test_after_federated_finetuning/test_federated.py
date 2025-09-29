import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# === 1. Charger distilgpt2 et les poids fédérés ===
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# Charger les poids entraînés fédérés
model.load_state_dict(torch.load(
    "/media/patricia/Donnees/Federated_learning/distilgpt2_aggregated_model.pt",
    map_location=torch.device("cpu")
))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# === 2. Charger dataset de test ===
df_test = pd.read_csv("/media/patricia/Donnees/Federated_learning/test_local_finetune/test_dataset.csv")
df_test = df_test.dropna(subset=["label"])

# === 3. Fonction de génération ===
def generate_annotation(label, max_length=100):
    prompt = f"Generate the semantic annotation of: {label} \n Please do not include any other text in your response, just give uri of the entity if you know it. \nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=inputs["input_ids"].shape[1] + max_length,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Answer:" in generated:
        result = generated.split("Answer:")[1].strip().split("\n")[0].strip()
    else:
        result = generated.strip()
    return result

# === 4. Appliquer sur tout le dataset ===
predictions = []

for i, label in enumerate(df_test["label"]):
    print(f"🔍 [{i+1}/{len(df_test)}] Génération pour : {label}")
    try:
        annotation = generate_annotation(label)
    except Exception as e:
        print(f"⚠️ Erreur pour '{label}' : {e}")
        annotation = "ERROR"
    predictions.append(annotation)

df_test["entity"] = predictions

# === 5. Sauvegarder les résultats ===
output_path = "/media/patricia/Donnees/Federated_learning/response_after_federated_training/test2_distilgpt2_output_annotations.csv"
df_test.to_csv(output_path, index=False)

print(f"\n✅ Annotations terminées. Résultat sauvegardé dans : {output_path}")
