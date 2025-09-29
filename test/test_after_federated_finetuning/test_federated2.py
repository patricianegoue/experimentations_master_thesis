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

# === 3. Fonction de génération (avec prompt CEA) ===
def generate_annotation_cta(label, max_length=100):
    prompt = f"""Please what is sepses uri  of this entity: {label}?"""

    instruction = """You are a helpful assistant on semantic table interpretation in Cybersecurity domain.
Your domain is to provide the uri of the entity in the sepses knowledge graph for CWE entities.
Please do not include any other text in your response, just give uri of the entity if you know it.
If you don't know the uri of the entity, please return 'I don't know' or 'NIL'.
Don't include explanation or any other text.

Now responds:
Question: {prompt}
Answer:"""

    inputs = tokenizer(instruction, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=inputs["input_ids"].shape[1] + max_length,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extraire la réponse juste après le dernier 'Q:' ou 'A:'
    if "A:" in generated:
        result = generated.split("A:")[-1].strip().split("\n")[0].strip()
    else:
        result = generated.strip()

    return result

# === 4. Appliquer la génération sur chaque entrée ===
predictions = []

for i, label in enumerate(df_test["label"]):
    print(f"🔍 [{i+1}/{len(df_test)}] Génération pour : {label}")
    try:
        annotation = generate_annotation_cta(label)
    except Exception as e:
        print(f"⚠️ Erreur pour '{label}' : {e}")
        annotation = "ERROR"
    predictions.append(annotation)

df_test["entity"] = predictions

# === 5. Sauvegarder les résultats ===
output_path = "/media/patricia/Donnees/Federated_learning/response_after_federated_training/test_distilgpt2_sepses_annotations.csv"
df_test.to_csv(output_path, index=False)

print(f"\n✅ Annotations terminées. Résultat sauvegardé dans : {output_path}")




