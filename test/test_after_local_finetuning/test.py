import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
# --- Charger modèle fine-tuné ---
model_path = "/media/patricia/Donnees/Federated_learning/models_finetune/tiny_mistral_finetuned_output"
# Vérifie que le dossier existe
if not os.path.isdir(model_path):
    raise ValueError(f" Le dossier du modèle n'existe pas : {model_path}")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# --- Charger dataset de test ---
df_test = pd.read_csv("/media/patricia/Donnees/Federated_learning/test_local_finetune/cleaned_dataset_split_5000.csv")

# Supprimer les lignes où 'label' est vide ou NaN
df_test = df_test.dropna(subset=["label"])

# --- Génération des annotations ---
def generate_annotation(label, max_length=100):
    prompt = f"""<s>
### Instruction: Use this agent to make semantic annotation of your datas
### Human: Generate SEPSES URI of {label}
### Assistant:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=inputs["input_ids"].shape[1] + max_length,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "### Assistant:" in generated:
        result = generated.split("### Assistant:")[1].strip().split("</s>")[0].strip()
    else:
        result = generated.strip()

    return result

# --- Appliquer à chaque ligne du DataFrame avec gestion des erreurs ---
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

# --- Sauvegarder le fichier enrichi ---
output_path = "/media/patricia/Donnees/Federated_learning/response_after_local_finetuning/test_mistral_output_annotations.csv"
df_test.to_csv(output_path, index=False)

print(f"✅ Annotations terminées. Résultat sauvegardé dans : {output_path}")
