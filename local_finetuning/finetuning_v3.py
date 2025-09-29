from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset
import pandas as pd
import time
import torch
import psutil
import platform
import threading
import subprocess

# === CONFIGURATION ===
MODEL_ID = "distilgpt2"  # léger mais puissant (82M paramètres)
CSV_PATH = "/media/patricia/Donnees/Federated_learning/entrainement/dataset/dataset_local/dataset_local_train.csv"
OUTPUT_DIR = "distilgpt2_finetuned_output"
NUM_EPOCHS = 3
BATCH_SIZE = 1

# === Charger et formater le dataset ===
def load_and_format_dataset(path): 
    df = pd.read_csv(path)

    def format_row(row):
        return {
            "text": f"""<s>
### Instruction: Use this agent to make semantic annotation of your datas
### Human: Generate SEPSES URI of {row['label']}
### Assistant: The SEPSES uri of {row['label']} is {row['entity']}
</s>"""
        }

    formatted = [format_row(row) for _, row in df.iterrows()]
    return Dataset.from_list(formatted)

dataset = load_and_format_dataset(CSV_PATH)

# === Tokenizer et modèle ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token  # distilgpt2 n’a pas de pad_token, on le définit

model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
model.to("cpu")  # CPU only

# === Tokenisation ===
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# === Collator ===
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# === Entraînement ===
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=5e-5,
    optim="adamw_torch_fused",
    remove_unused_columns=True,
    report_to="none",
    push_to_hub=False,
    no_cuda=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# === Début Exécution ===
print("Début de l'entraînement sur CPU avec distilgpt2...")

# --- Données réseau avant entraînement
net_before = psutil.net_io_counters()
time_before = time.perf_counter()

# === Démarrage monitoring dynamique dans un thread ===
def monitor_resources(interval=1, duration=10000):  # durée grande pour couvrir entraînement
    end_time = time.time() + duration
    with open("monitoring_log.txt", "w") as f:
        f.write("timestamp,cpu_percent,ram_percent,bytes_sent,bytes_recv\n")
        while time.time() < end_time:
            ts = time.time()
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent
            net = psutil.net_io_counters()
            f.write(f"{ts},{cpu},{ram},{net.bytes_sent},{net.bytes_recv}\n")
            time.sleep(interval)

monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
monitor_thread.start()

# === Entraînement ===
start = time.perf_counter()
trainer.train()
duration = (time.perf_counter() - start) / 60  # minutes

# --- Données réseau après entraînement
time_after = time.perf_counter()
net_after = psutil.net_io_counters()

net_sent_MB = (net_after.bytes_sent - net_before.bytes_sent) / (1024 ** 2)
net_recv_MB = (net_after.bytes_recv - net_before.bytes_recv) / (1024 ** 2)
net_total_MB = net_sent_MB + net_recv_MB
net_bandwidth_MBps = net_total_MB / (time_after - time_before)

print(f"Entraînement terminé en {duration:.2f} minutes")

# === Sauvegarde modèle ===
trainer.save_model(OUTPUT_DIR)
print(f"Modèle sauvegardé dans {OUTPUT_DIR}")

# === Infos machine ===
machine_info = {
    "machine_type": "desktop",  # à adapter
    "cpu": platform.processor(),
    "cores": psutil.cpu_count(logical=False),
    "threads": psutil.cpu_count(logical=True),
    "freq_MHz": psutil.cpu_freq().max,
    "ram_GB": round(psutil.virtual_memory().total / (1024**3), 2),
    "disk_GB": round(psutil.disk_usage('/').total / (1024**3), 2),
    "os": platform.system(),
    "os_version": platform.version()
}

# === Calcul speedup et efficacité ===
try:
    with open("baseline_metrics.txt", "r") as f:
        lines = f.readlines()
        t_base = float([line for line in lines if line.startswith("T_base_minutes=")][0].split("=")[1])
        cores_base = int([line for line in lines if line.startswith("Cores=")][0].split("=")[1])
        speedup = t_base / duration
        efficiency = speedup / machine_info["cores"]
except Exception:
    t_base = None
    speedup = None
    efficiency = None

# === Latence réseau (ping 8.8.8.8) ===
try:
    ping_output = subprocess.check_output(["ping", "-c", "4", "8.8.8.8"], universal_newlines=True)
    latency_lines = [line for line in ping_output.split('\n') if "time=" in line]
    avg_latency = sum([float(line.split("time=")[1].split(" ")[0]) for line in latency_lines]) / len(latency_lines)
except Exception:
    avg_latency = -1

# === Consommation d’énergie - NON MESURÉE (documenter si nécessaire) ===

# === Écriture résultats dans fichier ===
with open("evaluation_metrics.txt", "w") as f:
    f.write("=== ÉVALUATION DE PERFORMANCE ===\n")
    f.write(f"Tcal (minutes): {duration:.4f}\n")
    if speedup:
        f.write(f"Speedup: {speedup:.4f}\n")
        f.write(f"Efficacité: {efficiency:.4f}\n")
        f.write(f"T_base (référence): {t_base:.4f} minutes\n")
    f.write("\n=== MACHINE ===\n")
    for k, v in machine_info.items():
        f.write(f"{k}: {v}\n")
    f.write("\n=== RÉSEAU ===\n")
    f.write(f"Bande passante utilisée: {net_bandwidth_MBps:.2f} MB/s\n")
    f.write(f"Envoyé: {net_sent_MB:.2f} MB\n")
    f.write(f"Reçu: {net_recv_MB:.2f} MB\n")
    f.write(f"Latence moyenne (ping 8.8.8.8): {avg_latency if avg_latency != -1 else 'Erreur'} ms\n")
    f.write("\n=== LOG DYNAMIQUE ===\n")
    f.write("Voir fichier: monitoring_log.txt\n")

print("\n Tous les résultats sont sauvegardés dans 'evaluation_metrics.txt' et 'monitoring_log.txt'")
