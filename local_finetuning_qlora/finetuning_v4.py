import pandas as pd
import torch
import matplotlib.pyplot as plt
import psutil
import threading
import time
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

# === CONFIGURATION ===
MODEL_ID = "distilgpt2"
CSV_PATH = "/media/patricia/Donnees/Federated_learning/entrainement/dataset/dataset_local/dataset_local_train.csv"
OUTPUT_DIR = "distilgpt2_qlora_finetuned__v2"
MAX_LENGTH = 64
BATCH_SIZE = 1
EPOCHS = 3
MONITOR_INTERVAL = 1  # secondes

# === MONITORING DES RESSOURCES ===
cpu_usage = []
ram_usage = []
gpu_usage = []
timestamps = []
monitoring_active = True

def monitor_resources():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)  # GPU 0
    start_time = time.time()
    while monitoring_active:
        cpu_usage.append(psutil.cpu_percent(interval=None))
        ram_usage.append(psutil.virtual_memory().percent)
        gpu_mem = nvmlDeviceGetMemoryInfo(handle)
        gpu_usage.append(gpu_mem.used / 1024 ** 2)  # en MB
        timestamps.append(time.time() - start_time)
        time.sleep(MONITOR_INTERVAL)

monitor_thread = threading.Thread(target=monitor_resources)
monitor_thread.start()

# === Dataset formatting ===
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

    return Dataset.from_list([format_row(row) for _, row in df.iterrows()])

dataset = load_and_format_dataset(CSV_PATH)

# === Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# === Tokenization
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

tokenized_dataset = dataset.map(tokenize)

# === Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# === Load quantized model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16
)

# === Prepare model for QLoRA
model = prepare_model_for_kbit_training(model)

# === PEFT configuration
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn", "c_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

# === TrainingArguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    report_to="none",
    logging_dir=f"{OUTPUT_DIR}/logs",
)

# === Entraînement avec SFTTrainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

print("🚀 Début de l'entraînement QLoRA avec distilgpt2...")
trainer.train()
trainer.save_model(OUTPUT_DIR)
print(f"✅ Modèle QLoRA sauvegardé dans : {OUTPUT_DIR}")

# Stopper le monitoring
monitoring_active = False
monitor_thread.join()

# === Courbe de perte
training_loss = trainer.state.log_history
steps = [log["step"] for log in training_loss if "loss" in log]
losses = [log["loss"] for log in training_loss if "loss" in log]

plt.figure(figsize=(10, 5))
plt.plot(steps, losses, label="Training Loss", color="red")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Courbe de Loss - Fine-tuning distilgpt2 avec QLoRA")
plt.legend()
plt.grid(True)
plt.savefig(f"{OUTPUT_DIR}/training_loss_curve.png")
plt.show()

# === Courbe combinée CPU / RAM / GPU ===
plt.figure(figsize=(10, 5))
plt.plot(timestamps, cpu_usage, label="CPU (%)", color="orange")
plt.plot(timestamps, ram_usage, label="RAM (%)", color="green")
plt.plot(timestamps, gpu_usage, label="GPU Memory (MB)", color="blue")
plt.xlabel("Temps (s)")
plt.ylabel("Utilisation")
plt.title("Utilisation des ressources pendant l'entraînement")
plt.legend()
plt.grid(True)
plt.savefig(f"{OUTPUT_DIR}/resources_usage.png")
plt.show()
