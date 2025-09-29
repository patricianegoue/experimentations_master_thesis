import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..") 
os.environ["WANDB_DISABLED"] = "true"
import argparse
import time
import torch
import flwr as fl
import threading
import psutil
import platform
import subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from shared.data_utils import load_client_dataset_and_format



class DistilGPT2Client(fl.client.NumPyClient):
    def __init__(self, client_id):
        print(f"[INFO] Initialisation du client {client_id}")
        self.client_id = client_id

        print("[INFO] Chargement du modèle distilgpt2 (FP32)...")
        self.model = AutoModelForCausalLM.from_pretrained("distilgpt2")
        self.tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        print("[INFO] Chargement du dataset du cliegnt...")
        self.data = load_client_dataset_and_format(client_id)

        self.checkpoint_path = f"./checkpoints/client_{self.client_id}/model.pt"
        self.load_model_locally()

        print("[INFO] Client prêt.")

        # Variables monitoring
        self.monitoring = True
        self.cpu_samples = []
        self.ram_samples = []
        self.net_sent_start = psutil.net_io_counters().bytes_sent
        self.net_recv_start = psutil.net_io_counters().bytes_recv

    def save_model_locally(self):
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        torch.save(self.model.state_dict(), self.checkpoint_path)
        print(f"[INFO] Modèle sauvegardé dans {self.checkpoint_path}")

    def load_model_locally(self):
        if os.path.exists(self.checkpoint_path):
            self.model.load_state_dict(torch.load(self.checkpoint_path))
            print(f"[INFO] Modèle chargé depuis {self.checkpoint_path}")
        else:
            print("[INFO] Aucun checkpoint trouvé, initialisation à partir des poids du serveur")

    def get_parameters(self, config):
        try:
            print("[INFO] Envoi des paramètres du client au serveur...")
            return [val.cpu().detach().numpy() for _, val in self.model.named_parameters()]
        except Exception as e:
            print(f"[ERREUR] Échec de l'envoi des paramètres : {e}")
            raise

    def set_parameters(self, parameters):
        print("[INFO] Mise à jour des paramètres du modèle...")
        for param, new_val in zip(self.model.parameters(), parameters):
            if isinstance(new_val, torch.Tensor):
                param.data = torch.tensor(new_val, dtype=param.dtype).to(param.device)
            else:
                param.data.copy_(torch.tensor(new_val, dtype=param.dtype, device=param.device))

    def monitor_resources(self, interval=1):
        print(f"[INFO] Démarrage du monitoring ressources client {self.client_id}...")
        with open(f"client_{self.client_id}_monitoring_log.txt", "w") as f:
            f.write("timestamp,cpu_percent,ram_percent,bytes_sent,bytes_recv\n")
            while self.monitoring:
                ts = time.time()
                cpu = psutil.cpu_percent(interval=None)
                ram = psutil.virtual_memory().percent
                net = psutil.net_io_counters()
                f.write(f"{ts},{cpu},{ram},{net.bytes_sent},{net.bytes_recv}\n")
                f.flush()
                # Stockage pour moyenne
                self.cpu_samples.append(cpu)
                self.ram_samples.append(ram)
                time.sleep(interval)
        print(f"[INFO] Monitoring client {self.client_id} terminé.")

    def evaluate(self, parameters, config):
        print(f"[INFO] Client {self.client_id} : Évaluation ignorée")
        return 0.0, len(self.data), {}

    def fit(self, parameters, config):
        print(f"[INFO] Client {self.client_id} : Début de l'entraînement local")

        # Appliquer les poids du serveur
        self.set_parameters(parameters)

        # Reprendre le modèle local s'il existe
        self.load_model_locally()

        # Préparer subset données local
        round_frac = 0.5
        total_size = len(self.data)
        round_size = int(total_size * round_frac)
        round_subset = self.data.select(range(min(round_size, total_size)))

        training_args = TrainingArguments(
            output_dir=f"./output/client_{self.client_id}",
            per_device_train_batch_size=1,
            num_train_epochs=1,
            logging_steps=10,
            save_strategy="no",
            learning_rate=2e-5,
            optim="adamw_torch"
        )

        trainer = SFTTrainer(
            model=self.model,
            train_dataset=round_subset,
            args=training_args
        )

        # Lancer monitoring dans un thread
        self.monitoring = True
        monitor_thread = threading.Thread(target=self.monitor_resources, daemon=True)
        monitor_thread.start()

        start = time.perf_counter()
        trainer.train()
        end = time.perf_counter()

        self.monitoring = False
        monitor_thread.join(timeout=2)

        training_time_min = (end - start) / 60

        self.save_model_locally()

        # Calcul moyennes CPU, RAM
        avg_cpu = sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0
        avg_ram = sum(self.ram_samples) / len(self.ram_samples) if self.ram_samples else 0

        net_after = psutil.net_io_counters()
        net_sent_MB = (net_after.bytes_sent - self.net_sent_start) / (1024 ** 2)
        net_recv_MB = (net_after.bytes_recv - self.net_recv_start) / (1024 ** 2)

        machine_info = {
            "machine_type": "client_desktop",
            "cpu": platform.processor(),
            "cores": psutil.cpu_count(logical=False),
            "threads": psutil.cpu_count(logical=True),
            "freq_MHz": psutil.cpu_freq().max,
            "ram_GB": round(psutil.virtual_memory().total / (1024**3), 2),
            "disk_GB": round(psutil.disk_usage('/').total / (1024**3), 2),
            "os": platform.system(),
            "os_version": platform.version()
        }

        latency = -1
        try:
            ping_output = subprocess.check_output(["ping", "-c", "4", "8.8.8.8"], universal_newlines=True)
            latency_lines = [line for line in ping_output.split('\n') if "time=" in line]
            latency = sum([float(line.split("time=")[1].split(" ")[0]) for line in latency_lines]) / len(latency_lines)
        except Exception:
            pass

        # Sauvegarde résumé métriques client
        with open(f"client_{self.client_id}_evaluation_metrics.txt", "w") as f:
            f.write(f"=== Client {self.client_id} - Évaluation de performance ===\n")
            f.write(f"Temps entraînement local (minutes): {training_time_min:.4f}\n")
            f.write(f"CPU moyen (%): {avg_cpu:.2f}\n")
            f.write(f"RAM moyenne (%): {avg_ram:.2f}\n")
            f.write(f"Bytes envoyés (MB): {net_sent_MB:.2f}\n")
            f.write(f"Bytes reçus (MB): {net_recv_MB:.2f}\n")
            f.write(f"Latence réseau moyenne (ping 8.8.8.8) ms: {latency if latency!=-1 else 'Erreur'}\n")
            f.write("\n=== Machine ===\n")
            for k,v in machine_info.items():
                f.write(f"{k}: {v}\n")

        print(f"[INFO] Client {self.client_id} : Entraînement terminé en {training_time_min:.2f} minutes")

        # Retourner métriques au serveur
        return self.get_parameters({}), len(round_subset), {
            "train_time_min": training_time_min,
            "cpu_percent_avg": avg_cpu,
            "ram_percent_avg": avg_ram,
            "net_sent_MB": net_sent_MB,
            "net_recv_MB": net_recv_MB,
            "latency_ms": latency,
            **machine_info,
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Client Flower DistilGPT2")
    parser.add_argument("--client_id", type=int, required=True, help="ID du client")
    args = parser.parse_args()

    print(f"[INFO] Lancement du client Flower avec ID {args.client_id}")

    fl.client.start_numpy_client(
        server_address="192.168.1.170:8080",
        client=DistilGPT2Client(args.client_id)
    )
