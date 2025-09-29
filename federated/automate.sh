#!/bin/bash

CLIENTS=(
    #"borista@192.168.1.126 /home/borista/Desktop/Expe-patricia"
    "ingenieur@192.168.1.140 /home/ingenieur/Desktop/Expe-patricia"
    "chinjoyce@192.168.1.151 /home/chinjoyce/Desktop/Expe-patricia"
    "emmanuel@192.168.1.111  /home/chinjoyce/Desktop/Expe-patricia"
)

PROJECT_DIR="./Expe-patricia"
SSH_KEY="/home/patricia/.ssh/id_rsa"
ENV_NAME="federated_env"
ENV_FILE="environment.yml"

# === ÉTAPE 0 : Export de l’environnement Conda local ===
echo "Export de l'environnement Conda local..."
conda env export > "$ENV_FILE"

# === ÉTAPE 1 : Copie du projet et de l’environnement vers les clients ===
echo "Copie du projet et de l’environnement vers les clients..."
for entry in "${CLIENTS[@]}"; do
  IFS=' ' read -r CLIENT TARGET_PATH <<< "$entry"
  echo "Copie vers $CLIENT:$TARGET_PATH"
  
  ssh -i $SSH_KEY "$CLIENT" "rm -rf $TARGET_PATH"
  scp -i $SSH_KEY -r "$PROJECT_DIR" "$CLIENT:$TARGET_PATH"
  scp -i $SSH_KEY "$ENV_FILE" "$CLIENT:$TARGET_PATH/"
done

# === ÉTAPE 2 : Création de l’environnement Conda sur les clients ===
echo "Création de l’environnement Conda sur les clients (si nécessaire)..."
for entry in "${CLIENTS[@]}"; do
  IFS=' ' read -r CLIENT TARGET_PATH <<< "$entry"
  echo "Préparation de l'environnement sur $CLIENT..."

  ssh -i $SSH_KEY "$CLIENT" "bash -c '
    source ~/.bashrc
    if ! conda info --envs | grep -q \"$ENV_NAME\"; then
      echo \"Création de l’environnement $ENV_NAME...\"
      conda env create -f $TARGET_PATH/$ENV_FILE -n $ENV_NAME
    else
      echo \"Environnement $ENV_NAME déjà présent.\"
    fi
  '"
done

# === ÉTAPE 3 : Lancement des clients avec activation de l’environnement ===
echo "Lancement des clients..."
for entry in "${CLIENTS[@]}"; do
  IFS=' ' read -r CLIENT TARGET_PATH <<< "$entry"
  echo "Démarrage de $CLIENT..."
  ssh -i $SSH_KEY "$CLIENT" "bash -c '
    source ~/.bashrc
    conda activate $ENV_NAME
    cd $TARGET_PATH
    conda run -n $ENV_NAME python client.py
  '" &
done

# === ÉTAPE 4 : Lancement du serveur en local ===
echo "Lancement du serveur sur cette machine..."
conda run -n $ENV_NAME python "$PROJECT_DIR/server.py"

