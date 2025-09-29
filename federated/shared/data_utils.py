from datasets import Dataset
import pandas as pd
import os

def load_client_dataset_and_format(client_id: int, base_dir="client_dataset"):
    # Compose le nom du fichier correspondant au client
    filename = f"dataset_secutable_sepses_cea_train_split_{client_id}.csv"
    file_path = os.path.join(base_dir, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset {file_path} introuvable.")

    df = pd.read_csv(file_path)

    # Transformation en format instruct-tune
    def format_instruction(row):
        return f"""<s>
###Instruction: Use this agent to make semantic annotation of your datas
###Human: Generate wikidata URI of {row['label']}
###Assistant: The wikidata uri of {row['label']} is {row['entity']}
</s>"""

    formatted = [format_instruction(row) for _, row in df.iterrows()]
    
    # Important : conversion en HuggingFace Dataset (champ: 'text')
    dataset = Dataset.from_dict({"text": formatted})
    return dataset
