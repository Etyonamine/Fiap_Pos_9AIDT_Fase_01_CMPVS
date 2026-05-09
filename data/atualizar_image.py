import os
import shutil
import pandas as pd

# Caminhos dos arquivos CSV
arquivo1 = "ISIC_2020_Training_GroundTruth.csv"
arquivo2 = "ISIC_2020_Test_Metadata.csv"

# Caminhos das pastas
train_dir = os.path.join("data", "train")
test_dir = os.path.join("data", "test")
backup_train_dir = os.path.join("data", "backup-image", "train")
backup_test_dir = os.path.join("data", "backup-image", "test")

# Ler os arquivos CSV
df_train = pd.read_csv(arquivo1)
df_test = pd.read_csv(arquivo2)

# Capturar nomes das imagens (sem extensão)
train_images = df_train["image_name"].astype(str).tolist()
test_images = df_test["image"].astype(str).tolist()

def verificar_e_copiar(imagens, destino, backup):
    for img in imagens:
        filename = f"{img}.jpeg"
        destino_path = os.path.join(destino, filename)
        backup_path = os.path.join(backup, filename)

        # Verifica se existe no destino
        if not os.path.exists(destino_path):
            # Se não existe, tenta copiar do backup
            if os.path.exists(backup_path):
                shutil.copy(backup_path, destino_path)
                print(f"Copiado: {filename} de {backup} para {destino}")
            else:
                print(f"Arquivo ausente: {filename} (não encontrado nem no backup)")

# Verificar e copiar imagens de treino
verificar_e_copiar(train_images, train_dir, backup_train_dir)

# Verificar e copiar imagens de teste
verificar_e_copiar(test_images, test_dir, backup_test_dir)

print("Processo concluído.")
