#!/bin/bash
# Script para baixar ISIC 2020 dataset (Dermatoscopia)
# Uso: bash download_data.sh

mkdir -p ./data/isic2020

# Baixa imagens de treino (JPEG)
aws s3 cp --no-sign-request s3://isic-archive/challenges/2020/ISIC_2020_Training_JPEG.zip ./data/isic2020/

# Baixa labels de treino
aws s3 cp --no-sign-request s3://isic-archive/challenges/2020/ISIC_2020_Training_GroundTruth.csv ./data/isic2020/
aws s3 cp --no-sign-request s3://isic-archive/challenges/2020/ISIC_2020_Training_GroundTruth_v2.csv ./data/isic2020/

# Baixa imagens de teste (JPEG)
aws s3 cp --no-sign-request s3://isic-archive/challenges/2020/ISIC_2020_Test_JPEG.zip ./data/isic2020/

# Baixa metadados de teste
aws s3 cp --no-sign-request s3://isic-archive/challenges/2020/ISIC_2020_Test_Metadata.csv ./data/isic2020/

echo "Download concluído. Agora extraia os arquivos ZIP:"
echo "unzip ./data/isic2020/ISIC_2020_Training_JPEG.zip -d ./data/isic2020/train"
echo "unzip ./data/isic2020/ISIC_2020_Test_JPEG.zip -d ./data/isic2020/test"
