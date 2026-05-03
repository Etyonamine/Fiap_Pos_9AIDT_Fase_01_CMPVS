# Integração do endpoint `/predict/melanoma` na API

Este diretório contém todos os arquivos necessários para adicionar o endpoint
de classificação de melanoma (ISIC 2020 – EfficientNet-B4) ao repositório
[Fiap_Pos_9AIDT_Fase_01_API](https://github.com/Etyonamine/Fiap_Pos_9AIDT_Fase_01_API).

## Arquivos

| Arquivo | Destino na API | Ação |
|---------|----------------|------|
| `predict_melanoma.py` | raiz do projeto | **copiar** |
| `schemas/output_schema_melanoma.py` | `schemas/` | **copiar** |
| `app_melanoma_additions.py` | `app.py` | **integrar** (ver instruções abaixo) |
| `requirements_melanoma.txt` | `requirements.txt` | **acrescentar** as linhas |

## Passo a passo

### 1. Copiar arquivos

```bash
cp predict_melanoma.py           ../Fiap_Pos_9AIDT_Fase_01_API/
cp schemas/output_schema_melanoma.py  ../Fiap_Pos_9AIDT_Fase_01_API/schemas/
```

### 2. Atualizar `requirements.txt`

Acrescentar ao final do `requirements.txt` da API:

```
torch>=2.1.0
torchvision>=0.16.0
timm>=0.9.0
albumentations>=1.3.0
Pillow>=10.0.0
```

### 3. Integrar em `app.py`

O arquivo `app_melanoma_additions.py` contém instruções comentadas (pontos 1–4)
de exatamente onde inserir cada trecho. Em resumo:

**a) Imports** (no topo, após os imports existentes):
```python
from schemas.output_schema_melanoma import output_schema_melanoma
from predict_melanoma import load_melanoma_model, predict_melanoma as run_predict_melanoma
```

**b) Definição Swagger** (dentro de `swagger_template["definitions"]`):
```python
"OutputModelMelanoma": output_schema_melanoma,
```

**c) Carregamento do modelo** (após o bloco `try/except` que carrega os joblib):
```python
_MELANOMA_CKPT = "model/efficientnet_b4_melanoma.pth"
melanoma_model = None
try:
    melanoma_model = load_melanoma_model(_MELANOMA_CKPT)
except FileNotFoundError:
    logger.warning(
        "Checkpoint do modelo de melanoma não encontrado em %s. "
        "O endpoint /predict/melanoma estará indisponível.",
        _MELANOMA_CKPT,
    )
```

**d) Endpoint** (após o endpoint `/predict` existente):  
Copiar a função `predict_melanoma_endpoint` de `app_melanoma_additions.py`.

### 4. Copiar o checkpoint do modelo

O arquivo `.pth` gerado pelo treino no notebook (`outputs/best_fold0.pth`)
deve ser copiado para:

```
Fiap_Pos_9AIDT_Fase_01_API/model/efficientnet_b4_melanoma.pth
```

### 5. Testar o endpoint

```bash
# com um arquivo de imagem real
curl -X POST http://localhost:5000/predict/melanoma \
     -F "image=@lesao.jpg" \
     -F "tta_steps=5"

# resposta esperada
{
  "probabilidade_melanoma": 0.7821,
  "classificacao": "maligno",
  "alerta": true
}
```

## Arquitetura do modelo

```
Input (imagem RGB 380×380)
  └── EfficientNet-B4 backbone (timm, pesos treinados no ISIC 2020)
       └── Global average pool → 1792 features
            └── Linear(1792→512) → BN → SiLU → Dropout(0.3)
                 └── Linear(512→1) → sigmoid → probabilidade de melanoma
```

**TTA (Test-Time Augmentation):** 5 passes com random crop + flip horizontal/vertical,
resultando em média das probabilidades para maior robustez.

> **Nota:** O treino foi realizado apenas com pacientes do sexo feminino
> (filtro `sex == 'female'` no notebook). Leve isso em conta ao interpretar
> as predições para outros públicos.
