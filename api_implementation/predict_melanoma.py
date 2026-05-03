import io
import numpy as np
import torch
import torch.nn as nn
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

IMG_SIZE = 380
TTA_STEPS = 5
CLASSIFICATION_THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MelanomaClassifier(nn.Module):
    """EfficientNet-B4 com cabeça de classificação binária (benigno × maligno)."""

    def __init__(self, model_name: str = "efficientnet_b4", drop_rate: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            drop_rate=drop_rate,
        )
        in_features = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features).squeeze(1)


def _get_val_transform():
    return A.Compose([
        A.Resize(height=IMG_SIZE, width=IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def _get_tta_transform():
    return A.Compose([
        A.RandomResizedCrop(IMG_SIZE, IMG_SIZE, scale=(0.85, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def load_melanoma_model(checkpoint_path: str) -> MelanomaClassifier:
    """Carrega o modelo a partir de um checkpoint .pth.

    Args:
        checkpoint_path: Caminho para o arquivo ``best_fold0.pth`` gerado
            pelo notebook de treino.

    Returns:
        Instância de :class:`MelanomaClassifier` com pesos carregados,
        em modo de avaliação.
    """
    model = MelanomaClassifier().to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


@torch.no_grad()
def predict_melanoma(
    image_bytes: bytes,
    model: MelanomaClassifier,
    tta_steps: int = TTA_STEPS,
) -> dict:
    """Executa inferência com TTA sobre uma imagem de lesão cutânea.

    O primeiro passo usa a transformação de validação (resize + normalize)
    e os passos seguintes usam TTA (random crop + flip + normalize) para
    aumentar a robustez da predição, conforme definido no notebook de treino.

    Args:
        image_bytes: Bytes brutos de uma imagem JPEG ou PNG.
        model: Modelo carregado com :func:`load_melanoma_model`.
        tta_steps: Número de passes de TTA (1–10, padrão 5).

    Returns:
        Dicionário com as chaves:
        - ``probabilidade_melanoma`` (float): Probabilidade de melanoma (0–1).
        - ``classificacao`` (str): ``"maligno"`` ou ``"benigno"``.
        - ``alerta`` (bool): ``True`` se probabilidade ≥ 0.5.
    """
    image = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))

    prob_sum = 0.0
    for t in range(tta_steps):
        transform = _get_tta_transform() if t > 0 else _get_val_transform()
        tensor = transform(image=image)["image"].unsqueeze(0).to(DEVICE)
        logit = model(tensor)
        prob = torch.sigmoid(logit).item()
        prob_sum += prob

    prob = prob_sum / tta_steps
    label = int(prob >= CLASSIFICATION_THRESHOLD)
    return {
        "probabilidade_melanoma": round(prob, 4),
        "classificacao": "maligno" if label else "benigno",
        "alerta": bool(prob >= CLASSIFICATION_THRESHOLD),
    }
