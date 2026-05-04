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

# Categorias de localização anatômica — mesma ordem usada no treino.
ANATOM_SITE_CATS = [
    "torso", "lower extremity", "upper extremity",
    "head/neck", "palms/soles", "oral/genital",
]
# dim do vetor de metadados: 1 (age_norm) + len(ANATOM_SITE_CATS)
NUM_META = 1 + len(ANATOM_SITE_CATS)  # = 7


def _build_meta_tensor(age_approx: float | None, anatom_site: str | None) -> torch.Tensor:
    """Constrói o tensor de metadados clínicos normalizado.

    Args:
        age_approx: Idade aproximada do paciente (anos). ``None`` usa 0.5 como
            valor de preenchimento (equivale à mediana ~45 anos / 90).
        anatom_site: Localização anatômica da lesão. Deve ser um dos valores
            de :data:`ANATOM_SITE_CATS`; ``None`` ou valor desconhecido resulta
            em vetor one-hot todo zero.

    Returns:
        Tensor 1-D de shape ``(NUM_META,)`` com ``dtype=float32``.
    """
    age_norm = (age_approx / 90.0) if age_approx is not None else 0.5
    age_norm = float(np.clip(age_norm, 0.0, 1.0))

    one_hot = [1.0 if (anatom_site or "").strip() == cat else 0.0
               for cat in ANATOM_SITE_CATS]

    return torch.tensor([age_norm] + one_hot, dtype=torch.float32)


class MelanomaClassifier(nn.Module):
    """EfficientNet-B4 multi-modal com metadados clínicos.

    Combina features visuais (backbone EfficientNet-B4) com metadados
    tabulares (age_approx normalizada + one-hot de anatom_site) por
    concatenação antes da cabeça de classificação.
    """

    def __init__(self, model_name: str = "efficientnet_b4", drop_rate: float = 0.3,
                 num_meta: int = NUM_META):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            drop_rate=drop_rate,
        )
        in_features = self.backbone.num_features

        # Ramo de metadados: MLP pequeno
        self.meta_branch = nn.Sequential(
            nn.Linear(num_meta, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.2),
        )

        # Cabeça de classificação (img features + meta features)
        self.head = nn.Sequential(
            nn.Linear(in_features + 64, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
        )

    def forward(self, x, meta):
        img_features  = self.backbone(x)        # (B, in_features)
        meta_features = self.meta_branch(meta)  # (B, 64)
        combined = torch.cat([img_features, meta_features], dim=1)
        return self.head(combined).squeeze(1)   # (B,)


def _get_val_transform():
    return A.Compose([
        A.Resize(height=IMG_SIZE, width=IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def _get_tta_transform():
    return A.Compose([
        A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.85, 1.0)),
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
    age_approx: float | None = None,
    anatom_site: str | None = None,
    tta_steps: int = TTA_STEPS,
) -> dict:
    """Executa inferência multi-modal com TTA sobre uma imagem de lesão cutânea.

    Combina features visuais (EfficientNet-B4) com metadados clínicos
    (``age_approx`` e ``anatom_site``) para produzir a predição final.

    O primeiro passo usa a transformação de validação (resize + normalize)
    e os passos seguintes usam TTA (random crop + flip + normalize) para
    aumentar a robustez da predição, conforme definido no notebook de treino.

    Args:
        image_bytes: Bytes brutos de uma imagem JPEG ou PNG.
        model: Modelo carregado com :func:`load_melanoma_model`.
        age_approx: Idade aproximada do paciente em anos.  ``None`` usa o
            valor de preenchimento padrão (mediana ~45 anos).
        anatom_site: Localização anatômica da lesão — um dos valores de
            :data:`ANATOM_SITE_CATS` (ex.: ``"torso"``, ``"head/neck"``).
            ``None`` ou valor não reconhecido resultam em vetor one-hot todo
            zero (nenhuma categoria selecionada).
        tta_steps: Número de passes de TTA (1–10, padrão 5).

    Returns:
        Dicionário com as chaves:

        - ``probabilidade_melanoma`` (float): Probabilidade de melanoma (0–1).
        - ``classificacao`` (str): ``"maligno"`` ou ``"benigno"``.
        - ``alerta`` (bool): ``True`` se probabilidade ≥ 0.5.
        - ``age_approx`` (float | None): Idade utilizada na predição.
        - ``anatom_site`` (str | None): Localização anatômica utilizada.
    """
    image = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
    meta = _build_meta_tensor(age_approx, anatom_site).unsqueeze(0).to(DEVICE)

    prob_sum = 0.0
    for t in range(tta_steps):
        transform = _get_tta_transform() if t > 0 else _get_val_transform()
        tensor = transform(image=image)["image"].unsqueeze(0).to(DEVICE)
        logit = model(tensor, meta)
        prob = torch.sigmoid(logit).item()
        prob_sum += prob

    prob = prob_sum / tta_steps
    label = int(prob >= CLASSIFICATION_THRESHOLD)
    return {
        "probabilidade_melanoma": round(prob, 4),
        "classificacao": "maligno" if label else "benigno",
        "alerta": bool(prob >= CLASSIFICATION_THRESHOLD),
        "age_approx": age_approx,
        "anatom_site": anatom_site,
    }
