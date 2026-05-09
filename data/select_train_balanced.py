"""
select_train_balanced.py
========================
Lê ISIC_2020_Training_GroundTruth.csv, seleciona de forma aleatória e
equilibrada 15.000 imagens (7.500 por classe de target), copia os arquivos
encontrados na pasta train/ para train_selecionado/ e gera um novo CSV
com os registros selecionados.

Passos executados
-----------------
1. Lê ISIC_2020_Training_GroundTruth.csv.
2. Seleciona 15.000 imagens balanceadas (7.500 com target=0 e 7.500 com
   target=1).
3. Para cada imagem selecionada, busca o arquivo na pasta train/ (suporta
   .jpg, .jpeg, .png, .tif, .tiff, .bmp). Se não encontrar, descarta e
   substitui por outra imagem da mesma classe.
4. Cria a pasta train_selecionado/ (se não existir) e copia as imagens.
5. Renomeia o CSV original acrescentando _bkup ao nome.
6. Grava um novo CSV com o nome original contendo apenas os registros
   selecionados.

Uso
---
    # A partir da raiz do projeto:
    python data/select_train_balanced.py

    # Parâmetros opcionais:
    python data/select_train_balanced.py \\
        --data-dir data \\
        --images-dir data/isic2020/train \\
        --output-dir data/isic2020/train_selecionado \\
        --total 15000 \\
        --seed 42
"""

import argparse
import shutil
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def find_image(image_name: str, train_dir: Path) -> Path | None:
    """Procura image_name (sem ou com extensão) dentro de train_dir.

    Retorna o caminho completo do arquivo se encontrado, ou None.
    """
    stem = Path(image_name).stem  # remove extensão se já veio com ela
    for ext in IMAGE_EXTENSIONS:
        candidate = train_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


def backup_csv(csv_path: Path) -> Path:
    """Renomeia o CSV original inserindo _bkup antes da extensão."""
    bkup_path = csv_path.with_name(f"{csv_path.stem}_bkup{csv_path.suffix}")
    if bkup_path.exists():
        print(f"  [aviso] Backup já existe e será sobrescrito: {bkup_path.name}")
    csv_path.rename(bkup_path)
    print(f"  Renomeado: {csv_path.name} → {bkup_path.name}")
    return bkup_path


def select_balanced_with_files(
    df: pd.DataFrame,
    img_col: str,
    target_col: str,
    train_dir: Path,
    n_total: int,
    seed: int,
) -> tuple[pd.DataFrame, list[Path]]:
    """Seleciona n_total registros balanceados verificando a existência dos
    arquivos de imagem em train_dir.

    Para cada classe (0 e 1) tenta selecionar n_total // 2 registros com
    arquivo disponível. Caso não encontre o arquivo de uma imagem sorteada,
    descarta-a e tenta a próxima da mesma classe (sem reposição).

    Retorna:
        selected_df : DataFrame com os registros selecionados
        image_paths : lista de Path dos arquivos encontrados (mesma ordem)
    """
    n_per_class = n_total // 2
    rng = pd.Series(dtype=object)  # só para satisfazer o linter

    selected_rows: list[pd.Series] = []
    found_paths: list[Path] = []

    for class_val in (0, 1):
        class_df = df[df[target_col] == class_val].sample(frac=1, random_state=seed)
        collected = 0
        discarded = 0

        for _, row in class_df.iterrows():
            if collected >= n_per_class:
                break
            img_path = find_image(str(row[img_col]), train_dir)
            if img_path is None:
                discarded += 1
                continue
            selected_rows.append(row)
            found_paths.append(img_path)
            collected += 1

        available = len(class_df) - discarded
        print(
            f"  target={class_val}: {collected}/{n_per_class} selecionados "
            f"| {discarded} descartados (arquivo não encontrado) "
            f"| {available} disponíveis no total"
        )
        if collected < n_per_class:
            print(
                f"  [aviso] Não foi possível atingir {n_per_class} imagens para "
                f"target={class_val}. Apenas {collected} disponíveis."
            )

    selected_df = pd.DataFrame(selected_rows).reset_index(drop=True)
    return selected_df, found_paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    _script_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(
        description="Seleciona 15.000 imagens balanceadas do ISIC 2020."
    )
    parser.add_argument(
        "--data-dir",
        default=str(_script_dir),
        help="Pasta onde está o CSV de treino (padrão: diretório do script)",
    )
    parser.add_argument(
        "--images-dir",
        default=str(_script_dir / "isic2020" / "train"),
        help="Pasta com as imagens de treino (padrão: <data-dir>/isic2020/train)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_script_dir / "isic2020" / "train_selecionado"),
        help=(
            "Pasta de destino para as imagens selecionadas "
            "(padrão: <data-dir>/isic2020/train_selecionado)"
        ),
    )
    parser.add_argument(
        "--total",
        type=int,
        default=15_000,
        help="Total de imagens a selecionar (padrão: 15000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semente para reprodutibilidade (padrão: 42)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    n_total = args.total
    seed = args.seed

    # -----------------------------------------------------------------------
    # 1. Lê o CSV
    # -----------------------------------------------------------------------
    train_csv = data_dir / "ISIC_2020_Training_GroundTruth.csv"
    print(f"\n{'='*60}")
    print(f"Lendo: {train_csv}")
    print("=" * 60)

    if not train_csv.exists():
        raise FileNotFoundError(f"CSV não encontrado: {train_csv}")

    df = pd.read_csv(train_csv)
    print(f"  Total de registros: {len(df)}")

    if "target" not in df.columns:
        raise KeyError(
            f"Coluna 'target' não encontrada. Colunas disponíveis: {df.columns.tolist()}"
        )

    img_col = "image_name" if "image_name" in df.columns else "image"
    print(f"  Coluna de imagem: '{img_col}'")
    print(f"  Distribuição de target:\n{df['target'].value_counts().to_string()}")

    # -----------------------------------------------------------------------
    # 2. Valida pasta de imagens
    # -----------------------------------------------------------------------
    if not train_dir.exists():
        raise FileNotFoundError(
            f"Pasta de imagens não encontrada: {train_dir}\n"
            "Verifique o parâmetro --images-dir."
        )

    # -----------------------------------------------------------------------
    # 3. Cria pasta de saída
    # -----------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Pasta de saída: {output_dir}")

    # -----------------------------------------------------------------------
    # 4. Seleção balanceada verificando existência dos arquivos
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Selecionando {n_total} imagens balanceadas...")
    print("=" * 60)

    selected_df, image_paths = select_balanced_with_files(
        df=df,
        img_col=img_col,
        target_col="target",
        train_dir=train_dir,
        n_total=n_total,
        seed=seed,
    )

    print(f"\n  Total selecionado: {len(selected_df)} imagens")

    # -----------------------------------------------------------------------
    # 5. Copia imagens para train_selecionado/
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Copiando imagens para {output_dir.name}/...")
    print("=" * 60)

    for i, src_path in enumerate(image_paths, 1):
        dst_path = output_dir / src_path.name
        shutil.copy2(src_path, dst_path)
        if i % 1000 == 0 or i == len(image_paths):
            print(f"  Copiadas: {i}/{len(image_paths)}")

    # -----------------------------------------------------------------------
    # 6. Renomeia CSV original e grava novo CSV
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Atualizando CSV...")
    print("=" * 60)

    bkup_path = backup_csv(train_csv)
    selected_df.to_csv(train_csv, index=False)
    print(f"  Novo CSV gravado: {train_csv.name}  ({len(selected_df)} registros)")

    # -----------------------------------------------------------------------
    # Resumo
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Concluído!")
    print(f"  Backup CSV    : {bkup_path}")
    print(f"  Novo CSV      : {train_csv}")
    print(f"  Imagens copiadas: {len(image_paths)}")
    print(f"  Pasta destino : {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
