"""
prepare_dataset.py
==================
Cria subconjuntos balanceados (1/3 do total, 50 % benigno / 50 % maligno)
a partir dos arquivos ISIC 2020 e limpa as pastas de imagens correspondentes.

Passos executados
-----------------
1. Lê ISIC_2020_Training_GroundTruth.csv e ISIC_2020_Test_Metadata.csv.
2. Para o arquivo de treino: seleciona 1/3 do total balanceando 50 % benign
   (target=0) e 50 % malignant (target=1).  Como a classe maligna é bem
   menor, o tamanho do lote fica limitado a 2 × n_malignant.
   Adicionalmente, quando as colunas age_approx e anatom_site_general_challenge
   estiverem presentes, a amostragem dentro de cada classe é estratificada
   por essas colunas, preservando a distribuição etária e por localização
   anatômica de cada classe.
3. Para o arquivo de teste: como não existe coluna target, seleciona
   aleatoriamente 1/3 do total.
4. Renomeia os CSVs originais acrescentando o sufixo _bkup ao nome.
5. Grava novos CSVs com os nomes originais contendo apenas as linhas
   selecionadas.
6. Nas pastas de imagens (data/isic2020/train e data/isic2020/test) mantém
   somente os arquivos cujo nome base (sem extensão) esteja na seleção,
   excluindo os demais.

Uso
---
    # A partir da raiz do projeto ou de qualquer diretório:
    python data/prepare_dataset.py

    # Parâmetros opcionais (caminhos relativos ou absolutos)
    python data/prepare_dataset.py --data-dir data --images-dir data/isic2020 --seed 42
"""

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def backup_csv(csv_path: Path) -> Path:
    """Renomeia o CSV original inserindo _bkup antes da extensão."""
    stem = csv_path.stem          # nome sem extensão
    suffix = csv_path.suffix      # extensão
    bkup_path = csv_path.with_name(f"{stem}_bkup{suffix}")
    if bkup_path.exists():
        print(f"  [aviso] backup já existe, será sobrescrito: {bkup_path.name}")
    csv_path.rename(bkup_path)
    print(f"  Renomeado: {csv_path.name} → {bkup_path.name}")
    return bkup_path


def _stratified_sample(class_df: pd.DataFrame, n: int, strat_cols: list, seed: int) -> pd.DataFrame:
    """
    Retorna exatamente n linhas de class_df amostrando proporcionalmente
    dentro de cada estrato definido pelas colunas em strat_cols.
    Grupos com poucos registros recebem pelo menos 1 amostra, desde que
    o número de estratos não exceda n (caso contrário, cada estrato recebe
    exatamente 1 amostra e o excedente é descartado aleatoriamente).
    """
    groups = list(class_df.groupby(strat_cols, dropna=False))
    n_strata = len(groups)

    if n_strata >= n:
        # Mais estratos do que amostras desejadas: 1 por estrato, depois corta
        sampled_parts = [grp.sample(n=1, random_state=seed) for _, grp in groups]
        result = pd.concat(sampled_parts).sample(n=n, random_state=seed)
        return result

    frac = n / len(class_df)
    sampled_parts = []

    for _, group in groups:
        n_g = max(1, round(len(group) * frac))
        sampled_parts.append(group.sample(n=min(n_g, len(group)), random_state=seed))

    result = pd.concat(sampled_parts)

    # Ajuste fino para garantir exatamente n linhas
    if len(result) > n:
        result = result.sample(n=n, random_state=seed)
    elif len(result) < n:
        remaining = class_df.loc[~class_df.index.isin(result.index)]
        extra_n = min(n - len(result), len(remaining))
        if extra_n > 0:
            result = pd.concat([result, remaining.sample(n=extra_n, random_state=seed)])

    return result


def sample_balanced(
    df: pd.DataFrame,
    target_col: str,
    seed: int,
    strat_cols: list | None = None,
) -> pd.DataFrame:
    """
    Seleciona 1/3 do total do dataframe com 50 % de cada classe (target 0/1).
    O tamanho real é limitado pela classe minoritária.

    Se strat_cols for fornecido, a amostragem dentro de cada classe é
    estratificada por essas colunas, preservando a distribuição proporcional
    de cada estrato (ex.: faixa etária e localização anatômica).
    """
    n_total = len(df)
    n_third = n_total // 6

    class_0 = df[df[target_col] == 0]
    class_1 = df[df[target_col] == 1]

    n_minority = min(len(class_0), len(class_1))
    n_per_class = min(n_third // 2, n_minority)

    if n_per_class == 0:
        raise ValueError("Não há amostras suficientes para balancear as classes.")

    if strat_cols:
        sampled_0 = _stratified_sample(class_0, n_per_class, strat_cols, seed)
        sampled_1 = _stratified_sample(class_1, n_per_class, strat_cols, seed)
    else:
        sampled_0 = class_0.sample(n=n_per_class, random_state=seed)
        sampled_1 = class_1.sample(n=n_per_class, random_state=seed)

    result = pd.concat([sampled_0, sampled_1]).sample(frac=1, random_state=seed)
    print(
        f"  Selecionados: {len(result)} linhas  "
        f"(benigno={len(sampled_0)}, maligno={len(sampled_1)}) "
        f"de {n_total} totais"
    )
    if strat_cols:
        print(f"  Estratificação adicional por: {strat_cols}")
    return result


def sample_random(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Seleciona aleatoriamente 1/3 do dataframe (sem coluna target)."""
    n_total = len(df)
    n_third = n_total // 6
    result = df.sample(n=n_third, random_state=seed)
    print(f"  Selecionados: {len(result)} linhas de {n_total} totais (seleção aleatória)")
    return result


def clean_image_folder(folder: Path, selected_names: set, dry_run: bool = False) -> None:
    """
    Remove da pasta 'folder' todos os arquivos de imagem cujo nome base
    (sem extensão) NÃO esteja em selected_names.
    """
    if not folder.exists():
        print(f"  [aviso] Pasta não encontrada, nada a limpar: {folder}")
        return

    image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    removed = 0
    kept = 0

    for file in sorted(folder.iterdir()):
        if file.suffix.lower() not in image_extensions:
            continue
        if file.stem in selected_names:
            kept += 1
        else:
            if not dry_run:
                file.unlink()
            removed += 1

    action = "Seriam removidos" if dry_run else "Removidos"
    print(f"  {action}: {removed} arquivos  |  Mantidos: {kept} arquivos  ({folder})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepara subconjuntos balanceados do ISIC 2020.")
    _script_dir = Path(__file__).parent
    parser.add_argument(
        "--data-dir",
        default=str(_script_dir),
        help="Pasta onde estão os CSVs (padrão: diretório do script)",
    )
    parser.add_argument(
        "--images-dir",
        default=str(_script_dir / "isic2020"),
        help="Pasta raiz das imagens com subpastas train/ e test/ (padrão: <data-dir>/isic2020)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semente para reprodutibilidade (padrão: 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simula a limpeza de imagens sem apagar arquivos",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    images_dir = Path(args.images_dir)
    seed = args.seed
    dry_run = args.dry_run

    # -----------------------------------------------------------------------
    # 1. Arquivo de TREINO
    # -----------------------------------------------------------------------
    train_csv = data_dir / "ISIC_2020_Training_GroundTruth.csv"
    print(f"\n{'='*60}")
    print(f"Processando treino: {train_csv}")
    print("="*60)

    df_train = pd.read_csv(train_csv)

    # Valida coluna target
    if "target" not in df_train.columns:
        raise KeyError(f"Coluna 'target' não encontrada em {train_csv}. Colunas: {df_train.columns.tolist()}")

    # Determina coluna de nome de imagem
    train_img_col = "image_name" if "image_name" in df_train.columns else "image"
    print(f"  Coluna de imagem: '{train_img_col}'")

    # Colunas extras para estratificação (usadas quando presentes no CSV)
    _candidate_strat_cols = ["age_approx", "anatom_site_general_challenge"]
    train_strat_cols = [c for c in _candidate_strat_cols if c in df_train.columns]
    if train_strat_cols:
        print(f"  Colunas de estratificação encontradas: {train_strat_cols}")

    df_train_selected = sample_balanced(
        df_train, target_col="target", seed=seed,
        strat_cols=train_strat_cols or None,
    )

    # Renomeia original e salva novo CSV
    bkup_train = backup_csv(train_csv)
    df_train_selected.to_csv(train_csv, index=False)
    print(f"  Novo CSV gravado: {train_csv.name}")

    # Limpa pasta de imagens de treino
    train_images_dir = images_dir / "train"
    selected_train_names = set(df_train_selected[train_img_col].astype(str))
    clean_image_folder(train_images_dir, selected_train_names, dry_run=dry_run)

    # -----------------------------------------------------------------------
    # 2. Arquivo de TESTE
    # -----------------------------------------------------------------------
    test_csv = data_dir / "ISIC_2020_Test_Metadata.csv"
    print(f"\n{'='*60}")
    print(f"Processando teste: {test_csv}")
    print("="*60)

    df_test = pd.read_csv(test_csv)

    # Determina coluna de nome de imagem
    test_img_col = "image_name" if "image_name" in df_test.columns else "image"
    print(f"  Coluna de imagem: '{test_img_col}'")

    if "target" in df_test.columns:
        print("  Coluna 'target' encontrada — usando seleção balanceada.")
        df_test_selected = sample_balanced(df_test, target_col="target", seed=seed)
    else:
        print("  Coluna 'target' ausente — usando seleção aleatória (1/3).")
        df_test_selected = sample_random(df_test, seed=seed)

    # Renomeia original e salva novo CSV
    bkup_test = backup_csv(test_csv)
    df_test_selected.to_csv(test_csv, index=False)
    print(f"  Novo CSV gravado: {test_csv.name}")

    # Limpa pasta de imagens de teste
    test_images_dir = images_dir / "test"
    selected_test_names = set(df_test_selected[test_img_col].astype(str))
    clean_image_folder(test_images_dir, selected_test_names, dry_run=dry_run)

    # -----------------------------------------------------------------------
    # Resumo
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Concluído!")
    print(f"  Backup treino : {bkup_train}")
    print(f"  Backup teste  : {bkup_test}")
    if dry_run:
        print("  [dry-run] Nenhum arquivo de imagem foi apagado.")
    print("="*60)


if __name__ == "__main__":
    main()
