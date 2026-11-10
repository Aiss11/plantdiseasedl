#!/usr/bin/env python3
"""Télécharge les données du projet : PlantVillage et PlantDoc (classes tomate).

Les deux jeux de données sont des dépôts GitHub publics. Pour éviter de
télécharger ~3 Go inutiles (PlantVillage contient toutes les cultures), le
script utilise un clone partiel + « sparse checkout » et ne récupère que les
dossiers de tomate.

Disposition créée (identique à celle attendue par le notebook) :

    <dest>/PlantDoc/train/Tomato .../
    <dest>/PlantDoc/test/Tomato .../
    <dest>/PlantVillage_repo/raw/color/Tomato___.../

Usage :
    python scripts/download_data.py                  # -> ./data
    python scripts/download_data.py --dest /content/data   # Colab
    python scripts/download_data.py --full           # clone complet (~3 Go)
    python scripts/download_data.py --verify-only    # vérifie sans télécharger

Prérequis : git (>= 2.25) et une connexion à GitHub. Aucun paquet Python externe.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PLANTDOC_URL = "https://github.com/pratikkayal/PlantDoc-Dataset.git"
PLANTVILLAGE_URL = "https://github.com/spMohanty/PlantVillage-Dataset.git"

# Motifs « sparse checkout » (mode non-cone) : uniquement la tomate.
PLANTDOC_PATTERNS = ["/train/Tomato*/", "/test/Tomato*/"]
PLANTVILLAGE_PATTERNS = ["/raw/color/Tomato___*/"]

# Correspondance des 7 classes communes (identique au notebook, section 3).
# Spider_mites est écartée du projet (trop peu d'images dans PlantDoc).
CLASS_MAP = {
    "Bacterial_spot": ("Tomato___Bacterial_spot", "Tomato leaf bacterial spot"),
    "Early_blight": ("Tomato___Early_blight", "Tomato Early blight leaf"),
    "Late_blight": ("Tomato___Late_blight", "Tomato leaf late blight"),
    "Leaf_Mold": ("Tomato___Leaf_Mold", "Tomato mold leaf"),
    "Septoria_leaf_spot": ("Tomato___Septoria_leaf_spot", "Tomato Septoria leaf spot"),
    "Tomato_mosaic_virus": ("Tomato___Tomato_mosaic_virus", "Tomato leaf mosaic virus"),
    "Yellow_Leaf_Curl_Virus": ("Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato leaf yellow virus"),
}
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def run(cmd, cwd=None, stdin=None):
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, input=stdin, text=True, check=True)


def clone(url, target: Path, patterns, full: bool):
    if (target / ".git").is_dir():
        print(f"[ok] {target} existe déjà, téléchargement ignoré.")
        return
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if full:
        run(["git", "clone", "--depth", "1", url, str(target)])
        return
    run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", url, str(target)])
    run(["git", "sparse-checkout", "set", "--no-cone", "--stdin"],
        cwd=target, stdin="\n".join(patterns) + "\n")


def count_images(folder: Path) -> int:
    if not folder.is_dir():
        return -1
    return sum(1 for f in folder.iterdir() if f.suffix.lower() in IMG_EXT)


def verify(dest: Path) -> bool:
    pv = dest / "PlantVillage_repo" / "raw" / "color"
    pdoc = dest / "PlantDoc"
    ok = True
    print(f"\n{'Classe':<24}{'PlantVillage':>14}{'PlantDoc train':>16}{'PlantDoc test':>15}")
    for name, (pv_dir, pd_dir) in CLASS_MAP.items():
        n_pv = count_images(pv / pv_dir)
        n_tr = count_images(pdoc / "train" / pd_dir)
        n_te = count_images(pdoc / "test" / pd_dir)
        print(f"{name:<24}{n_pv:>14}{n_tr:>16}{n_te:>15}")
        ok &= n_pv > 0 and n_tr > 0 and n_te > 0
    print("\nVérification :", "OK" if ok else "ÉCHEC (dossiers manquants, voir -1 ci-dessus)")
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest", default="data", help="dossier de destination (défaut : ./data)")
    p.add_argument("--full", action="store_true", help="clone complet (~3 Go) au lieu de la tomate seule")
    p.add_argument("--verify-only", action="store_true", help="ne télécharge rien, vérifie seulement")
    args = p.parse_args()
    dest = Path(args.dest).resolve()

    if not args.verify_only:
        if shutil.which("git") is None:
            sys.exit("git est introuvable : installez-le puis relancez.")
        clone(PLANTDOC_URL, dest / "PlantDoc", PLANTDOC_PATTERNS, args.full)
        clone(PLANTVILLAGE_URL, dest / "PlantVillage_repo", PLANTVILLAGE_PATTERNS, args.full)
    sys.exit(0 if verify(dest) else 1)


if __name__ == "__main__":
    main()
