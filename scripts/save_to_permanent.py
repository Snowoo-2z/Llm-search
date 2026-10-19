#!/usr/bin/env python3
"""
Sauvegarde permanente BlackCode-1B 0 trace après fine-tune TPU
Kaggle Output = éphémère (supprimé si notebook supprimé, pas 50 ans)
Solutions permanentes: HuggingFace Hub + Kaggle Dataset + Google Drive

Usage:
  python scripts/save_to_permanent.py --model runs/blackcode_1b_0trace --hf-repo BlackCode-Team/BlackCode-1B --kaggle-dataset blackcode-1b-0trace --drive
"""

import argparse, os, shutil, subprocess
from pathlib import Path

def save_to_hf(model_path, repo_id, private=True):
    print(f"[HF Hub] push {model_path} → {repo_id} private={private}")
    try:
        from huggingface_hub import HfApi, create_repo
        api = HfApi()
        create_repo(repo_id, private=private, exist_ok=True)
        api.upload_folder(
            folder_path=str(model_path),
            repo_id=repo_id,
            commit_message="BlackCode-1B 0 trace — entraîné de zéro",
        )
        print(f"[HF Hub] OK https://huggingface.co/{repo_id}")
        return True
    except Exception as e:
        print(f"[HF Hub] échec {e} — installe huggingface_hub et HF_TOKEN")
        print(f"  pip install huggingface_hub")
        print(f"  huggingface-cli login")
        return False

def save_to_kaggle_dataset(model_path, dataset_slug):
    print(f"[Kaggle Dataset] création dataset {dataset_slug} depuis {model_path}")
    # Kaggle dataset metadata
    dataset_dir = Path("/kaggle/working") / dataset_slug
    dataset_dir.mkdir(parents=True, exist_ok=True)
    # Copy model
    for f in Path(model_path).glob("*"):
        if f.is_file():
            shutil.copy2(f, dataset_dir / f.name)
    # Create dataset-metadata.json
    metadata = f"""{{
  "title": "{dataset_slug}",
  "id": "YOUR_USERNAME/{dataset_slug}",
  "licenses": [{{"name": "apache-2-0"}}],
  "resources": [{{"path": "{dataset_slug}"}}]
}}"""
    (Path("/kaggle/working") / "dataset-metadata.json").write_text(metadata)
    print(f"[Kaggle Dataset] Pour uploader:")
    print(f"  !kaggle datasets create -p /kaggle/working --dir-mode tar")
    print(f"  Ou via UI: Add Data → New Dataset → Upload {dataset_dir}")
    print(f"  Dataset restera permanent dans ton compte Kaggle")
    return True

def save_to_drive_kaggle(model_path):
    print(f"[Google Drive] depuis Kaggle TPU")
    print(f"  Kaggle n'a pas mount Drive natif comme Colab, mais tu peux:")
    print(f"  1. Utiliser pydrive2 avec token")
    print(f"  2. Ou utiliser rclone + service account")
    print(f"  3. Ou plus simple: après download sur ton PC, tu upload sur Drive manuellement")
    print(f"")
    print(f"  Méthode la plus simple (recommandée):")
    print(f"  - Download blackcode-1b-0trace-ollama.tar.gz depuis Kaggle Output")
    print(f"  - Sur ton PC: upload sur Google Drive → dossier BlackCode-1B")
    print(f"  - Drive garde 15Go gratuit, permanent")
    print(f"")
    print(f"  Méthode auto depuis Kaggle (si tu veux):")
    code = '''
# Dans notebook Kaggle TPU cell:
!pip install -q pydrive2
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from google.colab import auth  # pas dispo sur Kaggle, utiliser auth manuelle

# Alternative: utiliser HF Hub comme intermédiaire permanent (mieux que Drive)
# HF Hub = gratuit, illimité pour modèles, versionné, permanent
'''
    print(code)
    # Create tar.gz for Drive upload
    tar_path = Path("/kaggle/working") / "blackcode-1b-0trace-drive.tar.gz"
    subprocess.run(["tar", "-czf", str(tar_path), "-C", str(Path(model_path).parent), Path(model_path).name], check=False)
    print(f"[Drive] tar.gz créé {tar_path} ({tar_path.stat().st_size/1e9:.2f} Go) prêt pour upload manuel Drive")
    return True

def save_to_drive_colab(model_path):
    print(f"[Google Drive] depuis Colab (si tu utilises Colab TPU)")
    code = '''
# Dans Colab:
from google.colab import drive
drive.mount('/content/drive')
!cp -r runs/blackcode_1b_0trace /content/drive/MyDrive/BlackCode-1B-0trace
!tar -czf /content/drive/MyDrive/blackcode-1b-0trace-ollama.tar.gz -C runs blackcode_1b_0trace
print("Sauvegardé sur Drive: /content/drive/MyDrive/BlackCode-1B-0trace")
'''
    print(code)
    return True

def main():
    p = argparse.ArgumentParser(description="Sauvegarde permanente BlackCode")
    p.add_argument("--model", type=str, required=True, help="chemin modèle 0 trace")
    p.add_argument("--hf-repo", type=str, default=None, help="HF repo id ex: BlackCode-Team/BlackCode-1B")
    p.add_argument("--kaggle-dataset", type=str, default=None, help="slug kaggle dataset")
    p.add_argument("--drive", action="store_true", help="préparer sauvegarde Drive")
    p.add_argument("--colab-drive", action="store_true", help="sauvegarde Drive depuis Colab")
    args = p.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[error] modèle {model_path} n'existe pas")
        return

    if args.hf_repo:
        save_to_hf(model_path, args.hf_repo, private=True)

    if args.kaggle_dataset:
        save_to_kaggle_dataset(model_path, args.kaggle_dataset)

    if args.drive:
        save_to_drive_kaggle(model_path)

    if args.colab_drive:
        save_to_drive_colab(model_path)

    if not any([args.hf_repo, args.kaggle_dataset, args.drive, args.colab_drive]):
        print("[info] Aucune destination spécifiée, je fais tout:")
        save_to_hf(model_path, "BlackCode-Team/BlackCode-1B", private=True)
        save_to_kaggle_dataset(model_path, "blackcode-1b-0trace")
        save_to_drive_kaggle(model_path)

if __name__ == "__main__":
    main()
